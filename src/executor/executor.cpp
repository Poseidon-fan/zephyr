#include "executor/executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

#include "common/exception.hpp"
#include "common/logger.hpp"
#include "executor/worker.hpp"

namespace zephyr::executor {

auto ExecutionLimits::Resolve(int64_t model_max_seq_len) const -> ExecutionLimits {
  constexpr auto max_index = std::numeric_limits<int32_t>::max();
  if (max_num_seqs_ == 0 || std::cmp_greater(max_num_seqs_, max_index) || max_num_output_tokens_ == 0 ||
      max_seq_len_ < 0 || model_max_seq_len <= 0) {
    throw ConfigurationException("execution requires positive sequence and output limits and a valid model context");
  }
  auto resolved = *this;
  resolved.max_seq_len_ = max_seq_len_ == 0 ? model_max_seq_len : max_seq_len_;
  if (resolved.max_seq_len_ > model_max_seq_len || resolved.max_seq_len_ > max_index) {
    throw ConfigurationException("execution context limit must fit the model context and INT32 indexing");
  }
  const auto context_length = static_cast<size_t>(resolved.max_seq_len_);
  if (max_num_seqs_ > std::numeric_limits<size_t>::max() / context_length) {
    throw ConfigurationException("execution sequence and context limits overflow the batch token capacity");
  }
  const auto full_batch_tokens = max_num_seqs_ * context_length;
  resolved.max_num_batched_tokens_ =
      max_num_batched_tokens_ == 0 ? full_batch_tokens : std::min(max_num_batched_tokens_, full_batch_tokens);
  if (std::cmp_greater(resolved.max_num_batched_tokens_, max_index)) {
    throw ConfigurationException("execution batch token limit exceeds INT32 indexing");
  }
  resolved.max_num_output_tokens_ = std::min(max_num_output_tokens_, resolved.max_num_batched_tokens_);
  return resolved;
}

Executor::Executor(ttl::Runtime &runtime, const ExecutorOptions &options, ExecutionFactory factory)
    : state_(std::make_unique<WorkerState>(runtime, options, std::move(factory))) {}

auto Executor::Create(ttl::Runtime &runtime, const ExecutorOptions &options, ExecutionFactory factory)
    -> std::unique_ptr<Executor> {
  if (!factory) {
    throw ConfigurationException("executor requires an execution factory");
  }
  const auto &limits = options.execution_limits_;
  if (limits.max_num_seqs_ == 0 || limits.max_num_output_tokens_ == 0 || limits.max_seq_len_ < 0) {
    throw ConfigurationException(
        "executor requires positive sequence and output limits and a nonnegative context limit");
  }
  if (!std::isfinite(options.gpu_memory_utilization_) || options.gpu_memory_utilization_ <= 0.0 ||
      options.gpu_memory_utilization_ > 1.0) {
    throw ConfigurationException("GPU memory utilization must be finite and in (0, 1]");
  }
  const auto initialize_start = std::chrono::steady_clock::now();
  ZEPHYR_LOG_INFO("Initializing tensor-parallel executor with {} rank(s), dtype={}", options.devices_.size(),
                  ttl::GetDTypeInfo(options.dtype_).name_);
  auto executor = std::unique_ptr<Executor>{new Executor{runtime, options, std::move(factory)}};
  auto &state = *executor->state_;
  try {
    executor->workers_.reserve(options.devices_.size());
    for (size_t rank = 0; rank < options.devices_.size(); ++rank) {
      executor->workers_.emplace_back([&state, rank] { RunWorker(state, rank); });
    }
    executor->WaitForWorkers();
    {
      const std::scoped_lock lock{state.latch_};
      state.completed_ = 0;
      state.initialization_started_ = true;
    }
    state.changed_.notify_all();
    executor->WaitForWorkers();
    const auto &processor = *state.processors_.front();
    const auto &spec = processor.GetSpec();
    for (size_t rank = 0; rank < state.processors_.size(); ++rank) {
      const auto &rank_processor = *state.processors_[rank];
      const auto &rank_spec = rank_processor.GetSpec();
      if (rank_spec.device_ != options.devices_[rank] || rank_spec.dtype_ != spec.dtype_ ||
          rank_spec.limits_ != spec.limits_ || !processor.IsCompatible(rank_processor)) {
        throw ConfigurationException("tensor-parallel ranks must report consistent execution capabilities");
      }
    }
    executor->processor_ = std::move(state.processors_.front());
    state.processors_.clear();
    state.factory_ = {};
    state.checkpoint_.reset();
    ZEPHYR_LOG_INFO("Executor ready in {:.3f}s: max_seq_len={}, max_num_seqs={}, max_batch_tokens={}",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - initialize_start).count(),
                    spec.limits_.max_seq_len_, spec.limits_.max_num_seqs_, spec.limits_.max_num_batched_tokens_);
    return executor;
  } catch (...) {
    state.Fail(std::current_exception());
    executor->StopWorkers();
    std::rethrow_exception(state.error_);
  }
}

Executor::~Executor() noexcept {
  try {
    Close();
  } catch (const std::exception &error) {
    state_->parallel_->Abort();
    ZEPHYR_LOG_ERROR("executor shutdown failed: {}", error.what());
  } catch (...) {
    state_->parallel_->Abort();
    ZEPHYR_LOG_ERROR("executor shutdown failed with an unknown error");
  }
}

auto Executor::IsFailed() const -> bool {
  {
    const std::scoped_lock lock{state_->latch_};
    if (state_->error_ != nullptr) {
      return true;
    }
  }
  const auto status = state_->parallel_->GetStatus();
  return status == ttl::CommunicatorStatus::FAILED || status == ttl::CommunicatorStatus::ABORTED;
}

void Executor::WaitForWorkers() {
  std::unique_lock lock{state_->latch_};
  const auto finished = [&] { return state_->error_ != nullptr || state_->completed_ == workers_.size(); };
  while (true) {
    state_->changed_.wait_for(lock, std::chrono::milliseconds{10}, finished);
    if (state_->error_ != nullptr) {
      std::rethrow_exception(state_->error_);
    }
    const auto completed = state_->completed_ == workers_.size();
    // Forward can block on device copies before returning. Keep NCCL error progress active throughout the call.
    // Poll once more after all GPU work completes, before committing a successful result.
    lock.unlock();
    state_->runtime_.Poll();
    if (state_->parallel_->GetStatus() != ttl::CommunicatorStatus::READY) {
      state_->Fail(std::make_exception_ptr(InternalException("tensor-parallel communication failed")));
    }
    lock.lock();
    if (state_->error_ != nullptr) {
      std::rethrow_exception(state_->error_);
    }
    if (completed) {
      return;
    }
  }
}

auto Executor::Execute(const ExecutionBatch &batch) -> ExecutionResult {
  {
    const std::scoped_lock lock{state_->latch_};
    if (state_->stopping_) {
      throw InvalidArgumentException("cannot execute on a closed or failed executor");
    }
  }
  // Preparation owns all host inputs and may reject the batch before any worker or GPU sees it.
  auto plan = processor_->Prepare(batch);
  if (plan == nullptr) {
    throw InternalException("input processor returned no execution plan");
  }
  {
    const std::scoped_lock lock{state_->latch_};
    state_->plan_ = std::move(plan);
    state_->completed_ = 0;
    ++state_->round_;
  }
  state_->changed_.notify_all();
  try {
    WaitForWorkers();
    const std::scoped_lock lock{state_->latch_};
    auto result = std::move(*state_->result_);
    state_->result_.reset();
    state_->plan_.reset();
    return result;
  } catch (...) {
    state_->Fail(std::current_exception());
    StopWorkers();
    std::rethrow_exception(state_->error_);
  }
}

void Executor::StopWorkers() {
  const auto worker_count = workers_.size();
  const auto stop_start = std::chrono::steady_clock::now();
  if (worker_count != 0) {
    ZEPHYR_LOG_INFO("Stopping {} executor worker(s)", worker_count);
  }
  {
    const std::scoped_lock lock{state_->latch_};
    state_->stopping_ = true;
  }
  state_->changed_.notify_all();
  workers_.clear();
  state_->plan_.reset();
  state_->result_.reset();
  state_->checkpoint_.reset();
  state_->factory_ = {};
  state_->processors_.clear();
  if (worker_count != 0) {
    ZEPHYR_LOG_INFO("Executor workers stopped in {:.3f}s",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - stop_start).count());
  }
}

void Executor::Close() {
  StopWorkers();
  try {
    const auto status = state_->parallel_->GetStatus();
    if (status == ttl::CommunicatorStatus::READY) {
      const auto close_start = std::chrono::steady_clock::now();
      ZEPHYR_LOG_DEBUG("Closing tensor-parallel communication");
      state_->parallel_->Close();
      ZEPHYR_LOG_DEBUG("Tensor-parallel communication closed in {:.3f}s",
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - close_start).count());
    } else if (status != ttl::CommunicatorStatus::CLOSED && status != ttl::CommunicatorStatus::ABORTED) {
      state_->parallel_->Abort();
      const auto aborted_status = state_->parallel_->GetStatus();
      if (aborted_status != ttl::CommunicatorStatus::ABORTED && aborted_status != ttl::CommunicatorStatus::CLOSED) {
        throw InternalException("tensor-parallel communication resources could not be released");
      }
    }
  } catch (...) {
    state_->Fail(std::current_exception());
    throw;
  }
}

}  // namespace zephyr::executor
