#include "executor/executor.hpp"

#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

#include "common/exception.hpp"
#include "common/logger.hpp"
#include "executor/worker.hpp"

namespace zephyr::executor {

Executor::Executor(ttl::Runtime &runtime, const ExecutorOptions &options, ExecutionFactory factory)
    : state_(std::make_unique<WorkerState>(runtime, options, std::move(factory))) {}

auto Executor::Create(ttl::Runtime &runtime, const ExecutorOptions &options, ExecutionFactory factory)
    -> std::unique_ptr<Executor> {
  if (!factory) {
    throw ConfigurationException("executor requires an execution factory");
  }
  auto executor = std::unique_ptr<Executor>{new Executor{runtime, options, std::move(factory)}};
  auto &state = *executor->state_;
  try {
    executor->workers_.reserve(options.devices_.size());
    for (size_t rank = 0; rank < options.devices_.size(); ++rank) {
      executor->workers_.emplace_back([&state, rank] { RunWorker(state, rank); });
    }
    executor->WaitForWorkers();
    const auto &processor = *state.processors_.front();
    const auto &spec = processor.GetSpec();
    for (size_t rank = 0; rank < state.processors_.size(); ++rank) {
      const auto &rank_processor = *state.processors_[rank];
      const auto &rank_spec = rank_processor.GetSpec();
      if (rank_spec.device_ != options.devices_[rank] || rank_spec.dtype_ != spec.dtype_ ||
          !processor.IsCompatible(rank_processor)) {
        throw ConfigurationException("tensor-parallel ranks must report consistent execution capabilities");
      }
    }
    executor->processor_ = std::move(state.processors_.front());
    state.processors_.clear();
    state.factory_ = {};
    state.checkpoint_.reset();
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
}

void Executor::Close() {
  StopWorkers();
  try {
    const auto status = state_->parallel_->GetStatus();
    if (status == ttl::CommunicatorStatus::READY) {
      state_->parallel_->Close();
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
