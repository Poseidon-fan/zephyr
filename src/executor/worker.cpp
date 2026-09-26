#include "executor/worker.hpp"

#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "common/exception.hpp"
#include "common/logger.hpp"
#include "weight/weight_builder.hpp"

namespace zephyr::executor {

WorkerState::WorkerState(ttl::Runtime &runtime, const ExecutorOptions &options, ExecutionFactory factory)
    : runtime_(runtime),
      options_(options),
      checkpoint_(std::make_unique<weight::Checkpoint>(options.model_dir_)),
      parallel_(parallel::TpContext::Create(runtime, options.devices_)),
      factory_(std::move(factory)),
      processors_(options.devices_.size()) {}

void WorkerState::Fail(std::exception_ptr error) noexcept {
  {
    const std::scoped_lock lock{latch_};
    if (error_ == nullptr) {
      error_ = std::move(error);
    }
    stopping_ = true;
  }
  changed_.notify_all();
  parallel_->Abort();
}

void RunWorker(WorkerState &state, size_t rank) noexcept {
  // Keep device resources alive through the failure handler, so Abort precedes synchronization and destruction.
  std::optional<ttl::ExecutionContext> context;
  std::unique_ptr<Execution> execution;
  const auto device = state.options_.devices_[rank].GetOrdinal();
  const char *phase = "loading model weights";
  try {
    const auto load_start = std::chrono::steady_clock::now();
    ZEPHYR_LOG_INFO("rank {} on cuda:{}: loading model weights (dtype={})", rank, device,
                    ttl::GetDTypeInfo(state.options_.dtype_).name_);
    const auto rank_context = state.parallel_->GetRank(rank);
    context.emplace(state.runtime_.CreateExecutionContext(rank_context.Device()));
    {
      const auto builder = weight::WeightBuilder{*state.checkpoint_, state.runtime_, state.options_.dtype_};
      execution =
          state.factory_(state.runtime_, *context, state.options_.model_dir_ / "config.json", builder, rank_context);
      if (execution == nullptr) {
        throw ConfigurationException("execution factory returned no execution");
      }
    }
    // All model uploads must finish before any rank profiles or allocates its runtime resources.
    context->Synchronize();
    ZEPHYR_LOG_INFO("rank {} on cuda:{}: model weights loaded in {:.3f}s", rank, device,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count());
    {
      const std::scoped_lock lock{state.latch_};
      ++state.completed_;
    }
    state.changed_.notify_all();
    {
      std::unique_lock lock{state.latch_};
      state.changed_.wait(lock, [&] { return state.stopping_ || state.initialization_started_; });
      if (state.stopping_) {
        ZEPHYR_LOG_INFO("rank {} on cuda:{}: initialization canceled", rank, device);
        return;
      }
    }
    phase = "initializing execution";
    const auto initialize_start = std::chrono::steady_clock::now();
    auto processor = execution->Initialize(*context, state.options_, rank_context);
    if (processor == nullptr) {
      throw ConfigurationException("execution returned no input processor");
    }
    context->Synchronize();
    ZEPHYR_LOG_INFO("rank {} on cuda:{}: execution ready in {:.3f}s", rank, device,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - initialize_start).count());
    {
      const std::scoped_lock lock{state.latch_};
      state.processors_[rank] = std::move(processor);
      ++state.completed_;
    }
    state.changed_.notify_all();

    phase = "executing a batch";
    uint64_t seen_round = 0;
    while (true) {
      const ExecutionPlan *plan = nullptr;
      {
        std::unique_lock lock{state.latch_};
        state.changed_.wait(lock, [&] { return state.stopping_ || state.round_ != seen_round; });
        if (state.stopping_) {
          break;
        }
        seen_round = state.round_;
        plan = state.plan_.get();
      }
      auto result = execution->Execute(*context, *plan);
      // Model return is only submission; synchronization also observes attention's device semantic errors.
      context->Synchronize();
      {
        const std::scoped_lock lock{state.latch_};
        if (rank == 0) {
          state.result_ = std::move(result);
        }
        ++state.completed_;
      }
      state.changed_.notify_all();
    }
    ZEPHYR_LOG_INFO("rank {} on cuda:{}: worker exiting", rank, device);
  } catch (...) {
    const auto error = std::current_exception();
    state.Fail(error);
    try {
      std::rethrow_exception(error);
    } catch (const std::exception &failure) {
      ZEPHYR_LOG_ERROR("rank {} on cuda:{} failed while {}: {}", rank, device, phase, failure.what());
    } catch (...) {
      ZEPHYR_LOG_ERROR("rank {} on cuda:{} failed while {} with an unknown error", rank, device, phase);
    }
    if (context.has_value()) {
      try {
        context->Synchronize();
      } catch (...) {
        ZEPHYR_LOG_WARN("worker rank {} failed to synchronize after abort; preserving the original failure", rank);
      }
    }
  }
}

}  // namespace zephyr::executor
