#include "executor/worker.hpp"

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
  try {
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
    // Weight uploads may still refer to the shared checkpoint. Publish readiness only after they complete.
    context->Synchronize();
    auto processor = execution->CreateInputProcessor();
    if (processor == nullptr) {
      throw ConfigurationException("execution returned no input processor");
    }
    {
      const std::scoped_lock lock{state.latch_};
      state.processors_[rank] = std::move(processor);
      ++state.completed_;
    }
    state.changed_.notify_all();

    uint64_t seen_round = 0;
    while (true) {
      const ExecutionPlan *plan = nullptr;
      {
        std::unique_lock lock{state.latch_};
        state.changed_.wait(lock, [&] { return state.stopping_ || state.round_ != seen_round; });
        if (state.stopping_) {
          return;
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
  } catch (...) {
    state.Fail(std::current_exception());
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
