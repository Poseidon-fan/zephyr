#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "executor/executor.hpp"
#include "parallel/tp.hpp"
#include "weight/checkpoint.hpp"

namespace zephyr::executor {

/**
 * One broadcast task slot and its completion state, shared by the control thread and fixed rank workers.
 * Options and initialized input processors are immutable. The latch protects task publication and completion;
 * no device work or communication call runs while holding it.
 */
struct WorkerState final {
  WorkerState(ttl::Runtime &runtime, const ExecutorOptions &options, ExecutionFactory factory);

  /** Preserve the first failure, stop task admission, and abort the whole TP group. */
  void Fail(std::exception_ptr error) noexcept;

  ttl::Runtime &runtime_;
  const ExecutorOptions options_;
  std::unique_ptr<weight::Checkpoint> checkpoint_;
  std::unique_ptr<parallel::TpContext> parallel_;
  ExecutionFactory factory_;
  std::vector<std::unique_ptr<InputProcessor>> processors_;

  std::mutex latch_;
  std::condition_variable changed_;
  /** Kept until all ranks complete, or until all workers join after a failure. */
  std::unique_ptr<const ExecutionPlan> plan_;
  std::optional<ExecutionResult> result_;
  std::exception_ptr error_;
  uint64_t round_{0};
  size_t completed_{0};
  bool initialization_started_{false};
  bool stopping_{false};
};

/** Own this rank's context, model, and cache until shutdown; report failures through the shared state. */
void RunWorker(WorkerState &state, size_t rank) noexcept;

}  // namespace zephyr::executor
