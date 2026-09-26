#pragma once

#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include <ttl/common/device.hpp>
#include <ttl/runtime/runtime.hpp>
#include <ttl/tensor/dtype.hpp>

#include "executor/execution.hpp"

namespace zephyr::executor {

/** One loaded model and its process-local tensor-parallel device order. */
struct ExecutorOptions final {
  std::filesystem::path model_dir_;
  ttl::DType dtype_{ttl::DType::BFLOAT16};
  std::vector<ttl::Device> devices_{ttl::Device{0}};
  ExecutionLimits execution_limits_;
  /** Target total device occupancy, including pre-existing allocations; execution peaks are reserved before KV. */
  double gpu_memory_utilization_{0.9};
};

struct WorkerState;

/**
 * Executes one batch on a fixed group of rank threads. Public calls belong to one control thread.
 * Execute is synchronous: success means every rank's GPU work has completed. Runtime is borrowed and
 * must outlive this object and all returned tensors. Requests, sequences, and page ownership stay with the caller.
 * Create profiles the selected runtime devices; exclude concurrent allocations on those devices until it returns.
 */
class Executor final {
 public:
  [[nodiscard]] static auto Create(ttl::Runtime &runtime, const ExecutorOptions &options, ExecutionFactory factory)
      -> std::unique_ptr<Executor>;

  Executor(const Executor &) = delete;
  auto operator=(const Executor &) -> Executor & = delete;
  Executor(Executor &&) = delete;
  auto operator=(Executor &&) -> Executor & = delete;
  ~Executor() noexcept;

  [[nodiscard]] auto GetSpec() const noexcept -> const ExecutionSpec & { return processor_->GetSpec(); }
  /** Query terminal worker or communication failure; rejected inputs and successful Close are not failures. */
  [[nodiscard]] auto IsFailed() const -> bool;
  /** Input errors reject only this batch; a running worker failure terminates the complete group. */
  [[nodiscard]] auto Execute(const ExecutionBatch &batch) -> ExecutionResult;
  /** Release workers and communication resources. Successfully closing twice is harmless. */
  void Close();

 private:
  Executor(ttl::Runtime &runtime, const ExecutorOptions &options, ExecutionFactory factory);

  /** Wait for initialization or batch completion while advancing asynchronous communication errors. */
  void WaitForWorkers();
  /** Wake and join all rank threads before releasing any shared task or checkpoint. */
  void StopWorkers();

  std::unique_ptr<WorkerState> state_;
  std::vector<std::jthread> workers_;
  std::unique_ptr<InputProcessor> processor_;
};

}  // namespace zephyr::executor
