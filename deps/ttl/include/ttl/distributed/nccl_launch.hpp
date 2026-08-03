#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <source_location>
#include <span>
#include <string_view>

#include <nccl.h>

#include "ttl/runtime/kernel_launch.hpp"

namespace ttl {

class ExecutionContext;
class NcclCommunicator;
class Tensor;

/** One rank participating in a process-local checked NCCL extension submission. */
struct LocalNcclKernelCall final {
  ExecutionContext *context_;
  NcclCommunicator *communicator_;
  std::span<const Tensor> inputs_;
  std::span<Tensor *const> outputs_;
  CudaKernelLaunchOptions options_{};
};

/** Callback-scoped NCCL handle plus the checked CUDA launch capability for its rank. */
class NcclKernelLaunch final {
 public:
  NcclKernelLaunch(const NcclKernelLaunch &) = delete;
  auto operator=(const NcclKernelLaunch &) -> NcclKernelLaunch & = delete;
  NcclKernelLaunch(NcclKernelLaunch &&) = delete;
  auto operator=(NcclKernelLaunch &&) -> NcclKernelLaunch & = delete;
  ~NcclKernelLaunch() noexcept;

  [[nodiscard]] auto GetCudaLaunch() noexcept -> CudaKernelLaunch &;
  [[nodiscard]] auto GetCommunicator() const noexcept -> ncclComm_t;
  [[nodiscard]] auto GetRank() const noexcept -> size_t;
  [[nodiscard]] auto GetWorldSize() const noexcept -> size_t;

 private:
  friend void SubmitNcclKernel(ExecutionContext &context, NcclCommunicator &communicator, std::string_view operation,
                               std::span<const Tensor> inputs, std::span<Tensor *const> outputs,
                               const std::function<ncclResult_t(NcclKernelLaunch &)> &function,
                               const CudaKernelLaunchOptions &options, std::source_location location);
  friend void SubmitNcclKernelsLocal(std::span<const LocalNcclKernelCall> calls, std::string_view operation,
                                     const std::function<ncclResult_t(size_t, NcclKernelLaunch &)> &function,
                                     std::source_location location);

  class Impl;
  explicit NcclKernelLaunch(std::unique_ptr<Impl> impl) noexcept;
  void Finish();
  void FailAfterCallbackException() noexcept;

  std::unique_ptr<Impl> impl_;
};

using NcclKernelFunction = std::function<ncclResult_t(NcclKernelLaunch &)>;
using LocalNcclKernelFunction = std::function<ncclResult_t(size_t, NcclKernelLaunch &)>;

/** Submit one rank of a checked NCCL extension; ranks may be called concurrently from graph-group workers. */
void SubmitNcclKernel(ExecutionContext &context, NcclCommunicator &communicator, std::string_view operation,
                      std::span<const Tensor> inputs, std::span<Tensor *const> outputs,
                      const NcclKernelFunction &function, const CudaKernelLaunchOptions &options = {},
                      std::source_location location = std::source_location::current());

/** Submit every rank of a process-local checked NCCL extension as one NCCL group. */
void SubmitNcclKernelsLocal(std::span<const LocalNcclKernelCall> calls, std::string_view operation,
                            const LocalNcclKernelFunction &function,
                            std::source_location location = std::source_location::current());

}  // namespace ttl
