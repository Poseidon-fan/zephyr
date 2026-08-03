#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <source_location>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include <driver_types.h>

#include "ttl/tensor/dtype.hpp"

namespace ttl {

class ExecutionContext;
class Tensor;

/** Whether an external CUDA submission may participate in stream capture. */
enum class CudaCapturePolicy : uint8_t {
  FORBIDDEN,
  SAFE,
};

/** Resources and capture contract requested by one external CUDA submission. */
struct CudaKernelLaunchOptions final {
  size_t workspace_bytes_{0};
  size_t workspace_alignment_{256};
  CudaCapturePolicy capture_policy_{CudaCapturePolicy::FORBIDDEN};
};

/** A byte range in TTL's stream-local scratch arena. It is valid only during the submission callback. */
struct CudaWorkspace final {
  void *data_;
  size_t size_bytes_;
};

class CudaKernelLaunch;

template <typename Function>
  requires std::invocable<Function, CudaKernelLaunch &>
void SubmitCudaKernel(ExecutionContext &context, std::string_view operation, std::span<const Tensor> inputs,
                      std::span<Tensor *const> outputs, Function &&function,
                      const CudaKernelLaunchOptions &options = {},
                      std::source_location location = std::source_location::current());

/**
 * Checked bridge for inference-engine CUDA kernels implemented outside TTL.
 *
 * Construction claims exclusive host-side use of the context, selects its device, validates and records all tensor
 * storage on the context stream, and obtains optional stream-local workspace. The object must remain on the stack
 * until every kernel in the submission has been enqueued. Prefer SubmitCudaKernel so launch errors are always checked.
 *
 * Output tensors are read-write. An input that is also written must be listed in outputs as well. Tensor metadata and
 * output alias requirements remain the custom operator's responsibility.
 */
class CudaKernelLaunch final {
 public:
  CudaKernelLaunch(const CudaKernelLaunch &) = delete;
  auto operator=(const CudaKernelLaunch &) -> CudaKernelLaunch & = delete;
  CudaKernelLaunch(CudaKernelLaunch &&) = delete;
  auto operator=(CudaKernelLaunch &&) -> CudaKernelLaunch & = delete;
  ~CudaKernelLaunch() noexcept;

  [[nodiscard]] auto GetStream() const noexcept -> cudaStream_t;
  [[nodiscard]] auto GetWorkspace() const noexcept -> CudaWorkspace;
  [[nodiscard]] auto IsCapturing() const noexcept -> bool;

  [[nodiscard]] auto GetInputData(const Tensor &tensor,
                                  std::source_location location = std::source_location::current()) const -> const
      void *;
  [[nodiscard]] auto GetOutputData(Tensor &tensor,
                                   std::source_location location = std::source_location::current()) const -> void *;

  template <TensorStorageType T>
  [[nodiscard]] auto GetInputDataAs(const Tensor &tensor,
                                    std::source_location location = std::source_location::current()) const
      -> const std::remove_cv_t<T> * {
    return static_cast<const std::remove_cv_t<T> *>(GetInputDataAsDType(tensor, DTYPE_OF<T>, location));
  }

  template <TensorStorageType T>
  [[nodiscard]] auto GetOutputDataAs(Tensor &tensor,
                                     std::source_location location = std::source_location::current()) const
      -> std::remove_cv_t<T> * {
    return static_cast<std::remove_cv_t<T> *>(GetOutputDataAsDType(tensor, DTYPE_OF<T>, location));
  }

 private:
  template <typename Function>
    requires std::invocable<Function, CudaKernelLaunch &>
  friend void SubmitCudaKernel(ExecutionContext &context, std::string_view operation, std::span<const Tensor> inputs,
                               std::span<Tensor *const> outputs, Function &&function,
                               const CudaKernelLaunchOptions &options, std::source_location location);

  class Impl;

  CudaKernelLaunch(ExecutionContext &context, std::string_view operation, std::span<const Tensor> inputs,
                   std::span<Tensor *const> outputs, const CudaKernelLaunchOptions &options,
                   std::source_location location);

  void CheckLaunch() const;
  void FailAfterCallbackException() noexcept;

  [[nodiscard]] auto GetInputDataAsDType(const Tensor &tensor, DType dtype, std::source_location location) const
      -> const void *;
  [[nodiscard]] auto GetOutputDataAsDType(Tensor &tensor, DType dtype, std::source_location location) const -> void *;

  std::unique_ptr<Impl> impl_;
};

/**
 * Run one checked external CUDA submission.
 *
 * The callback receives the only valid access path to registered output pointers, the native stream, and scratch
 * workspace. A normal callback return is followed by cudaGetLastError through TTL's error translation layer. If the
 * callback throws, TTL invalidates the submission context, clears and reports any pending CUDA launch error, and then
 * rethrows the original exception so partially submitted work cannot be followed by unrelated operations.
 */
template <typename Function>
  requires std::invocable<Function, CudaKernelLaunch &>
void SubmitCudaKernel(ExecutionContext &context, std::string_view operation, std::span<const Tensor> inputs,
                      std::span<Tensor *const> outputs, Function &&function, const CudaKernelLaunchOptions &options,
                      std::source_location location) {
  CudaKernelLaunch launch{context, operation, inputs, outputs, options, location};
  try {
    std::invoke(std::forward<Function>(function), launch);
  } catch (...) {
    launch.FailAfterCallbackException();
    throw;
  }
  launch.CheckLaunch();
}

static_assert(!std::copy_constructible<CudaKernelLaunch>);
static_assert(!std::move_constructible<CudaKernelLaunch>);

}  // namespace ttl
