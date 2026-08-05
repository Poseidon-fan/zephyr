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

#include "ttl/runtime/device_error.hpp"
#include "ttl/runtime/philox.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl::internal {

class CommunicatorGroupState;

}  // namespace ttl::internal

namespace ttl {

class ExecutionContext;
class Generator;
class NcclKernelLaunch;
class Tensor;

/** Whether an external CUDA submission may participate in stream capture. */
enum class CudaCapturePolicy : uint8_t {
  FORBIDDEN,
  SAFE,
};

/** One stream-local scratch workspace requested for an external CUDA submission lane. */
struct CudaWorkspaceRequest final {
  size_t size_bytes_{0};
  size_t alignment_{256};
};

/** Resources and capture contract requested by one external CUDA submission. */
struct CudaKernelLaunchOptions final {
  CudaWorkspaceRequest workspace_{};
  size_t auxiliary_stream_count_{0};
  /** Optional request for each leading auxiliary lane; omitted lanes receive an empty workspace. */
  // NOLINTNEXTLINE(readability-redundant-member-init): keeps designated initializers warning-free under NVCC.
  std::span<const CudaWorkspaceRequest> auxiliary_workspaces_{};
  CudaCapturePolicy capture_policy_{CudaCapturePolicy::FORBIDDEN};
};

/** A byte range in TTL's stream-local scratch arena. It is valid only during the submission callback. */
struct CudaWorkspace final {
  void *data_{nullptr};
  size_t size_bytes_{0};
};

class CudaKernelLaunch;

template <typename Function>
  requires std::invocable<Function, CudaKernelLaunch &>
void SubmitCudaKernel(ExecutionContext &context, std::string_view operation, std::span<const Tensor> inputs,
                      std::span<Tensor *const> outputs, Function &&function,
                      const CudaKernelLaunchOptions &options = {},
                      std::source_location location = std::source_location::current());

/**
 * @brief Checked bridge for inference-engine CUDA kernels implemented outside TTL.
 *
 * Construction claims exclusive host-side use of the context, selects its device, validates and records all tensor
 * storage on every requested execution lane, and obtains optional stream-local workspaces. Auxiliary lanes form a
 * structured fork/join region: they wait for prior primary-stream work before the callback, and the primary stream
 * waits for them before the submission completes. The object must remain on the stack until every kernel has been
 * enqueued. Prefer SubmitCudaKernel so launch errors and the fork/join boundary are always checked.
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
  [[nodiscard]] auto GetAuxiliaryStreamCount() const noexcept -> size_t;
  [[nodiscard]] auto GetAuxiliaryStream(size_t index,
                                        std::source_location location = std::source_location::current()) const
      -> cudaStream_t;
  [[nodiscard]] auto GetAuxiliaryWorkspace(size_t index,
                                           std::source_location location = std::source_location::current()) const
      -> CudaWorkspace;
  [[nodiscard]] auto IsCapturing() const noexcept -> bool;

  /** Make every auxiliary lane wait for primary-lane work submitted before this call. */
  void PublishPrimaryToAuxiliary();

  /** Make the primary lane wait for work submitted on every auxiliary lane before this call. */
  void PublishAuxiliaryToPrimary();

  /** Register this submission with the context's sticky first-error channel. */
  [[nodiscard]] auto GetDeviceErrorContext(DType source_dtype = DType::BOOL, DType target_dtype = DType::BOOL)
      -> CudaDeviceErrorContext;

  /** Reserve Philox4x32 blocks in stream order for a fused external sampling kernel. */
  [[nodiscard]] auto ReservePhilox(Generator &generator, uint64_t block_count) -> CudaPhiloxReservation;

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
  friend class NcclKernelLaunch;
  template <typename Function>
    requires std::invocable<Function, CudaKernelLaunch &>
  friend void SubmitCudaKernel(ExecutionContext &context, std::string_view operation, std::span<const Tensor> inputs,
                               std::span<Tensor *const> outputs, Function &&function,
                               const CudaKernelLaunchOptions &options, std::source_location location);

  class Impl;

  CudaKernelLaunch(ExecutionContext &context, std::string_view operation, std::span<const Tensor> inputs,
                   std::span<Tensor *const> outputs, const CudaKernelLaunchOptions &options,
                   std::source_location location);

  void Finish();
  void FailAfterCallbackException() noexcept;
  void RetainCommunicator(const std::shared_ptr<internal::CommunicatorGroupState> &communicator);

  [[nodiscard]] auto GetInputDataAsDType(const Tensor &tensor, DType dtype, std::source_location location) const
      -> const void *;
  [[nodiscard]] auto GetOutputDataAsDType(Tensor &tensor, DType dtype, std::source_location location) const -> void *;

  std::unique_ptr<Impl> impl_;
};

/**
 * @brief Run one checked external CUDA submission.
 *
 * The callback receives the only valid access path to registered output pointers, native streams, and scratch
 * workspaces. A normal callback return is followed by cudaGetLastError and, when auxiliary streams were requested, a
 * join back to the primary stream. If the callback throws, TTL best-effort joins the lanes, invalidates the submission
 * context, clears and reports any pending CUDA launch error, and then rethrows the original exception so partially
 * submitted work cannot be followed by unrelated operations.
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
  launch.Finish();
}

static_assert(!std::copy_constructible<CudaKernelLaunch>);
static_assert(!std::move_constructible<CudaKernelLaunch>);

}  // namespace ttl
