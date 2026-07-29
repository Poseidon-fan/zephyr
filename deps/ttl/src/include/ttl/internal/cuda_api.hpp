#pragma once

#include <cuda_runtime_api.h>

namespace ttl::internal {

/**
 * Injectable CUDA Runtime entry points used at failure-sensitive boundaries.
 *
 * The table contains only calls that require deterministic fault injection. Every function pointer must be non-null.
 */
struct CudaApi final {
  decltype(&cudaGetDeviceCount) get_device_count_;
  decltype(&cudaGetDeviceProperties) get_device_properties_;
  decltype(&cudaGetDevice) get_device_;
  decltype(&cudaSetDevice) set_device_;
  decltype(&cudaGetLastError) get_last_error_;
  decltype(&cudaDeviceGetStreamPriorityRange) get_stream_priority_range_;
  decltype(&cudaStreamCreateWithPriority) create_stream_with_priority_;
  decltype(&cudaStreamDestroy) destroy_stream_;
  decltype(&cudaEventCreateWithFlags) create_event_with_flags_;
  decltype(&cudaEventRecord) record_event_;
  decltype(&cudaEventQuery) query_event_;
  decltype(&cudaEventSynchronize) synchronize_event_;
  decltype(&cudaEventDestroy) destroy_event_;
  decltype(&cudaStreamWaitEvent) stream_wait_event_;
};

/** Return the active process-wide table, backed by the real CUDA Runtime API outside scoped tests. */
[[nodiscard]] auto GetCudaApi() noexcept -> const CudaApi &;

/**
 * Process-wide CUDA API override for deterministic unit tests.
 *
 * Overrides may be nested only in strict LIFO order and must not overlap across host threads. The supplied table must
 * outlive the override, and every operation using it must finish before the override is destroyed.
 */
class ScopedCudaApiOverride final {
 public:
  explicit ScopedCudaApiOverride(const CudaApi &cuda_api) noexcept;

  ScopedCudaApiOverride(const ScopedCudaApiOverride &) = delete;
  auto operator=(const ScopedCudaApiOverride &) -> ScopedCudaApiOverride & = delete;
  ScopedCudaApiOverride(ScopedCudaApiOverride &&) = delete;
  auto operator=(ScopedCudaApiOverride &&) -> ScopedCudaApiOverride & = delete;

  ~ScopedCudaApiOverride() noexcept;

 private:
  const CudaApi *cuda_api_;
  const CudaApi *previous_cuda_api_;
};

}  // namespace ttl::internal
