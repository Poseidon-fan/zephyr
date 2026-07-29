#pragma once

#include <cuda_runtime_api.h>

namespace ttl::internal {

/**
 * Injectable CUDA Runtime entry points used at failure-sensitive boundaries.
 *
 * The table contains only calls that require deterministic fault injection. Every function pointer must be non-null
 * and the table must outlive every object that references it.
 */
struct CudaApi final {
  decltype(&cudaGetDeviceCount) get_device_count_;
  decltype(&cudaGetDeviceProperties) get_device_properties_;
  decltype(&cudaGetDevice) get_device_;
  decltype(&cudaSetDevice) set_device_;
  decltype(&cudaDeviceGetStreamPriorityRange) get_stream_priority_range_;
  decltype(&cudaStreamCreateWithPriority) create_stream_with_priority_;
  decltype(&cudaStreamDestroy) destroy_stream_;
};

/** Return the process-lifetime table backed by the real CUDA Runtime API. */
[[nodiscard]] auto GetCudaApi() noexcept -> const CudaApi &;

}  // namespace ttl::internal
