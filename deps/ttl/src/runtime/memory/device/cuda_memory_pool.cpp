#include "ttl/internal/runtime/memory/device/cuda_memory_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"

namespace ttl::internal {

CudaMemoryPool::CudaMemoryPool(Device device, std::shared_ptr<ErrorSink> error_sink, cudaMemPool_t pool,
                               std::source_location location) noexcept
    : device_(device), error_sink_(std::move(error_sink)), pool_(pool), location_(location) {}

auto CudaMemoryPool::Create(Device device, const std::shared_ptr<ErrorSink> &error_sink,
                            uint64_t release_threshold_bytes, std::source_location location)
    -> std::unique_ptr<CudaMemoryPool> {
  cudaMemPool_t pool = nullptr;
  DeviceGuard device_guard{device, *error_sink, location};
  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.handleTypes = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id = device.GetOrdinal();
  CheckCuda(GetCudaApi().create_memory_pool_(&pool, &properties), "cudaMemPoolCreate", location);
  if (pool == nullptr) {
    throw InternalError("cudaMemPoolCreate returned a null pool", location);
  }

  try {
    int enabled = 1;
    CheckCuda(GetCudaApi().set_memory_pool_attribute_(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold_bytes),
              "cudaMemPoolSetAttribute (release threshold)", location);
    CheckCuda(GetCudaApi().set_memory_pool_attribute_(pool, cudaMemPoolReuseFollowEventDependencies, &enabled),
              "cudaMemPoolSetAttribute (follow event dependencies)", location);
    CheckCuda(GetCudaApi().set_memory_pool_attribute_(pool, cudaMemPoolReuseAllowOpportunistic, &enabled),
              "cudaMemPoolSetAttribute (allow opportunistic reuse)", location);
    CheckCuda(GetCudaApi().set_memory_pool_attribute_(pool, cudaMemPoolReuseAllowInternalDependencies, &enabled),
              "cudaMemPoolSetAttribute (allow internal dependencies)", location);
    return std::unique_ptr<CudaMemoryPool>{new CudaMemoryPool{device, error_sink, pool, location}};
  } catch (...) {
    const ErrorReportContext context{.location_ = location, .device_ = device, .stream_id_ = std::nullopt};
    TryCuda(GetCudaApi().destroy_memory_pool_(pool), "cudaMemPoolDestroy", "memory-pool construction rollback",
            *error_sink, context);
    throw;
  }
}

CudaMemoryPool::~CudaMemoryPool() noexcept { static_cast<void>(TryCloseNoexcept()); }

auto CudaMemoryPool::AllocateAsync(void **pointer, size_t bytes, cudaStream_t stream) const noexcept -> cudaError_t {
  return GetCudaApi().malloc_from_pool_async_(pointer, bytes, pool_, stream);
}

auto CudaMemoryPool::FreeAsync(void *pointer, cudaStream_t stream) const noexcept -> cudaError_t {
  return GetCudaApi().free_async_(pointer, stream);
}

void CudaMemoryPool::SetPeerAccess(Device peer, bool enabled, std::source_location location) {
  const cudaMemAccessDesc descriptor{
      .location = {.type = cudaMemLocationTypeDevice, .id = peer.GetOrdinal()},
      .flags = enabled ? cudaMemAccessFlagsProtReadWrite : cudaMemAccessFlagsProtNone,
  };
  DeviceGuard device_guard{device_, *error_sink_, location};
  CheckCuda(GetCudaApi().set_memory_pool_access_(pool_, &descriptor, 1), "cudaMemPoolSetAccess", location);
}

void CudaMemoryPool::TrimTo(size_t target_reserved_bytes, std::source_location location) {
  DeviceGuard device_guard{device_, *error_sink_, location};
  CheckCuda(GetCudaApi().trim_memory_pool_(pool_, target_reserved_bytes), "cudaMemPoolTrimTo", location);
}

auto CudaMemoryPool::GetStats(std::source_location location) const -> CudaMemoryPoolStats {
  CudaMemoryPoolStats stats{};
  DeviceGuard device_guard{device_, *error_sink_, location};
  CheckCuda(GetCudaApi().get_memory_pool_attribute_(pool_, cudaMemPoolAttrUsedMemCurrent, &stats.used_bytes_),
            "cudaMemPoolGetAttribute (used current)", location);
  CheckCuda(GetCudaApi().get_memory_pool_attribute_(pool_, cudaMemPoolAttrReservedMemCurrent, &stats.reserved_bytes_),
            "cudaMemPoolGetAttribute (reserved current)", location);
  return stats;
}

auto CudaMemoryPool::TryGetStats(const ErrorReportContext &context) const noexcept -> CudaMemoryPoolStats {
  CudaMemoryPoolStats stats{};
  CleanupDeviceGuard device_guard{device_, *error_sink_, context, "query device memory-pool statistics",
                                  "restore after querying device memory-pool statistics"};
  if (!device_guard) {
    return stats;
  }
  TryCuda(GetCudaApi().get_memory_pool_attribute_(pool_, cudaMemPoolAttrUsedMemCurrent, &stats.used_bytes_),
          "cudaMemPoolGetAttribute", "used current", *error_sink_, context);
  TryCuda(GetCudaApi().get_memory_pool_attribute_(pool_, cudaMemPoolAttrReservedMemCurrent, &stats.reserved_bytes_),
          "cudaMemPoolGetAttribute", "reserved current", *error_sink_, context);
  return stats;
}

void CudaMemoryPool::Close(std::source_location location) {
  if (pool_ == nullptr) {
    return;
  }
  DeviceGuard device_guard{device_, *error_sink_, location};
  CheckCuda(GetCudaApi().destroy_memory_pool_(pool_), "cudaMemPoolDestroy", location);
  pool_ = nullptr;
}

auto CudaMemoryPool::TryCloseNoexcept() noexcept -> bool {
  if (pool_ == nullptr) {
    return true;
  }
  const ErrorReportContext context{.location_ = location_, .device_ = device_, .stream_id_ = std::nullopt};
  CleanupDeviceGuard device_guard{device_, *error_sink_, context, "destroy device memory pool",
                                  "restore after device memory pool destruction"};
  if (!device_guard ||
      !TryCuda(GetCudaApi().destroy_memory_pool_(pool_), "cudaMemPoolDestroy", *error_sink_, context)) {
    return false;
  }
  pool_ = nullptr;
  return true;
}

}  // namespace ttl::internal
