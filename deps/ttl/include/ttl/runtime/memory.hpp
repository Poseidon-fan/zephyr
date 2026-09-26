#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "ttl/common/device.hpp"

namespace ttl {

/** Configures one CUDA device's stream-ordered memory pool. */
struct DeviceMemoryOptions final {
  /** Bytes the CUDA pool may retain after frees; defaults to no automatic release. */
  uint64_t release_threshold_bytes_{std::numeric_limits<uint64_t>::max()};
  /** Maximum bytes allocated or asynchronously retiring; zero disables the TTL budget. */
  uint64_t max_live_bytes_{0};
};

/** CUDA device memory snapshot and the runtime allocator's configured admission limit. */
struct DeviceMemoryInfo final {
  uint64_t total_bytes_;
  uint64_t free_bytes_;
  /** Zero disables the TTL budget; external allocations and cached pool pages are not charged. */
  uint64_t max_live_bytes_;
};

/** Configures the process-wide page-locked host-memory cache and admission budget. */
struct PinnedMemoryOptions final {
  size_t max_cached_bytes_{256U * 1024U * 1024U};
  /** Maximum live or asynchronously retiring capacity; zero disables the admission budget. */
  size_t max_live_bytes_{512U * 1024U * 1024U};
};

/** Allocator and dependent-resource statistics for one registered CUDA device. */
struct DeviceMemoryStatistics final {
  Device device_;
  uint64_t logical_live_bytes_;
  uint64_t retiring_bytes_;
  /**
   * Peak charged allocation capacity since construction or the last peak reset, including scratch and cuBLAS
   * workspaces, live and retiring storage, and pending reservations. Excludes external memory and cached pool pages.
   */
  uint64_t peak_physical_in_use_bytes_;
  uint64_t allocation_count_;
  uint64_t retirement_count_;
  uint64_t retry_count_;
  uint64_t oom_count_;
  uint64_t trim_count_;
  uint64_t pending_retirement_count_;
  uint64_t quarantined_retirement_count_;
  uint64_t quarantined_bytes_;
  uint64_t pool_used_bytes_;
  uint64_t pool_reserved_bytes_;
  uint64_t outstanding_storage_count_;
  size_t cached_event_count_;
  size_t outstanding_event_count_;
  size_t event_cache_capacity_;
  size_t blas_workspace_bytes_;
};

/** Process-wide page-locked host-memory accounting. */
struct PinnedMemoryStatistics final {
  /** Sum of caller-requested bytes for live PinnedBuffer owners. */
  uint64_t logical_live_bytes_;
  /** Sum of rounded size-class capacities for live PinnedBuffer owners. */
  uint64_t live_capacity_bytes_;
  /** Capacity retained by asynchronous retirement records. */
  uint64_t retiring_capacity_bytes_;
  /** Capacity held by the reusable host cache. */
  uint64_t cached_capacity_bytes_;
  /** Live plus retiring capacity charged to the admission budget. */
  uint64_t budgeted_capacity_bytes_;
  /** High-water mark of budgeted live plus retiring capacity. */
  uint64_t peak_budgeted_capacity_bytes_;
  /** Total native host capacity held by live, retiring, and cached allocations. */
  uint64_t physical_bytes_;
  /** High-water mark of native host capacity. */
  uint64_t peak_physical_bytes_;
  uint64_t host_allocation_count_;
  uint64_t host_free_count_;
  uint64_t cache_hit_count_;
  uint64_t retirement_count_;
  uint64_t pending_retirement_count_;
  uint64_t quarantined_retirement_count_;
  uint64_t quarantined_bytes_;
  uint64_t outstanding_buffer_count_;
};

/** Existing CUDA device allocation and optional shared lifetime owner. */
struct ExternalDeviceMemory final {
  void *pointer_;
  size_t capacity_bytes_;
  Device device_;
  std::shared_ptr<void> owner_;
};

}  // namespace ttl
