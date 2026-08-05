#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <source_location>

#include "ttl/internal/runtime/error_report.hpp"
#include "ttl/internal/runtime/memory/pinned/allocation.hpp"

namespace ttl {

class ErrorSink;

}  // namespace ttl

namespace ttl::internal {

struct PinnedMemoryCacheStats final {
  uint64_t cached_bytes_;
  uint64_t physical_bytes_;
  uint64_t peak_physical_bytes_;
  uint64_t host_allocation_count_;
  uint64_t host_free_count_;
  uint64_t cache_hit_count_;
};

/** Size-class cache and native CUDA owner for page-locked host allocations. */
class PinnedMemoryCache final {
 public:
  PinnedMemoryCache(std::shared_ptr<ErrorSink> error_sink, size_t maximum_cached_bytes,
                    std::source_location location) noexcept;

  PinnedMemoryCache(const PinnedMemoryCache &) = delete;
  auto operator=(const PinnedMemoryCache &) -> PinnedMemoryCache & = delete;
  PinnedMemoryCache(PinnedMemoryCache &&) = delete;
  auto operator=(PinnedMemoryCache &&) -> PinnedMemoryCache & = delete;

  [[nodiscard]] static auto GetSizeClass(size_t bytes, std::source_location location) -> size_t;
  [[nodiscard]] auto Acquire(size_t requested_bytes, size_t capacity_bytes, std::source_location location)
      -> PinnedAllocation;
  [[nodiscard]] auto Release(PinnedAllocation allocation, std::source_location location) noexcept -> bool;
  void Free(PinnedAllocation allocation, std::source_location location);
  void Trim(std::source_location location);
  [[nodiscard]] auto TryTrimNoexcept() noexcept -> bool;
  [[nodiscard]] auto GetStats() const noexcept -> PinnedMemoryCacheStats;

 private:
  using Cache = std::map<size_t, std::list<PinnedAllocation>>;

  [[nodiscard]] auto Take(size_t capacity_bytes) -> PinnedAllocation;
  [[nodiscard]] auto Allocate(size_t requested_bytes, size_t capacity_bytes, std::source_location location)
      -> PinnedAllocation;
  [[nodiscard]] auto TryFree(PinnedAllocation allocation, const ErrorReportContext &context) noexcept -> bool;
  [[nodiscard]] auto TryCache(PinnedAllocation allocation) noexcept -> bool;
  [[nodiscard]] auto TakeAll() noexcept -> Cache;
  void Restore(Cache allocations) noexcept;
  void ClearExpectedAllocationError(std::source_location location);

  std::shared_ptr<ErrorSink> error_sink_;
  size_t maximum_cached_bytes_;
  std::source_location location_;
  mutable std::mutex latch_;
  Cache allocations_;
  std::atomic<uint64_t> cached_bytes_{0};
  std::atomic<uint64_t> physical_bytes_{0};
  std::atomic<uint64_t> peak_physical_bytes_{0};
  std::atomic<uint64_t> host_allocation_count_{0};
  std::atomic<uint64_t> host_free_count_{0};
  std::atomic<uint64_t> cache_hit_count_{0};
};

}  // namespace ttl::internal
