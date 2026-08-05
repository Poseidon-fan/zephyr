#include "ttl/internal/runtime/memory/pinned/cache.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"

namespace ttl::internal {
namespace {

constexpr size_t PINNED_PAGE_BYTES = 4U * 1024U;
constexpr size_t MAXIMUM_ROUNDED_SIZE_CLASS_BYTES = 64U * 1024U * 1024U;

[[nodiscard]] auto MakeContext(std::source_location location) noexcept -> ErrorReportContext {
  return {.location_ = location, .device_ = std::nullopt, .stream_id_ = std::nullopt};
}

}  // namespace

PinnedMemoryCache::PinnedMemoryCache(std::shared_ptr<ErrorSink> error_sink, size_t maximum_cached_bytes,
                                     std::source_location location) noexcept
    : error_sink_(std::move(error_sink)), maximum_cached_bytes_(maximum_cached_bytes), location_(location) {}

auto PinnedMemoryCache::GetSizeClass(size_t bytes, std::source_location location) -> size_t {
  if (bytes == 0) {
    return 0;
  }
  const auto page_rounded = AlignUp(bytes, PINNED_PAGE_BYTES, "pinned allocation size class", location);
  return page_rounded > MAXIMUM_ROUNDED_SIZE_CLASS_BYTES ? page_rounded : std::bit_ceil(page_rounded);
}

auto PinnedMemoryCache::Acquire(size_t requested_bytes, size_t capacity_bytes, std::source_location location)
    -> PinnedAllocation {
  auto allocation = Take(capacity_bytes);
  if (allocation.pointer_ != nullptr) {
    cache_hit_count_.fetch_add(1, std::memory_order_relaxed);
    return allocation;
  }
  return Allocate(requested_bytes, capacity_bytes, location);
}

auto PinnedMemoryCache::Release(PinnedAllocation allocation, std::source_location location) noexcept -> bool {
  return TryCache(allocation) || TryFree(allocation, MakeContext(location));
}

void PinnedMemoryCache::Free(PinnedAllocation allocation, std::source_location location) {
  CheckCuda(GetCudaApi().free_host_(allocation.pointer_), "cudaFreeHost", location);
  if (physical_bytes_.fetch_sub(allocation.capacity_bytes_, std::memory_order_relaxed) < allocation.capacity_bytes_) {
    std::terminate();
  }
  host_free_count_.fetch_add(1, std::memory_order_relaxed);
}

void PinnedMemoryCache::Trim(std::source_location location) {
  auto allocations = TakeAll();
  try {
    for (auto &[capacity, bucket] : allocations) {
      static_cast<void>(capacity);
      while (!bucket.empty()) {
        Free(bucket.back(), location);
        bucket.pop_back();
      }
    }
  } catch (...) {
    Restore(std::move(allocations));
    throw;
  }
}

auto PinnedMemoryCache::TryTrimNoexcept() noexcept -> bool {
  auto allocations = TakeAll();
  const auto context = MakeContext(location_);
  for (auto &[capacity, bucket] : allocations) {
    static_cast<void>(capacity);
    while (!bucket.empty()) {
      if (!TryFree(bucket.back(), context)) {
        Restore(std::move(allocations));
        return false;
      }
      bucket.pop_back();
    }
  }
  return true;
}

auto PinnedMemoryCache::GetStats() const noexcept -> PinnedMemoryCacheStats {
  return {
      .cached_bytes_ = cached_bytes_.load(std::memory_order_relaxed),
      .physical_bytes_ = physical_bytes_.load(std::memory_order_relaxed),
      .peak_physical_bytes_ = peak_physical_bytes_.load(std::memory_order_relaxed),
      .host_allocation_count_ = host_allocation_count_.load(std::memory_order_relaxed),
      .host_free_count_ = host_free_count_.load(std::memory_order_relaxed),
      .cache_hit_count_ = cache_hit_count_.load(std::memory_order_relaxed),
  };
}

auto PinnedMemoryCache::Take(size_t capacity_bytes) -> PinnedAllocation {
  std::scoped_lock lock{latch_};
  const auto iterator = allocations_.find(capacity_bytes);
  if (iterator == allocations_.end() || iterator->second.empty()) {
    return {.pointer_ = nullptr, .capacity_bytes_ = capacity_bytes};
  }
  auto allocation = iterator->second.back();
  iterator->second.pop_back();
  if (iterator->second.empty()) {
    allocations_.erase(iterator);
  }
  cached_bytes_.fetch_sub(capacity_bytes, std::memory_order_relaxed);
  return allocation;
}

auto PinnedMemoryCache::Allocate(size_t requested_bytes, size_t capacity_bytes, std::source_location location)
    -> PinnedAllocation {
  void *pointer = nullptr;
  auto status = GetCudaApi().host_alloc_(&pointer, capacity_bytes, cudaHostAllocPortable);
  if (status == cudaErrorMemoryAllocation) {
    ClearExpectedAllocationError(location);
    try {
      Trim(location);
    } catch (const Error &error) {
      std::string message{"cudaHostAlloc reported cudaErrorMemoryAllocation and pinned-cache recovery failed: "};
      message.append(error.GetMessage());
      throw CudaError(std::move(message), location);
    }
    pointer = nullptr;
    status = GetCudaApi().host_alloc_(&pointer, capacity_bytes, cudaHostAllocPortable);
  }
  if (status == cudaErrorMemoryAllocation) {
    ClearExpectedAllocationError(location);
    std::string message{"CUDA pinned host allocation failed after one cache trim: requested="};
    message.append(std::to_string(requested_bytes));
    message.append(", size_class=");
    message.append(std::to_string(capacity_bytes));
    message.append(", physical=");
    message.append(std::to_string(physical_bytes_.load(std::memory_order_relaxed)));
    message.append(", cached=");
    message.append(std::to_string(cached_bytes_.load(std::memory_order_relaxed)));
    throw OutOfMemoryError(std::move(message), location);
  }
  CheckCuda(status, "cudaHostAlloc", location);
  if (pointer == nullptr) {
    throw InternalError("cudaHostAlloc returned a null pointer", location);
  }

  host_allocation_count_.fetch_add(1, std::memory_order_relaxed);
  const auto physical = physical_bytes_.fetch_add(capacity_bytes, std::memory_order_relaxed) + capacity_bytes;
  auto peak = peak_physical_bytes_.load(std::memory_order_relaxed);
  while (physical > peak && !peak_physical_bytes_.compare_exchange_weak(peak, physical, std::memory_order_relaxed,
                                                                        std::memory_order_relaxed)) {
  }
  return {.pointer_ = pointer, .capacity_bytes_ = capacity_bytes};
}

auto PinnedMemoryCache::TryFree(PinnedAllocation allocation, const ErrorReportContext &context) noexcept -> bool {
  if (!TryCuda(GetCudaApi().free_host_(allocation.pointer_), "cudaFreeHost", "pinned allocator", *error_sink_,
               context)) {
    return false;
  }
  if (physical_bytes_.fetch_sub(allocation.capacity_bytes_, std::memory_order_relaxed) < allocation.capacity_bytes_) {
    std::terminate();
  }
  host_free_count_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

auto PinnedMemoryCache::TryCache(PinnedAllocation allocation) noexcept -> bool {
  if (allocation.capacity_bytes_ > maximum_cached_bytes_) {
    return false;
  }
  try {
    std::scoped_lock lock{latch_};
    const auto cached = cached_bytes_.load(std::memory_order_relaxed);
    if (cached > maximum_cached_bytes_ - allocation.capacity_bytes_) {
      return false;
    }
    allocations_[allocation.capacity_bytes_].push_back(allocation);
    cached_bytes_.fetch_add(allocation.capacity_bytes_, std::memory_order_relaxed);
    return true;
  } catch (...) {
    return false;
  }
}

auto PinnedMemoryCache::TakeAll() noexcept -> Cache {
  Cache allocations;
  {
    std::scoped_lock lock{latch_};
    allocations.swap(allocations_);
    cached_bytes_.store(0, std::memory_order_relaxed);
  }
  return allocations;
}

void PinnedMemoryCache::Restore(Cache allocations) noexcept {
  uint64_t restored_bytes = 0;
  std::scoped_lock lock{latch_};
  while (!allocations.empty()) {
    auto node = allocations.extract(allocations.begin());
    const auto existing = allocations_.find(node.key());
    for (const auto &allocation : node.mapped()) {
      if (restored_bytes > std::numeric_limits<uint64_t>::max() - allocation.capacity_bytes_) {
        std::terminate();
      }
      restored_bytes += allocation.capacity_bytes_;
    }
    if (existing == allocations_.end()) {
      allocations_.insert(std::move(node));
      continue;
    }
    existing->second.splice(existing->second.end(), node.mapped());
  }
  cached_bytes_.fetch_add(restored_bytes, std::memory_order_relaxed);
}

void PinnedMemoryCache::ClearExpectedAllocationError(std::source_location location) {
  const auto status = GetCudaApi().get_last_error_();
  if (status != cudaSuccess && status != cudaErrorMemoryAllocation) {
    CheckCuda(status, "cudaGetLastError after cudaHostAlloc", location);
  }
}

}  // namespace ttl::internal
