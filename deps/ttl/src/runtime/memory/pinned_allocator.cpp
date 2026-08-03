#include "ttl/internal/runtime/memory/pinned_allocator.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"
#include "ttl/internal/runtime/execution/event_pool.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/runtime/device.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {
namespace {

constexpr size_t PINNED_PAGE_BYTES = 4U * 1024U;
constexpr size_t MAXIMUM_ROUNDED_SIZE_CLASS_BYTES = 64U * 1024U * 1024U;

enum class PinnedAllocatorStatus : uint8_t {
  RUNNING,
  FAILED,
  CLOSING,
  CLOSED,
};

struct PinnedRetirement final {
  PinnedRetirement(PinnedAllocation allocation, std::vector<std::shared_ptr<StreamState>> streams,
                   std::vector<PooledEvent> events, std::source_location location, bool poisoned = false) noexcept
      : allocation_(allocation),
        streams_(std::move(streams)),
        events_(std::move(events)),
        location_(location),
        poisoned_(poisoned) {}

  PinnedRetirement(const PinnedRetirement &) = delete;
  auto operator=(const PinnedRetirement &) -> PinnedRetirement & = delete;
  PinnedRetirement(PinnedRetirement &&) noexcept = default;
  auto operator=(PinnedRetirement &&) noexcept -> PinnedRetirement & = default;

  PinnedAllocation allocation_;
  std::vector<std::shared_ptr<StreamState>> streams_;
  std::vector<PooledEvent> events_;
  std::source_location location_;
  bool poisoned_;
};

[[nodiscard]] auto MakeContext(std::source_location location, std::optional<Device> device = std::nullopt,
                               std::optional<uint64_t> stream_id = std::nullopt) noexcept -> ErrorReportContext {
  return {
      .location_ = location,
      .device_ = device,
      .stream_id_ = stream_id,
  };
}

void ReportError(ErrorCode code, std::string_view message, ErrorSink &error_sink,
                 const ErrorReportContext &context) noexcept {
  try {
    error_sink.Report(ErrorRecord{
        .code_ = code,
        .message_ = std::string{message},
        .device_ = context.device_,
        .stream_id_ = context.stream_id_,
        .location_ = context.location_,
    });
  } catch (...) {
    return;
  }
}

void ValidateDependencies(const std::shared_ptr<ErrorSink> &error_sink,
                          std::span<const std::shared_ptr<EventPool>> event_pools, std::source_location location) {
  if (error_sink == nullptr) {
    throw InvalidArgumentError("pinned allocator error sink must not be null", location);
  }
  if (event_pools.empty()) {
    throw InvalidArgumentError("pinned allocator requires at least one event pool", location);
  }
  for (size_t index = 0; index < event_pools.size(); index++) {
    if (event_pools[index] == nullptr) {
      throw InvalidArgumentError("pinned allocator event pool must not be null", location);
    }
    for (size_t other = index + 1; other < event_pools.size(); other++) {
      if (event_pools[other] != nullptr && event_pools[index]->GetDevice() == event_pools[other]->GetDevice()) {
        throw InvalidArgumentError("pinned allocator event pool devices must be unique", location);
      }
    }
  }
}

[[nodiscard]] auto GetSizeClass(size_t bytes, std::source_location location) -> size_t {
  if (bytes == 0) {
    return 0;
  }
  const auto page_rounded = AlignUp(bytes, PINNED_PAGE_BYTES, "pinned allocation size class", location);
  if (page_rounded > MAXIMUM_ROUNDED_SIZE_CLASS_BYTES) {
    return page_rounded;
  }
  return std::bit_ceil(page_rounded);
}

[[nodiscard]] auto FormatOutstandingBuffers(uint64_t count) -> std::string {
  std::string message{"cannot shut down pinned allocator while "};
  message.append(std::to_string(count));
  message.append(" PinnedBuffer object");
  if (count != 1) {
    message.push_back('s');
  }
  message.append(" remain alive");
  return message;
}

[[nodiscard]] auto FormatBudgetError(size_t requested_bytes, size_t capacity_bytes, uint64_t in_use_bytes,
                                     uint64_t maximum_bytes) -> std::string {
  std::string message{"pinned host allocation exceeds the configured in-use budget: requested="};
  message.append(std::to_string(requested_bytes));
  message.append(", size_class=");
  message.append(std::to_string(capacity_bytes));
  message.append(", in_use=");
  message.append(std::to_string(in_use_bytes));
  message.append(", maximum=");
  message.append(std::to_string(maximum_bytes));
  return message;
}

}  // namespace

class PinnedAllocatorImpl final {
 public:
  PinnedAllocatorImpl(std::shared_ptr<ErrorSink> error_sink, std::vector<std::shared_ptr<EventPool>> event_pools,
                      PinnedMemoryOptions options, std::source_location location) noexcept
      : error_sink_(std::move(error_sink)),
        event_pools_(std::move(event_pools)),
        options_(options),
        location_(location) {}

  PinnedAllocatorImpl(const PinnedAllocatorImpl &) = delete;
  auto operator=(const PinnedAllocatorImpl &) -> PinnedAllocatorImpl & = delete;
  PinnedAllocatorImpl(PinnedAllocatorImpl &&) = delete;
  auto operator=(PinnedAllocatorImpl &&) -> PinnedAllocatorImpl & = delete;

  ~PinnedAllocatorImpl() noexcept = default;

  [[nodiscard]] auto Allocate(const std::shared_ptr<PinnedAllocator> &allocator, size_t bytes,
                              std::source_location location) -> PinnedBuffer {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    ValidateRunning(location);
    const auto capacity_bytes = GetSizeClass(bytes, location);
    if (capacity_bytes == 0) {
      outstanding_buffer_count_.fetch_add(1, std::memory_order_relaxed);
      try {
        return PinnedBuffer{std::make_shared<PinnedBlock>(
            PinnedAllocation{
                .pointer_ = nullptr,
                .capacity_bytes_ = 0,
            },
            0, allocator, location)};
      } catch (...) {
        outstanding_buffer_count_.fetch_sub(1, std::memory_order_relaxed);
        throw;
      }
    }

    PollRetirements();
    ValidateRunning(location);
    ReserveInUse(bytes, capacity_bytes, location);
    try {
      ReserveRetirementSlot(location);
    } catch (...) {
      ReleaseInUse(capacity_bytes);
      throw;
    }

    auto allocation = TakeCached(capacity_bytes);
    if (!allocation.has_value()) {
      try {
        allocation.emplace(AllocateHost(bytes, capacity_bytes, location));
      } catch (...) {
        CancelRetirementSlot();
        ReleaseInUse(capacity_bytes);
        throw;
      }
    } else {
      cache_hit_count_.fetch_add(1, std::memory_order_relaxed);
    }

    logical_live_bytes_.fetch_add(capacity_bytes, std::memory_order_relaxed);
    outstanding_buffer_count_.fetch_add(1, std::memory_order_relaxed);
    try {
      auto block = std::make_shared<PinnedBlock>(*allocation, bytes, allocator, location);
      return PinnedBuffer{std::move(block)};
    } catch (...) {
      outstanding_buffer_count_.fetch_sub(1, std::memory_order_relaxed);
      logical_live_bytes_.fetch_sub(capacity_bytes, std::memory_order_relaxed);
      if (TryFreeHost(*allocation, MakeContext(location))) {
        CancelRetirementSlot();
        ReleaseInUse(capacity_bytes);
      } else {
        pending_bytes_.fetch_add(capacity_bytes, std::memory_order_relaxed);
        status_.store(PinnedAllocatorStatus::FAILED, std::memory_order_release);
        Enqueue(PinnedRetirement{*allocation, {}, {}, location, true});
      }
      throw;
    }
  }

  void Retire(PinnedAllocation allocation, std::vector<std::shared_ptr<StreamState>> streams,
              std::vector<PooledEvent> events, std::source_location location) noexcept {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    if (outstanding_buffer_count_.fetch_sub(1, std::memory_order_release) == 0) {
      std::terminate();
    }
    if (allocation.capacity_bytes_ == 0) {
      return;
    }

    const auto capacity_bytes = static_cast<uint64_t>(allocation.capacity_bytes_);
    logical_live_bytes_.fetch_sub(capacity_bytes, std::memory_order_relaxed);
    pending_bytes_.fetch_add(capacity_bytes, std::memory_order_relaxed);
    PinnedRetirement retirement{allocation, std::move(streams), std::move(events), location};
    if (retirement.streams_.empty()) {
      if (CacheOrFreeNoexcept(retirement.allocation_, retirement.location_)) {
        CancelRetirementSlot();
        pending_bytes_.fetch_sub(capacity_bytes, std::memory_order_relaxed);
        ReleaseInUse(allocation.capacity_bytes_);
      } else {
        retirement.poisoned_ = true;
        status_.store(PinnedAllocatorStatus::FAILED, std::memory_order_release);
        Enqueue(std::move(retirement));
      }
      return;
    }

    try {
      for (const auto &stream : retirement.streams_) {
        auto event = FindEventPool(stream->GetDevice(), location)->Acquire(location);
        const auto context = MakeContext(location, stream->GetDevice(), stream->GetId());
        CleanupDeviceGuard device_guard{stream->GetDevice(), *error_sink_, context, "record pinned-buffer usage",
                                        "restore after pinned-buffer usage record"};
        if (!device_guard || !TryCuda(GetCudaApi().record_event_(event.GetNative(), stream->GetNative()),
                                      "cudaEventRecord", "pinned-buffer retirement", *error_sink_, context)) {
          event.Discard();
          retirement.poisoned_ = true;
          break;
        }
        retirement.events_.push_back(std::move(event));
      }
    } catch (const Error &error) {
      ReportError(error.GetCode(), error.GetMessage(), *error_sink_, MakeContext(location));
      retirement.poisoned_ = true;
    } catch (const std::exception &error) {
      ReportError(ErrorCode::INTERNAL, error.what(), *error_sink_, MakeContext(location));
      retirement.poisoned_ = true;
    } catch (...) {
      ReportError(ErrorCode::INTERNAL, "unknown failure while retiring a pinned host allocation", *error_sink_,
                  MakeContext(location));
      retirement.poisoned_ = true;
    }

    if (retirement.poisoned_) {
      status_.store(PinnedAllocatorStatus::FAILED, std::memory_order_release);
    }
    Enqueue(std::move(retirement));
  }

  void Poll() noexcept {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    PollRetirements();
  }

  void Trim(std::source_location location) {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    ValidateRunning(location);
    PollRetirements();
    FreeCached(location);
  }

  void Shutdown(std::source_location location) {
    const std::scoped_lock shutdown_lock{shutdown_latch_};
    if (status_.load(std::memory_order_acquire) == PinnedAllocatorStatus::CLOSED) {
      return;
    }
    status_.store(PinnedAllocatorStatus::CLOSING, std::memory_order_release);

    const std::unique_lock lifecycle_lock{lifecycle_latch_};
    const auto outstanding_count = outstanding_buffer_count_.load(std::memory_order_acquire);
    if (outstanding_count != 0) {
      throw InvalidArgumentError(FormatOutstandingBuffers(outstanding_count), location);
    }

    DrainRetirements(location);
    FreeCached(location);
    status_.store(PinnedAllocatorStatus::CLOSED, std::memory_order_release);
  }

  [[nodiscard]] auto TryCloseWithoutSynchronization() noexcept -> bool {
    if (outstanding_buffer_count_.load(std::memory_order_acquire) != 0 ||
        pending_retirement_count_.load(std::memory_order_acquire) != 0) {
      ReportError(ErrorCode::INTERNAL,
                  "pinned allocator was destroyed without Shutdown while buffers or retirements remain", *error_sink_,
                  MakeContext(location_));
      return false;
    }
    if (!FreeCachedNoexcept()) {
      return false;
    }
    status_.store(PinnedAllocatorStatus::CLOSED, std::memory_order_release);
    return true;
  }

  [[nodiscard]] auto GetStats() const noexcept -> PinnedAllocatorStats {
    return {
        .live_bytes_ = logical_live_bytes_.load(std::memory_order_relaxed),
        .pending_bytes_ = pending_bytes_.load(std::memory_order_relaxed),
        .cached_bytes_ = cached_bytes_.load(std::memory_order_relaxed),
        .physical_bytes_ = physical_bytes_.load(std::memory_order_relaxed),
        .peak_physical_bytes_ = peak_physical_bytes_.load(std::memory_order_relaxed),
        .host_allocation_count_ = host_allocation_count_.load(std::memory_order_relaxed),
        .host_free_count_ = host_free_count_.load(std::memory_order_relaxed),
        .cache_hit_count_ = cache_hit_count_.load(std::memory_order_relaxed),
        .retirement_count_ = retirement_count_.load(std::memory_order_relaxed),
        .pending_retirement_count_ = pending_retirement_count_.load(std::memory_order_relaxed),
        .outstanding_buffer_count_ = outstanding_buffer_count_.load(std::memory_order_relaxed),
    };
  }

 private:
  void ValidateRunning(std::source_location location) const {
    if (status_.load(std::memory_order_acquire) != PinnedAllocatorStatus::RUNNING) {
      throw InvalidArgumentError("pinned allocator is not accepting new work", location);
    }
  }

  [[nodiscard]] auto FindEventPool(Device device, std::source_location location) const
      -> const std::shared_ptr<EventPool> & {
    const auto iterator = std::ranges::find_if(
        event_pools_, [device](const auto &event_pool) { return event_pool->GetDevice() == device; });
    if (iterator == event_pools_.end()) {
      throw InternalError("pinned allocation recorded usage on an unregistered CUDA device", location);
    }
    return *iterator;
  }

  void ReserveInUse(size_t requested_bytes, size_t capacity_bytes, std::source_location location) {
    const auto capacity = static_cast<uint64_t>(capacity_bytes);
    auto current = physical_in_use_bytes_.load(std::memory_order_relaxed);
    while (true) {
      if (capacity > std::numeric_limits<uint64_t>::max() - current) {
        throw OverflowError("pinned allocator in-use byte count overflow", location);
      }
      const auto maximum = static_cast<uint64_t>(options_.max_live_bytes_);
      if (maximum != 0 && (current > maximum || capacity > maximum - current)) {
        throw OutOfMemoryError(FormatBudgetError(requested_bytes, capacity_bytes, current, maximum), location);
      }
      if (physical_in_use_bytes_.compare_exchange_weak(current, current + capacity, std::memory_order_acq_rel,
                                                       std::memory_order_relaxed)) {
        return;
      }
    }
  }

  void ReleaseInUse(size_t capacity_bytes) noexcept {
    const auto capacity = static_cast<uint64_t>(capacity_bytes);
    if (physical_in_use_bytes_.fetch_sub(capacity, std::memory_order_relaxed) < capacity) {
      std::terminate();
    }
  }

  void ReserveRetirementSlot(std::source_location location) {
    std::scoped_lock lock{retirement_latch_};
    const auto required =
        CheckedAdd(retirements_.size(),
                   CheckedAdd(reserved_retirement_slots_, size_t{1}, "pinned retirement reservation count", location),
                   "pinned retirement queue capacity", location);
    if (retirements_.capacity() < required) {
      const auto doubled = retirements_.capacity() == 0 ? size_t{16}
                                                        : CheckedMultiply(retirements_.capacity(), size_t{2},
                                                                          "pinned retirement queue growth", location);
      retirements_.reserve(std::max(required, doubled));
    }
    reserved_retirement_slots_++;
  }

  void CancelRetirementSlot() noexcept {
    std::scoped_lock lock{retirement_latch_};
    if (reserved_retirement_slots_ == 0) {
      std::terminate();
    }
    reserved_retirement_slots_--;
  }

  void Enqueue(PinnedRetirement retirement) noexcept {
    std::scoped_lock lock{retirement_latch_};
    if (reserved_retirement_slots_ == 0 || retirements_.size() == retirements_.capacity()) {
      std::terminate();
    }
    reserved_retirement_slots_--;
    retirements_.push_back(std::move(retirement));
    pending_retirement_count_.fetch_add(1, std::memory_order_relaxed);
    retirement_count_.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] auto TakeCached(size_t capacity_bytes) -> std::optional<PinnedAllocation> {
    std::scoped_lock lock{cache_latch_};
    const auto iterator = cached_allocations_.find(capacity_bytes);
    if (iterator == cached_allocations_.end() || iterator->second.empty()) {
      return std::nullopt;
    }
    auto allocation = iterator->second.back();
    iterator->second.pop_back();
    if (iterator->second.empty()) {
      cached_allocations_.erase(iterator);
    }
    cached_bytes_.fetch_sub(capacity_bytes, std::memory_order_relaxed);
    return allocation;
  }

  [[nodiscard]] auto AllocateHost(size_t requested_bytes, size_t capacity_bytes, std::source_location location)
      -> PinnedAllocation {
    void *pointer = nullptr;
    auto status = GetCudaApi().host_alloc_(&pointer, capacity_bytes, cudaHostAllocPortable);
    if (status == cudaErrorMemoryAllocation) {
      ClearExpectedAllocationError(location);
      FreeCached(location);
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
    return {
        .pointer_ = pointer,
        .capacity_bytes_ = capacity_bytes,
    };
  }

  void ClearExpectedAllocationError(std::source_location location) {
    const auto status = GetCudaApi().get_last_error_();
    if (status != cudaSuccess && status != cudaErrorMemoryAllocation) {
      CheckCuda(status, "cudaGetLastError after cudaHostAlloc", location);
    }
  }

  [[nodiscard]] auto TryFreeHost(PinnedAllocation allocation, const ErrorReportContext &context) noexcept -> bool {
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

  [[nodiscard]] auto CacheOrFreeNoexcept(PinnedAllocation allocation, std::source_location location) noexcept -> bool {
    bool cached = false;
    if (allocation.capacity_bytes_ <= options_.max_cached_bytes_) {
      try {
        std::scoped_lock lock{cache_latch_};
        const auto current = cached_bytes_.load(std::memory_order_relaxed);
        if (current <= options_.max_cached_bytes_ - allocation.capacity_bytes_) {
          cached_allocations_[allocation.capacity_bytes_].push_back(allocation);
          cached_bytes_.fetch_add(allocation.capacity_bytes_, std::memory_order_relaxed);
          cached = true;
        }
      } catch (...) {
        cached = false;
      }
    }
    if (cached) {
      return true;
    }
    return TryFreeHost(allocation, MakeContext(location));
  }

  [[nodiscard]] auto QueryRetirement(PinnedRetirement &retirement) noexcept -> bool {
    if (retirement.poisoned_) {
      return false;
    }
    for (size_t index = 0; index < retirement.events_.size(); index++) {
      const auto &stream = retirement.streams_[index];
      const auto context = MakeContext(retirement.location_, stream->GetDevice(), stream->GetId());
      CleanupDeviceGuard device_guard{stream->GetDevice(), *error_sink_, context, "query pinned-buffer retirement",
                                      "restore after pinned-buffer retirement query"};
      if (!device_guard) {
        retirement.poisoned_ = true;
        status_.store(PinnedAllocatorStatus::FAILED, std::memory_order_release);
        return false;
      }
      const auto status = GetCudaApi().query_event_(retirement.events_[index].GetNative());
      if (status == cudaErrorNotReady) {
        return false;
      }
      if (status != cudaSuccess) {
        TryCuda(status, "cudaEventQuery", "pinned-buffer retirement", *error_sink_, context);
        retirement.events_[index].Discard();
        retirement.poisoned_ = true;
        status_.store(PinnedAllocatorStatus::FAILED, std::memory_order_release);
        return false;
      }
    }
    return true;
  }

  void PollRetirements() noexcept {
    size_t index = 0;
    while (true) {
      std::optional<PinnedRetirement> completed;
      {
        std::unique_lock lock{retirement_latch_};
        if (index >= retirements_.size()) {
          return;
        }
        if (!QueryRetirement(retirements_[index])) {
          index++;
          continue;
        }
        if (!CacheOrFreeNoexcept(retirements_[index].allocation_, retirements_[index].location_)) {
          retirements_[index].poisoned_ = true;
          status_.store(PinnedAllocatorStatus::FAILED, std::memory_order_release);
          index++;
          continue;
        }
        const auto capacity_bytes = retirements_[index].allocation_.capacity_bytes_;
        completed.emplace(std::move(retirements_[index]));
        if (index + 1 != retirements_.size()) {
          retirements_[index] = std::move(retirements_.back());
        }
        retirements_.pop_back();
        pending_retirement_count_.fetch_sub(1, std::memory_order_relaxed);
        pending_bytes_.fetch_sub(capacity_bytes, std::memory_order_relaxed);
        ReleaseInUse(capacity_bytes);
      }
    }
  }

  void SynchronizeRetirement(PinnedRetirement &retirement, std::source_location location) {
    if (!retirement.poisoned_) {
      for (size_t index = 0; index < retirement.events_.size(); index++) {
        const auto &stream = retirement.streams_[index];
        DeviceGuard device_guard{stream->GetDevice(), *error_sink_, location};
        CheckCuda(GetCudaApi().synchronize_event_(retirement.events_[index].GetNative()), "cudaEventSynchronize",
                  location);
      }
      return;
    }
    for (const auto &stream : retirement.streams_) {
      DeviceGuard device_guard{stream->GetDevice(), *error_sink_, location};
      CheckCuda(GetCudaApi().synchronize_stream_(stream->GetNative()),
                "cudaStreamSynchronize (pinned-buffer retirement)", location);
    }
  }

  void DrainRetirements(std::source_location location) {
    while (true) {
      std::scoped_lock lock{retirement_latch_};
      if (retirements_.empty()) {
        return;
      }
      auto &retirement = retirements_.back();
      SynchronizeRetirement(retirement, location);
      const auto allocation = retirement.allocation_;
      CheckCuda(GetCudaApi().free_host_(allocation.pointer_), "cudaFreeHost", location);
      if (physical_bytes_.fetch_sub(allocation.capacity_bytes_, std::memory_order_relaxed) <
          allocation.capacity_bytes_) {
        std::terminate();
      }
      host_free_count_.fetch_add(1, std::memory_order_relaxed);
      pending_bytes_.fetch_sub(allocation.capacity_bytes_, std::memory_order_relaxed);
      ReleaseInUse(allocation.capacity_bytes_);
      retirements_.pop_back();
      pending_retirement_count_.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  void FreeCached(std::source_location location) {
    std::scoped_lock lock{cache_latch_};
    for (auto iterator = cached_allocations_.begin(); iterator != cached_allocations_.end();) {
      auto &bucket = iterator->second;
      while (!bucket.empty()) {
        const auto allocation = bucket.back();
        CheckCuda(GetCudaApi().free_host_(allocation.pointer_), "cudaFreeHost", location);
        bucket.pop_back();
        cached_bytes_.fetch_sub(allocation.capacity_bytes_, std::memory_order_relaxed);
        if (physical_bytes_.fetch_sub(allocation.capacity_bytes_, std::memory_order_relaxed) <
            allocation.capacity_bytes_) {
          std::terminate();
        }
        host_free_count_.fetch_add(1, std::memory_order_relaxed);
      }
      iterator = cached_allocations_.erase(iterator);
    }
  }

  [[nodiscard]] auto FreeCachedNoexcept() noexcept -> bool {
    std::scoped_lock lock{cache_latch_};
    const auto context = MakeContext(location_);
    for (auto iterator = cached_allocations_.begin(); iterator != cached_allocations_.end();) {
      auto &bucket = iterator->second;
      while (!bucket.empty()) {
        const auto allocation = bucket.back();
        if (!TryFreeHost(allocation, context)) {
          return false;
        }
        bucket.pop_back();
        cached_bytes_.fetch_sub(allocation.capacity_bytes_, std::memory_order_relaxed);
      }
      iterator = cached_allocations_.erase(iterator);
    }
    return true;
  }

  std::shared_ptr<ErrorSink> error_sink_;
  std::vector<std::shared_ptr<EventPool>> event_pools_;
  PinnedMemoryOptions options_;
  std::source_location location_;

  mutable std::shared_mutex lifecycle_latch_;
  std::mutex shutdown_latch_;
  std::atomic<PinnedAllocatorStatus> status_{PinnedAllocatorStatus::RUNNING};
  std::atomic<uint64_t> logical_live_bytes_{0};
  std::atomic<uint64_t> pending_bytes_{0};
  std::atomic<uint64_t> physical_in_use_bytes_{0};
  std::atomic<uint64_t> cached_bytes_{0};
  std::atomic<uint64_t> physical_bytes_{0};
  std::atomic<uint64_t> peak_physical_bytes_{0};
  std::atomic<uint64_t> host_allocation_count_{0};
  std::atomic<uint64_t> host_free_count_{0};
  std::atomic<uint64_t> cache_hit_count_{0};
  std::atomic<uint64_t> retirement_count_{0};
  std::atomic<uint64_t> pending_retirement_count_{0};
  std::atomic<uint64_t> outstanding_buffer_count_{0};

  std::mutex retirement_latch_;
  std::vector<PinnedRetirement> retirements_;
  size_t reserved_retirement_slots_{0};

  std::mutex cache_latch_;
  std::map<size_t, std::vector<PinnedAllocation>> cached_allocations_;
};

PinnedBlock::PinnedBlock(PinnedAllocation allocation, size_t size_bytes, std::shared_ptr<PinnedAllocator> allocator,
                         std::source_location location) noexcept
    : allocation_(allocation), size_bytes_(size_bytes), allocator_(std::move(allocator)), location_(location) {}

PinnedBlock::~PinnedBlock() noexcept {
  allocator_->Retire(allocation_, std::move(streams_), std::move(retirement_events_), location_);
}

auto PinnedBlock::GetData() const noexcept -> void * { return allocation_.pointer_; }

auto PinnedBlock::GetSizeBytes() const noexcept -> size_t { return size_bytes_; }

void PinnedBlock::RecordUsage(const Stream &stream, std::source_location location) {
  const auto stream_state = StreamAccess::GetState(stream);
  const auto stream_id = stream_state->GetId();
  std::scoped_lock lock{usage_latch_};
  if (std::ranges::any_of(streams_, [stream_id](const auto &recorded) { return recorded->GetId() == stream_id; })) {
    return;
  }
  const auto required = CheckedAdd(streams_.size(), size_t{1}, "pinned-buffer stream count", location);
  streams_.reserve(required);
  retirement_events_.reserve(required);
  streams_.push_back(stream_state);
}

auto PinnedBufferAccess::GetBlock(const PinnedBuffer &buffer, std::source_location location) -> PinnedBlock & {
  if (buffer.block_ == nullptr) {
    throw InvalidArgumentError("pinned buffer is in a moved-from state", location);
  }
  return *buffer.block_;
}

auto PinnedAllocator::Create(const std::shared_ptr<ErrorSink> &error_sink,
                             std::span<const std::shared_ptr<EventPool>> event_pools, PinnedMemoryOptions options,
                             std::source_location location) -> std::shared_ptr<PinnedAllocator> {
  ValidateDependencies(error_sink, event_pools, location);
  std::vector<std::shared_ptr<EventPool>> owned_event_pools{event_pools.begin(), event_pools.end()};
  auto impl = std::make_unique<PinnedAllocatorImpl>(error_sink, std::move(owned_event_pools), options, location);
  return std::shared_ptr<PinnedAllocator>{new PinnedAllocator(std::move(impl))};
}

PinnedAllocator::PinnedAllocator(std::unique_ptr<PinnedAllocatorImpl> impl) noexcept : impl_(std::move(impl)) {}

PinnedAllocator::~PinnedAllocator() noexcept {
  if (impl_ != nullptr && !impl_->TryCloseWithoutSynchronization()) {
    [[maybe_unused]] auto *quarantined_impl = impl_.release();
  }
}

auto PinnedAllocator::Allocate(size_t bytes, std::source_location location) -> PinnedBuffer {
  return impl_->Allocate(shared_from_this(), bytes, location);
}

void PinnedAllocator::Poll() noexcept { impl_->Poll(); }

void PinnedAllocator::Trim(std::source_location location) { impl_->Trim(location); }

void PinnedAllocator::Shutdown(std::source_location location) { impl_->Shutdown(location); }

auto PinnedAllocator::GetStats() const noexcept -> PinnedAllocatorStats { return impl_->GetStats(); }

void PinnedAllocator::Retire(PinnedAllocation allocation, std::vector<std::shared_ptr<StreamState>> streams,
                             std::vector<PooledEvent> events, std::source_location location) noexcept {
  impl_->Retire(allocation, std::move(streams), std::move(events), location);
}

}  // namespace ttl::internal
