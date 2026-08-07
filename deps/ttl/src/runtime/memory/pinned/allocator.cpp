#include "ttl/internal/runtime/memory/pinned/allocator.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
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

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"
#include "ttl/internal/runtime/error_report.hpp"
#include "ttl/internal/runtime/execution/event_pool.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/memory/allocation_budget.hpp"
#include "ttl/internal/runtime/memory/allocator_lifecycle.hpp"
#include "ttl/internal/runtime/memory/pinned/block.hpp"
#include "ttl/internal/runtime/memory/pinned/cache.hpp"
#include "ttl/internal/runtime/memory/pinned/retirement.hpp"
#include "ttl/internal/runtime/memory/pinned/stream_usage.hpp"
#include "ttl/internal/runtime/memory/retirement_queue.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {
namespace {

[[nodiscard]] auto MakeContext(std::source_location location, std::optional<Device> device = std::nullopt,
                               std::optional<uint64_t> stream_id = std::nullopt) noexcept -> ErrorReportContext {
  return {
      .location_ = location,
      .device_ = device,
      .stream_id_ = stream_id,
  };
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

void SubtractOrTerminate(std::atomic<uint64_t> &counter, uint64_t value) noexcept {
  if (counter.fetch_sub(value, std::memory_order_relaxed) < value) {
    std::terminate();
  }
}

}  // namespace

class PinnedAllocatorImpl final {
 public:
  PinnedAllocatorImpl(std::shared_ptr<ErrorSink> error_sink, std::vector<std::shared_ptr<EventPool>> event_pools,
                      PinnedMemoryOptions options, std::source_location location) noexcept
      : error_sink_(std::move(error_sink)),
        event_pools_(std::move(event_pools)),
        location_(location),
        budget_(static_cast<uint64_t>(options.max_live_bytes_)),
        cache_(error_sink_, options.max_cached_bytes_, location) {}

  PinnedAllocatorImpl(const PinnedAllocatorImpl &) = delete;
  auto operator=(const PinnedAllocatorImpl &) -> PinnedAllocatorImpl & = delete;
  PinnedAllocatorImpl(PinnedAllocatorImpl &&) = delete;
  auto operator=(PinnedAllocatorImpl &&) -> PinnedAllocatorImpl & = delete;

  ~PinnedAllocatorImpl() noexcept = default;

  [[nodiscard]] auto Allocate(const std::shared_ptr<PinnedAllocator> &allocator, size_t bytes,
                              std::source_location location) -> PinnedBuffer {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    lifecycle_.RequireRunning("pinned allocator", location);
    const auto capacity_bytes = PinnedMemoryCache::GetSizeClass(bytes, location);
    if (capacity_bytes == 0) {
      outstanding_buffer_count_.fetch_add(1, std::memory_order_relaxed);
      try {
        return PinnedBuffer{std::make_shared<PinnedBlock>(
            PinnedAllocation{
                .pointer_ = nullptr,
                .capacity_bytes_ = 0,
            },
            0, allocator, AllocationBudget::Reservation{}, RetirementTicket{}, location)};
      } catch (...) {
        outstanding_buffer_count_.fetch_sub(1, std::memory_order_relaxed);
        throw;
      }
    }

    PollRetirements();
    lifecycle_.RequireRunning("pinned allocator", location);
    auto budget_result = budget_.Reserve(static_cast<uint64_t>(capacity_bytes));
    if (budget_result.failure_ == BudgetFailure::OVERFLOW) {
      throw OverflowError("pinned allocator in-use byte count overflow", location);
    }
    if (budget_result.failure_ == BudgetFailure::LIMIT) {
      throw OutOfMemoryError(
          FormatBudgetError(bytes, capacity_bytes, budget_result.previous_bytes_, budget_.GetMaximumBytes()), location);
    }
    auto budget = std::move(*budget_result.reservation_);
    auto retirement_ticket = retirements_.Reserve(location);

    const auto allocation = cache_.Acquire(bytes, capacity_bytes, location);

    logical_live_bytes_.fetch_add(static_cast<uint64_t>(bytes), std::memory_order_relaxed);
    live_capacity_bytes_.fetch_add(static_cast<uint64_t>(capacity_bytes), std::memory_order_relaxed);
    outstanding_buffer_count_.fetch_add(1, std::memory_order_relaxed);
    try {
      auto block = std::make_shared<PinnedBlock>(allocation, bytes, allocator, std::move(budget),
                                                 std::move(retirement_ticket), location);
      return PinnedBuffer{std::move(block)};
    } catch (...) {
      outstanding_buffer_count_.fetch_sub(1, std::memory_order_relaxed);
      SubtractOrTerminate(logical_live_bytes_, static_cast<uint64_t>(bytes));
      SubtractOrTerminate(live_capacity_bytes_, static_cast<uint64_t>(capacity_bytes));
      if (!cache_.Release(allocation, location)) {
        retiring_capacity_bytes_.fetch_add(capacity_bytes, std::memory_order_relaxed);
        auto retirement = PinnedRetirement{allocation, PinnedStreamUsageSnapshot{}, std::move(budget), location};
        Quarantine(retirement);
        retirements_.Enqueue(std::move(retirement_ticket), std::move(retirement));
      }
      throw;
    }
  }

  void Retire(PinnedAllocation allocation, size_t logical_size_bytes, AllocationBudget::Reservation budget,
              RetirementTicket retirement_ticket, PinnedStreamUsageSnapshot usage,
              std::source_location location) noexcept {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    if (outstanding_buffer_count_.fetch_sub(1, std::memory_order_release) == 0) {
      std::terminate();
    }
    if (allocation.capacity_bytes_ == 0) {
      return;
    }

    const auto capacity_bytes = static_cast<uint64_t>(allocation.capacity_bytes_);
    SubtractOrTerminate(logical_live_bytes_, static_cast<uint64_t>(logical_size_bytes));
    SubtractOrTerminate(live_capacity_bytes_, capacity_bytes);
    retiring_capacity_bytes_.fetch_add(capacity_bytes, std::memory_order_relaxed);
    PinnedRetirement retirement{allocation, std::move(usage), std::move(budget), location};
    if (retirement.usage_.entries_.empty()) {
      if (cache_.Release(retirement.allocation_, retirement.location_)) {
        SubtractOrTerminate(retiring_capacity_bytes_, capacity_bytes);
      } else {
        Quarantine(retirement);
        retirements_.Enqueue(std::move(retirement_ticket), std::move(retirement));
      }
      return;
    }

    try {
      // Pinned-buffer usage may span devices, so record one completion event per distinct usage stream. Reuse is
      // allowed only after every event succeeds; any ambiguous record is quarantined for a synchronized drain.
      for (auto &entry : retirement.usage_.entries_) {
        const auto &stream = entry.stream_;
        auto event = FindEventPool(stream->GetDevice(), location)->Acquire(location);
        const auto context = MakeContext(location, stream->GetDevice(), stream->GetId());
        CleanupDeviceGuard device_guard{stream->GetDevice(), *error_sink_, context, "record pinned-buffer usage",
                                        "restore after pinned-buffer usage record"};
        if (!device_guard || !TryCuda(GetCudaApi().record_event_(event.GetNative(), stream->GetNative()),
                                      "cudaEventRecord", "pinned-buffer retirement", *error_sink_, context)) {
          event.Discard();
          Quarantine(retirement);
          break;
        }
        entry.completion_event_.emplace(std::move(event));
      }
    } catch (const Error &error) {
      ReportErrorNoexcept(error.GetCode(), error.GetMessage(), *error_sink_, MakeContext(location));
      Quarantine(retirement);
    } catch (const std::exception &error) {
      ReportErrorNoexcept(ErrorCode::INTERNAL, error.what(), *error_sink_, MakeContext(location));
      Quarantine(retirement);
    } catch (...) {
      ReportErrorNoexcept(ErrorCode::INTERNAL, "unknown failure while retiring a pinned host allocation", *error_sink_,
                          MakeContext(location));
      Quarantine(retirement);
    }

    retirements_.Enqueue(std::move(retirement_ticket), std::move(retirement));
  }

  void Poll() noexcept {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    PollRetirements();
  }

  void Trim(std::source_location location) {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    lifecycle_.RequireRunning("pinned allocator", location);
    PollRetirements();
    cache_.Trim(location);
  }

  void Shutdown(std::source_location location) {
    const std::scoped_lock shutdown_lock{shutdown_latch_};
    if (!lifecycle_.BeginClosing()) {
      return;
    }

    const std::unique_lock lifecycle_lock{lifecycle_latch_};
    const auto outstanding_count = outstanding_buffer_count_.load(std::memory_order_acquire);
    if (outstanding_count != 0) {
      throw InvalidArgumentError(FormatOutstandingBuffers(outstanding_count), location);
    }

    DrainRetirements(location);
    cache_.Trim(location);
    lifecycle_.MarkClosed();
  }

  [[nodiscard]] auto TryCloseWithoutSynchronization() noexcept -> bool {
    if (outstanding_buffer_count_.load(std::memory_order_acquire) != 0 || retirements_.GetPendingCount() != 0) {
      ReportErrorNoexcept(ErrorCode::INTERNAL,
                          "pinned allocator was destroyed without Shutdown while buffers or retirements remain",
                          *error_sink_, MakeContext(location_));
      return false;
    }
    if (!cache_.TryTrimNoexcept()) {
      return false;
    }
    lifecycle_.MarkClosed();
    return true;
  }

  [[nodiscard]] auto GetStats() const noexcept -> PinnedAllocatorStats {
    const auto cache = cache_.GetStats();
    return {
        .logical_live_bytes_ = logical_live_bytes_.load(std::memory_order_relaxed),
        .live_capacity_bytes_ = live_capacity_bytes_.load(std::memory_order_relaxed),
        .retiring_capacity_bytes_ = retiring_capacity_bytes_.load(std::memory_order_relaxed),
        .cached_capacity_bytes_ = cache.cached_bytes_,
        .budgeted_capacity_bytes_ = budget_.GetCurrentBytes(),
        .peak_budgeted_capacity_bytes_ = budget_.GetPeakBytes(),
        .physical_bytes_ = cache.physical_bytes_,
        .peak_physical_bytes_ = cache.peak_physical_bytes_,
        .host_allocation_count_ = cache.host_allocation_count_,
        .host_free_count_ = cache.host_free_count_,
        .cache_hit_count_ = cache.cache_hit_count_,
        .retirement_count_ = retirements_.GetTotalCount(),
        .pending_retirement_count_ = retirements_.GetPendingCount(),
        .quarantined_retirement_count_ = quarantined_retirement_count_.load(std::memory_order_relaxed),
        .quarantined_bytes_ = quarantined_bytes_.load(std::memory_order_relaxed),
        .outstanding_buffer_count_ = outstanding_buffer_count_.load(std::memory_order_relaxed),
    };
  }

 private:
  [[nodiscard]] auto FindEventPool(Device device, std::source_location location) const
      -> const std::shared_ptr<EventPool> & {
    const auto iterator = std::ranges::find_if(
        event_pools_, [device](const auto &event_pool) { return event_pool->GetDevice() == device; });
    if (iterator == event_pools_.end()) {
      throw InternalError("pinned allocation recorded usage on an unregistered CUDA device", location);
    }
    return *iterator;
  }

  void Quarantine(PinnedRetirement &retirement) noexcept {
    if (!retirement.poisoned_) {
      retirement.poisoned_ = true;
      quarantined_retirement_count_.fetch_add(1, std::memory_order_relaxed);
      quarantined_bytes_.fetch_add(retirement.allocation_.capacity_bytes_, std::memory_order_relaxed);
    }
    lifecycle_.MarkFailed();
  }

  [[nodiscard]] auto QueryRetirement(PinnedRetirement &retirement) noexcept -> bool {
    if (retirement.poisoned_) {
      return false;
    }
    for (auto &entry : retirement.usage_.entries_) {
      const auto &stream = entry.stream_;
      const auto context = MakeContext(retirement.location_, stream->GetDevice(), stream->GetId());
      if (!entry.completion_event_.has_value()) {
        ReportErrorNoexcept(ErrorCode::INTERNAL, "pinned retirement is missing a completion event", *error_sink_,
                            context);
        Quarantine(retirement);
        return false;
      }
      CleanupDeviceGuard device_guard{stream->GetDevice(), *error_sink_, context, "query pinned-buffer retirement",
                                      "restore after pinned-buffer retirement query"};
      if (!device_guard) {
        Quarantine(retirement);
        return false;
      }
      const auto readiness = TryQueryCudaEvent(entry.completion_event_->GetNative(),
                                               "cudaEventQuery (pinned-buffer retirement)", *error_sink_, context);
      if (readiness == CudaReadiness::NOT_READY) {
        return false;
      }
      if (readiness == CudaReadiness::ERROR) {
        entry.completion_event_->Discard();
        Quarantine(retirement);
        return false;
      }
    }
    return true;
  }

  void PollRetirements() noexcept {
    auto remaining = retirements_.GetAvailableCount();
    while (remaining > 0) {
      remaining--;
      auto checkout = retirements_.CheckoutFront();
      if (!checkout.has_value()) {
        return;
      }
      auto &retirement = checkout->Get();
      if (!QueryRetirement(retirement)) {
        continue;
      }
      if (!cache_.Release(retirement.allocation_, retirement.location_)) {
        Quarantine(retirement);
        return;
      }
      SubtractOrTerminate(retiring_capacity_bytes_, retirement.allocation_.capacity_bytes_);
      checkout->Complete();
    }
  }

  void SynchronizeRetirement(PinnedRetirement &retirement, std::source_location location) {
    if (!retirement.poisoned_) {
      for (const auto &entry : retirement.usage_.entries_) {
        if (!entry.completion_event_.has_value()) {
          throw InternalError("pinned retirement is missing a completion event", location);
        }
        const auto &stream = entry.stream_;
        DeviceGuard device_guard{stream->GetDevice(), *error_sink_, location};
        CheckCuda(GetCudaApi().synchronize_event_(entry.completion_event_->GetNative()), "cudaEventSynchronize",
                  location);
      }
      return;
    }
    for (const auto &entry : retirement.usage_.entries_) {
      const auto &stream = entry.stream_;
      DeviceGuard device_guard{stream->GetDevice(), *error_sink_, location};
      CheckCuda(GetCudaApi().synchronize_stream_(stream->GetNative()),
                "cudaStreamSynchronize (pinned-buffer retirement)", location);
    }
  }

  void DrainRetirements(std::source_location location) {
    while (true) {
      auto checkout = retirements_.CheckoutFront();
      if (!checkout.has_value()) {
        return;
      }
      auto &retirement = checkout->Get();
      SynchronizeRetirement(retirement, location);
      const auto allocation = retirement.allocation_;
      cache_.Free(allocation, location);
      SubtractOrTerminate(retiring_capacity_bytes_, allocation.capacity_bytes_);
      if (retirement.poisoned_) {
        quarantined_retirement_count_.fetch_sub(1, std::memory_order_relaxed);
        quarantined_bytes_.fetch_sub(allocation.capacity_bytes_, std::memory_order_relaxed);
      }
      checkout->Complete();
    }
  }

  std::shared_ptr<ErrorSink> error_sink_;
  std::vector<std::shared_ptr<EventPool>> event_pools_;
  std::source_location location_;
  AllocationBudget budget_;
  PinnedMemoryCache cache_;

  mutable std::shared_mutex lifecycle_latch_;
  std::mutex shutdown_latch_;
  AllocatorLifecycle lifecycle_;
  std::atomic<uint64_t> logical_live_bytes_{0};
  std::atomic<uint64_t> live_capacity_bytes_{0};
  std::atomic<uint64_t> retiring_capacity_bytes_{0};
  std::atomic<uint64_t> quarantined_retirement_count_{0};
  std::atomic<uint64_t> quarantined_bytes_{0};
  std::atomic<uint64_t> outstanding_buffer_count_{0};

  RetirementQueue<PinnedRetirement> retirements_;
};

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

void PinnedAllocator::Retire(PinnedAllocation allocation, size_t logical_size_bytes,
                             AllocationBudget::Reservation budget, RetirementTicket retirement_ticket,
                             PinnedStreamUsageSnapshot usage, std::source_location location) noexcept {
  impl_->Retire(allocation, logical_size_bytes, std::move(budget), std::move(retirement_ticket), std::move(usage),
                location);
}

}  // namespace ttl::internal
