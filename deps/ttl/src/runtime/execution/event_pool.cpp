#include "ttl/internal/runtime/execution/event_pool.hpp"

#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"

namespace ttl::internal {
namespace {

void ValidateErrorSink(const std::shared_ptr<ErrorSink> &error_sink, std::source_location location) {
  if (error_sink == nullptr) {
    throw InvalidArgumentError("event pool error sink must not be null", location);
  }
}

[[nodiscard]] auto FormatReserveError(size_t count, size_t maximum) -> std::string {
  std::string message{"cannot reserve "};
  message.append(std::to_string(count));
  message.append(" events in a pool with capacity ");
  message.append(std::to_string(maximum));
  return message;
}

}  // namespace

class EventPoolState final : public std::enable_shared_from_this<EventPoolState> {
 public:
  EventPoolState(Device device, std::shared_ptr<ErrorSink> error_sink, size_t max_cached_event_count,
                 std::source_location location)
      : device_(device),
        error_sink_(std::move(error_sink)),
        max_cached_event_count_(max_cached_event_count),
        location_(location) {
    ValidateErrorSink(error_sink_, location_);
    cached_events_.reserve(max_cached_event_count_);
  }

  EventPoolState(const EventPoolState &) = delete;
  auto operator=(const EventPoolState &) -> EventPoolState & = delete;
  EventPoolState(EventPoolState &&) = delete;
  auto operator=(EventPoolState &&) -> EventPoolState & = delete;

  ~EventPoolState() noexcept { Close(); }

  [[nodiscard]] auto Acquire(std::source_location location) -> PooledEvent {
    {
      std::scoped_lock lock{latch_};
      if (is_closed_) {
        throw InvalidArgumentError("cannot acquire an event from a closed event pool", location);
      }
      if (outstanding_event_count_ == std::numeric_limits<size_t>::max()) {
        throw OverflowError("event pool outstanding count overflow", location);
      }
      if (!cached_events_.empty()) {
        const auto event = cached_events_.back();
        cached_events_.pop_back();
        outstanding_event_count_++;
        return PooledEvent{event, shared_from_this()};
      }
    }

    cudaEvent_t event = nullptr;
    DeviceGuard device_guard{device_, *error_sink_, location};
    CheckCuda(GetCudaApi().create_event_with_flags_(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags",
              location);
    if (event == nullptr) {
      throw InternalError("cudaEventCreateWithFlags returned a null event", location);
    }

    bool accepted = false;
    bool overflow = false;
    {
      std::scoped_lock lock{latch_};
      if (!is_closed_) {
        if (outstanding_event_count_ == std::numeric_limits<size_t>::max()) {
          overflow = true;
        } else {
          outstanding_event_count_++;
          accepted = true;
        }
      }
    }

    if (accepted) {
      return PooledEvent{event, shared_from_this()};
    }

    DestroyEvents(std::span<const cudaEvent_t>{&event, 1});
    if (overflow) {
      throw OverflowError("event pool outstanding count overflow", location);
    }
    throw InvalidArgumentError("event pool closed while creating an event", location);
  }

  void Release(cudaEvent_t event, bool reusable) noexcept {
    bool cache_event = false;
    {
      std::scoped_lock lock{latch_};
      if (outstanding_event_count_ > 0) {
        outstanding_event_count_--;
      } else {
        reusable = false;
      }

      if (reusable && !is_closed_ && cached_events_.size() < max_cached_event_count_) {
        cached_events_.push_back(event);
        cache_event = true;
      }
    }

    if (cache_event) {
      return;
    }
    DestroyEvents(std::span<const cudaEvent_t>{&event, 1});
  }

  void Close() noexcept {
    std::vector<cudaEvent_t> cached_events;
    {
      std::scoped_lock lock{latch_};
      if (is_closed_) {
        return;
      }
      is_closed_ = true;
      cached_events.swap(cached_events_);
    }
    DestroyEvents(cached_events);
  }

  [[nodiscard]] auto GetDevice() const noexcept -> Device { return device_; }

  [[nodiscard]] auto GetStats() const -> EventPoolStats {
    std::scoped_lock lock{latch_};
    return EventPoolStats{
        .cached_event_count_ = cached_events_.size(),
        .outstanding_event_count_ = outstanding_event_count_,
        .max_cached_event_count_ = max_cached_event_count_,
        .is_closed_ = is_closed_,
    };
  }

 private:
  void DestroyEvents(std::span<const cudaEvent_t> events) noexcept {
    if (events.empty()) {
      return;
    }

    const ErrorReportContext context{
        .location_ = location_,
        .device_ = device_,
        .stream_id_ = std::nullopt,
    };
    CleanupDeviceGuard device_guard{device_, *error_sink_, context, "destroy event pool resource",
                                    "restore after event pool cleanup"};
    if (!device_guard) {
      return;
    }

    const auto &cuda_api = GetCudaApi();
    for (const auto event : events) {
      TryCuda(cuda_api.destroy_event_(event), "cudaEventDestroy", "event pool", *error_sink_, context);
    }
  }

  Device device_;
  std::shared_ptr<ErrorSink> error_sink_;
  size_t max_cached_event_count_;
  std::source_location location_;
  mutable std::mutex latch_;
  std::vector<cudaEvent_t> cached_events_;
  size_t outstanding_event_count_{0};
  bool is_closed_{false};
};

PooledEvent::PooledEvent(cudaEvent_t event, std::shared_ptr<EventPoolState> pool) noexcept
    : event_(event), pool_(std::move(pool)) {}

PooledEvent::PooledEvent(PooledEvent &&other) noexcept
    : event_(std::exchange(other.event_, nullptr)), pool_(std::move(other.pool_)) {}

auto PooledEvent::operator=(PooledEvent &&other) noexcept -> PooledEvent & {
  if (this != &other) {
    Reset(true);
    event_ = std::exchange(other.event_, nullptr);
    pool_ = std::move(other.pool_);
  }
  return *this;
}

PooledEvent::~PooledEvent() noexcept { Reset(true); }

auto PooledEvent::GetNative() const noexcept -> cudaEvent_t { return event_; }

PooledEvent::operator bool() const noexcept { return event_ != nullptr; }

void PooledEvent::Discard() noexcept { Reset(false); }

void PooledEvent::Reset(bool reusable) noexcept {
  if (event_ == nullptr) {
    return;
  }

  const auto event = std::exchange(event_, nullptr);
  auto pool = std::move(pool_);
  pool->Release(event, reusable);
}

EventPool::EventPool(Device device, std::shared_ptr<ErrorSink> error_sink, size_t max_cached_event_count,
                     std::source_location location)
    : state_(std::make_shared<EventPoolState>(device, std::move(error_sink), max_cached_event_count, location)) {}

EventPool::~EventPool() noexcept { Close(); }

auto EventPool::Acquire(std::source_location location) -> PooledEvent {
  const auto state = state_;
  return state->Acquire(location);
}

void EventPool::Reserve(size_t count, std::source_location location) {
  const auto stats = GetStats();
  if (count > stats.max_cached_event_count_) {
    throw InvalidArgumentError(FormatReserveError(count, stats.max_cached_event_count_), location);
  }

  std::vector<PooledEvent> events;
  events.reserve(count);
  for (size_t index = 0; index < count; index++) {
    events.push_back(Acquire(location));
  }
}

void EventPool::Close() noexcept { state_->Close(); }

auto EventPool::GetDevice() const noexcept -> Device { return state_->GetDevice(); }

auto EventPool::GetStats() const -> EventPoolStats { return state_->GetStats(); }

}  // namespace ttl::internal
