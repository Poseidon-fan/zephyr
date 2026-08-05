#pragma once

#include <cstddef>
#include <memory>
#include <source_location>

#include <cuda_runtime_api.h>

#include "ttl/common/device.hpp"
#include "ttl/common/error_sink.hpp"

namespace ttl::internal {

class EventPoolState;

/** Snapshot of one per-device event pool's host-side state. */
struct EventPoolStats final {
  size_t cached_event_count_;
  size_t outstanding_event_count_;
  size_t max_cached_event_count_;
  bool is_closed_;
};

/**
 * @brief Move-only lease of one timing-disabled CUDA event.
 *
 * Destruction returns the event to its pool. The owner must therefore retain the lease until the recorded state is no
 * longer observed. Call Discard after a CUDA failure that makes the event unsuitable for reuse. A lease must not be
 * released or discarded from a CUDA host callback because either path may call the CUDA Runtime.
 */
class PooledEvent final {
 public:
  PooledEvent(const PooledEvent &) = delete;
  auto operator=(const PooledEvent &) -> PooledEvent & = delete;

  PooledEvent(PooledEvent &&other) noexcept;
  auto operator=(PooledEvent &&other) noexcept -> PooledEvent &;

  ~PooledEvent() noexcept;

  [[nodiscard]] auto GetNative() const noexcept -> cudaEvent_t;
  [[nodiscard]] explicit operator bool() const noexcept;

  /** Remove this event instead of returning it to the reusable cache, destroying it when CUDA cleanup is available. */
  void Discard() noexcept;

 private:
  friend class EventPoolState;

  PooledEvent(cudaEvent_t event, std::shared_ptr<EventPoolState> pool) noexcept;
  void Reset(bool reusable) noexcept;

  cudaEvent_t event_;
  std::shared_ptr<EventPoolState> pool_;
};

/**
 * @brief Thread-safe, per-device pool of timing-disabled CUDA events for internal high-frequency fences.
 *
 * This pool is independent from the public immutable Event type. Closing rejects new acquisitions, destroys cached
 * events, and makes outstanding leases destroy their events when released.
 */
class EventPool final {
 public:
  EventPool(Device device, std::shared_ptr<ErrorSink> error_sink, size_t max_cached_event_count,
            std::source_location location = std::source_location::current());

  EventPool(const EventPool &) = delete;
  auto operator=(const EventPool &) -> EventPool & = delete;
  EventPool(EventPool &&) = delete;
  auto operator=(EventPool &&) -> EventPool & = delete;

  ~EventPool() noexcept;

  [[nodiscard]] auto Acquire(std::source_location location = std::source_location::current()) -> PooledEvent;

  /** Ensure that at least count reusable events can be retained without creating events under steady-state load. */
  void Reserve(size_t count, std::source_location location = std::source_location::current());

  /** Idempotently reject new acquisitions and destroy all currently cached events. */
  void Close() noexcept;

  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetStats() const -> EventPoolStats;

 private:
  std::shared_ptr<EventPoolState> state_;
};

}  // namespace ttl::internal
