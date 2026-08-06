#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "ttl/internal/runtime/execution/event_pool.hpp"
#include "ttl/internal/runtime/memory/device/allocation.hpp"
#include "ttl/internal/runtime/memory/device/stream_usage.hpp"

namespace ttl::internal {

class StreamState;

/** Device allocation and completion state retained until asynchronous retirement finishes. */
struct DeviceRetirement final {
  DeviceRetirement(Allocation allocation, DeviceStreamUsageSnapshot usage) noexcept
      : allocation_(std::move(allocation)),
        allocation_stream_(std::move(usage.allocation_stream_)),
        side_streams_(std::move(usage.side_streams_)) {}

  DeviceRetirement(const DeviceRetirement &) = delete;
  auto operator=(const DeviceRetirement &) -> DeviceRetirement & = delete;
  DeviceRetirement(DeviceRetirement &&) noexcept = default;
  auto operator=(DeviceRetirement &&) noexcept -> DeviceRetirement & = default;

  Allocation allocation_;
  std::shared_ptr<StreamState> allocation_stream_;
  std::vector<std::shared_ptr<StreamState>> side_streams_;
  std::optional<PooledEvent> completion_event_;
  bool free_attempted_{false};
  bool free_submitted_{false};
  bool uses_reclaim_stream_{false};
  bool poisoned_{false};
};

}  // namespace ttl::internal
