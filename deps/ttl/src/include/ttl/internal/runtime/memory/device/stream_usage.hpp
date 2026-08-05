#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <source_location>
#include <vector>

namespace ttl::internal {

class StreamState;

struct DeviceStreamUsageSnapshot final {
  std::shared_ptr<StreamState> allocation_stream_;
  std::vector<std::shared_ptr<StreamState>> side_streams_;
};

/**
 * @brief Thread-safe stream lifetime leases associated with one device allocation.
 *
 * The allocation stream is retained separately from streams that use the allocation later. Record may be called
 * concurrently. TakeSnapshot requires exclusive ownership after no Record call can begin or remain in progress.
 */
class DeviceStreamUsage final {
 public:
  explicit DeviceStreamUsage(std::shared_ptr<StreamState> allocation_stream) noexcept;

  DeviceStreamUsage(const DeviceStreamUsage &) = delete;
  auto operator=(const DeviceStreamUsage &) -> DeviceStreamUsage & = delete;
  DeviceStreamUsage(DeviceStreamUsage &&) = delete;
  auto operator=(DeviceStreamUsage &&) -> DeviceStreamUsage & = delete;

  /** Retain one non-allocation stream exactly once. This may allocate and therefore may throw. */
  void Record(const std::shared_ptr<StreamState> &stream,
              std::source_location location = std::source_location::current());

  /** Transfer every retained stream. No concurrent Record call is permitted. */
  [[nodiscard]] auto TakeSnapshot() && noexcept -> DeviceStreamUsageSnapshot;

 private:
  std::shared_ptr<StreamState> allocation_stream_;
  std::atomic<uint64_t> most_recent_stream_id_;
  std::mutex latch_;
  std::vector<std::shared_ptr<StreamState>> side_streams_;
};

}  // namespace ttl::internal
