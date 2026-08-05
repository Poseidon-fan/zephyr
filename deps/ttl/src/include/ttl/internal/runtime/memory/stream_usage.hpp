#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <source_location>
#include <vector>

namespace ttl::internal {

class StreamState;

/**
 * @brief Thread-safe stream lifetime leases associated with one allocation.
 *
 * The allocation stream is retained separately from streams that use the allocation later. Record may be called
 * concurrently. TakeSideStreams requires exclusive ownership and must only be called after no Record call can begin or
 * remain in progress.
 */
class StreamUsage final {
 public:
  explicit StreamUsage(std::shared_ptr<StreamState> allocation_stream,
                       std::source_location location = std::source_location::current());

  StreamUsage(const StreamUsage &) = delete;
  auto operator=(const StreamUsage &) -> StreamUsage & = delete;
  StreamUsage(StreamUsage &&) = delete;
  auto operator=(StreamUsage &&) -> StreamUsage & = delete;

  /** Retain one stream exactly once. This may allocate and therefore may throw. */
  void Record(const std::shared_ptr<StreamState> &stream,
              std::source_location location = std::source_location::current());

  /** Return a lifetime lease for the stream on which the allocation was created. */
  [[nodiscard]] auto GetAllocationStream() const noexcept -> std::shared_ptr<StreamState>;

  /** Transfer the retained non-allocation streams. No concurrent Record call is permitted. */
  [[nodiscard]] auto TakeSideStreams() && noexcept -> std::vector<std::shared_ptr<StreamState>>;

 private:
  std::shared_ptr<StreamState> allocation_stream_;
  std::atomic<uint64_t> most_recent_stream_id_;
  std::mutex latch_;
  std::vector<std::shared_ptr<StreamState>> side_streams_;
};

}  // namespace ttl::internal
