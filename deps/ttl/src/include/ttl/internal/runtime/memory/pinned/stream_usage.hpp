#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <vector>

#include "ttl/internal/runtime/execution/event_pool.hpp"

namespace ttl::internal {

class StreamState;

struct PinnedStreamUsageEntry final {
  std::shared_ptr<StreamState> stream_;
  std::optional<PooledEvent> completion_event_;
};

struct PinnedStreamUsageSnapshot final {
  std::vector<PinnedStreamUsageEntry> entries_;
};

/**
 * @brief Thread-safe stream lifetime leases associated with one pinned allocation.
 *
 * Every distinct stream is retained with an empty event slot. Retirement fills those slots without growing the
 * container. TakeSnapshot requires exclusive ownership after no Record call can begin or remain in progress.
 */
class PinnedStreamUsage final {
 public:
  PinnedStreamUsage() noexcept = default;

  PinnedStreamUsage(const PinnedStreamUsage &) = delete;
  auto operator=(const PinnedStreamUsage &) -> PinnedStreamUsage & = delete;
  PinnedStreamUsage(PinnedStreamUsage &&) = delete;
  auto operator=(PinnedStreamUsage &&) -> PinnedStreamUsage & = delete;

  /** Retain one stream exactly once. This may allocate and therefore may throw. */
  void Record(const std::shared_ptr<StreamState> &stream,
              std::source_location location = std::source_location::current());

  /** Transfer every retained stream and its event slot. No concurrent Record call is permitted. */
  [[nodiscard]] auto TakeSnapshot() && noexcept -> PinnedStreamUsageSnapshot;

 private:
  std::mutex latch_;
  std::vector<PinnedStreamUsageEntry> entries_;
};

}  // namespace ttl::internal
