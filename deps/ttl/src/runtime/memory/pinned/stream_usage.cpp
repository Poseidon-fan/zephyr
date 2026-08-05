#include "ttl/internal/runtime/memory/pinned/stream_usage.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <source_location>
#include <utility>

#include "ttl/common/error.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"

namespace ttl::internal {

void PinnedStreamUsage::Record(const std::shared_ptr<StreamState> &stream, std::source_location location) {
  if (stream == nullptr) {
    throw InvalidArgumentError("pinned stream usage requires a non-null stream state", location);
  }
  const auto stream_id = stream->GetId();
  std::scoped_lock lock{latch_};
  if (std::ranges::any_of(entries_, [stream_id](const auto &entry) { return entry.stream_->GetId() == stream_id; })) {
    return;
  }
  entries_.push_back({
      .stream_ = stream,
      .completion_event_ = std::nullopt,
  });
}

auto PinnedStreamUsage::TakeSnapshot() && noexcept -> PinnedStreamUsageSnapshot {
  return {.entries_ = std::move(entries_)};
}

}  // namespace ttl::internal
