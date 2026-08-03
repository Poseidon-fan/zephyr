#include "ttl/internal/runtime/memory/stream_usage.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <source_location>
#include <utility>
#include <vector>

#include "ttl/common/error.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"

namespace ttl::internal {
namespace {

[[nodiscard]] auto ValidateStream(const std::shared_ptr<StreamState> &stream, std::source_location location)
    -> uint64_t {
  if (stream == nullptr) {
    throw InvalidArgumentError("stream usage requires a non-null stream state", location);
  }
  return stream->GetId();
}

}  // namespace

StreamUsage::StreamUsage(std::shared_ptr<StreamState> allocation_stream, std::source_location location)
    : allocation_stream_(std::move(allocation_stream)),
      most_recent_stream_id_(ValidateStream(allocation_stream_, location)) {}

void StreamUsage::Record(const std::shared_ptr<StreamState> &stream, std::source_location location) {
  const auto stream_id = ValidateStream(stream, location);
  if (most_recent_stream_id_.load(std::memory_order_acquire) == stream_id) {
    return;
  }

  if (stream_id == allocation_stream_->GetId()) {
    most_recent_stream_id_.store(stream_id, std::memory_order_release);
    return;
  }

  std::scoped_lock lock{latch_};
  const auto iterator =
      std::ranges::find_if(side_streams_, [stream_id](const std::shared_ptr<StreamState> &recorded_stream) {
        return recorded_stream->GetId() == stream_id;
      });
  if (iterator == side_streams_.end()) {
    side_streams_.push_back(stream);
  }
  most_recent_stream_id_.store(stream_id, std::memory_order_release);
}

auto StreamUsage::GetAllocationStream() const noexcept -> std::shared_ptr<StreamState> { return allocation_stream_; }

auto StreamUsage::TakeSideStreams() && noexcept -> std::vector<std::shared_ptr<StreamState>> {
  return std::move(side_streams_);
}

}  // namespace ttl::internal
