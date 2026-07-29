#include "ttl/internal/stream_usage.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {
namespace {

constexpr size_t FAKE_STREAM_COUNT = 32;
std::array<int, FAKE_STREAM_COUNT> fake_stream_storage;

class IgnoreErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord /* error */) noexcept override {}
};

[[nodiscard]] auto MakeExternalStream(size_t index, Device device, const std::shared_ptr<void> &owner,
                                      const std::shared_ptr<ErrorSink> &error_sink) -> Stream {
  return StreamAccess::WrapExternal(device, reinterpret_cast<cudaStream_t>(&fake_stream_storage.at(index)), owner,
                                    error_sink);
}

TEST(StreamUsageTest, KeepsTheAllocationStreamSeparateFromSideStreams) {
  const auto error_sink = std::make_shared<IgnoreErrorSink>();
  const auto allocation_stream = MakeExternalStream(0, Device{0}, nullptr, error_sink);
  const auto allocation_state = StreamAccess::GetState(allocation_stream);
  StreamUsage usage{allocation_state};

  for (size_t iteration = 0; iteration < 100; iteration++) {
    usage.Record(allocation_state);
  }

  EXPECT_EQ(usage.GetAllocationStream(), allocation_state);
  EXPECT_TRUE(std::move(usage).TakeSideStreams().empty());
}

TEST(StreamUsageTest, RetainsEveryDistinctStreamExactlyOnceAcrossDevices) {
  const auto error_sink = std::make_shared<IgnoreErrorSink>();
  const auto allocation_stream = MakeExternalStream(0, Device{0}, nullptr, error_sink);
  const auto side_stream = MakeExternalStream(1, Device{1}, nullptr, error_sink);
  const auto allocation_state = StreamAccess::GetState(allocation_stream);
  const auto side_state = StreamAccess::GetState(side_stream);
  StreamUsage usage{allocation_state};

  usage.Record(side_state);
  usage.Record(allocation_state);
  usage.Record(side_state);

  EXPECT_EQ(usage.GetAllocationStream(), allocation_state);
  const auto side_streams = std::move(usage).TakeSideStreams();
  ASSERT_EQ(side_streams.size(), 1);
  EXPECT_EQ(side_streams.front(), side_state);
}

TEST(StreamUsageTest, KeepsExternalStreamOwnersAliveThroughTheTransferredLeases) {
  const auto error_sink = std::make_shared<IgnoreErrorSink>();
  auto allocation_owner = std::make_shared<int>(1);
  auto side_owner = std::make_shared<int>(2);
  const std::weak_ptr<int> weak_allocation_owner = allocation_owner;
  const std::weak_ptr<int> weak_side_owner = side_owner;
  std::shared_ptr<StreamState> allocation_lease;
  std::vector<std::shared_ptr<StreamState>> side_leases;

  {
    const auto allocation_stream = MakeExternalStream(0, Device{0}, allocation_owner, error_sink);
    const auto side_stream = MakeExternalStream(1, Device{0}, side_owner, error_sink);
    StreamUsage usage{StreamAccess::GetState(allocation_stream)};
    usage.Record(StreamAccess::GetState(side_stream));

    allocation_owner.reset();
    side_owner.reset();
    EXPECT_FALSE(weak_allocation_owner.expired());
    EXPECT_FALSE(weak_side_owner.expired());

    allocation_lease = usage.GetAllocationStream();
    side_leases = std::move(usage).TakeSideStreams();
  }

  EXPECT_FALSE(weak_allocation_owner.expired());
  EXPECT_FALSE(weak_side_owner.expired());
  allocation_lease.reset();
  side_leases.clear();
  EXPECT_TRUE(weak_allocation_owner.expired());
  EXPECT_TRUE(weak_side_owner.expired());
}

TEST(StreamUsageTest, RejectsNullStreamStatesAtTheCallSite) {
  const auto error_sink = std::make_shared<IgnoreErrorSink>();
  const auto allocation_stream = MakeExternalStream(0, Device{0}, nullptr, error_sink);
  const auto location = std::source_location::current();

  try {
    StreamUsage usage{nullptr, location};
    FAIL() << "StreamUsage did not reject a null allocation stream";
  } catch (const InvalidArgumentError &error) {
    EXPECT_EQ(error.GetLocation().line(), location.line());
  }

  StreamUsage usage{StreamAccess::GetState(allocation_stream)};
  try {
    usage.Record(nullptr, location);
    FAIL() << "Record did not reject a null stream";
  } catch (const InvalidArgumentError &error) {
    EXPECT_EQ(error.GetLocation().line(), location.line());
  }
}

TEST(StreamUsageTest, SupportsConcurrentRecordingAndDeduplicatesAllStreams) {
  constexpr size_t stream_count = 8;
  constexpr size_t thread_count = 8;
  constexpr size_t iteration_count = 1000;
  const auto error_sink = std::make_shared<IgnoreErrorSink>();
  std::vector<Stream> stream_handles;
  std::vector<std::shared_ptr<StreamState>> stream_states;
  stream_handles.reserve(stream_count);
  stream_states.reserve(stream_count);
  for (size_t index = 0; index < stream_count; index++) {
    stream_handles.push_back(MakeExternalStream(index, Device{static_cast<int32_t>(index % 2)}, nullptr, error_sink));
    stream_states.push_back(StreamAccess::GetState(stream_handles.back()));
  }

  StreamUsage usage{stream_states.front()};
  std::barrier start_barrier{static_cast<ptrdiff_t>(thread_count)};
  std::vector<std::jthread> threads;
  threads.reserve(thread_count);
  for (size_t thread_index = 0; thread_index < thread_count; thread_index++) {
    threads.emplace_back([&, thread_index] {
      start_barrier.arrive_and_wait();
      for (size_t iteration = 0; iteration < iteration_count; iteration++) {
        usage.Record(stream_states[(thread_index + iteration) % stream_count]);
      }
    });
  }
  threads.clear();

  EXPECT_EQ(usage.GetAllocationStream(), stream_states.front());
  const auto side_streams = std::move(usage).TakeSideStreams();
  ASSERT_EQ(side_streams.size(), stream_count - 1);
  EXPECT_EQ(std::ranges::count(side_streams, stream_states.front()), 0);
  for (auto iterator = std::next(stream_states.begin()); iterator != stream_states.end(); ++iterator) {
    EXPECT_EQ(std::ranges::count(side_streams, *iterator), 1);
  }
}

static_assert(noexcept(std::declval<const StreamUsage &>().GetAllocationStream()));
static_assert(noexcept(std::declval<StreamUsage &&>().TakeSideStreams()));

}  // namespace
}  // namespace ttl::internal
