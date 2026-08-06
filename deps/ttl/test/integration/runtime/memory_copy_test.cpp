#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <source_location>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "support/tensor_test_utils.hpp"
#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/memory/pinned/cache.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/tensor/layout.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

[[nodiscard]] auto HasDevices(int count) -> bool {
  int device_count = 0;
  const auto status = cudaGetDeviceCount(&device_count);
  EXPECT_EQ(status, cudaSuccess);
  return status == cudaSuccess && device_count >= count;
}

[[nodiscard]] auto HasBidirectionalPeerAccess(Device first, Device second) -> bool {
  int first_to_second = 0;
  int second_to_first = 0;
  EXPECT_EQ(cudaDeviceCanAccessPeer(&first_to_second, first.GetOrdinal(), second.GetOrdinal()), cudaSuccess);
  EXPECT_EQ(cudaDeviceCanAccessPeer(&second_to_first, second.GetOrdinal(), first.GetOrdinal()), cudaSuccess);
  return first_to_second != 0 && second_to_first != 0;
}

auto ReportPinnedAllocationOom(void **pointer, size_t bytes, unsigned int flags) -> cudaError_t {
  static_cast<void>(bytes);
  static_cast<void>(flags);
  *pointer = nullptr;
  return cudaErrorMemoryAllocation;
}

auto FailPinnedFree(void *pointer) -> cudaError_t {
  static_cast<void>(pointer);
  return cudaErrorUnknown;
}

auto FailEventQuery(cudaEvent_t event) -> cudaError_t {
  static_cast<void>(event);
  return cudaErrorUnknown;
}

}  // namespace

TEST(RuntimeMemoryIntegrationTest, CopiesThroughPinnedMemoryAndTreatsMovedBufferAsEmpty) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    auto source = runtime.AllocatePinned(4 * sizeof(int32_t));
    const std::array<int32_t, 4> expected{3, -5, 8, 13};
    std::memcpy(source.GetData(), expected.data(), source.GetSizeBytes());

    auto tensor = Empty(context, Shape{4}, DType::INT32);
    CopyFromPinnedAsync(context, tensor, source);
    auto destination = runtime.AllocatePinned(source.GetSizeBytes());
    CopyToPinnedAsync(context, destination, tensor);
    context.Synchronize();

    const auto actual = std::span{static_cast<const int32_t *>(destination.GetData()), expected.size()};
    EXPECT_TRUE(std::ranges::equal(actual, expected));

    auto moved = std::move(destination);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_EQ(destination.GetData(), nullptr);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_EQ(destination.GetSizeBytes(), 0);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_TRUE(destination.AsBytes().empty());
    EXPECT_EQ(moved.GetSizeBytes(), source.GetSizeBytes());
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(RuntimeMemoryIntegrationTest, ReusesPinnedSizeClassesAndTracksContextMirror) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    const auto baseline = runtime.GetStatistics().pinned_memory_;
    EXPECT_GE(baseline.outstanding_buffer_count_, 1);

    {
      auto buffer = runtime.AllocatePinned(4097);
      const auto live = runtime.GetStatistics().pinned_memory_;
      EXPECT_GE(live.logical_live_bytes_, baseline.logical_live_bytes_ + 4097);
      EXPECT_GE(live.live_capacity_bytes_, baseline.live_capacity_bytes_ + 8192);
      EXPECT_GE(live.budgeted_capacity_bytes_, baseline.budgeted_capacity_bytes_ + 8192);
      EXPECT_EQ(buffer.GetSizeBytes(), 4097);
    }
    const auto cached = runtime.GetStatistics().pinned_memory_;
    EXPECT_GE(cached.cached_capacity_bytes_, baseline.cached_capacity_bytes_ + 8192);

    const auto hits_before = cached.cache_hit_count_;
    {
      auto buffer = runtime.AllocatePinned(5000);
      EXPECT_EQ(buffer.GetSizeBytes(), 5000);
      EXPECT_GE(runtime.GetStatistics().pinned_memory_.cache_hit_count_, hits_before + 1);
    }

    const auto before_usage = runtime.GetStatistics();
    {
      auto buffer = runtime.AllocatePinned(1);
      buffer.RecordUsage(context.GetStream());
      buffer.RecordUsage(context.GetStream());
    }
    const auto retiring = runtime.GetStatistics();
    EXPECT_EQ(retiring.pinned_memory_.pending_retirement_count_,
              before_usage.pinned_memory_.pending_retirement_count_ + 1);
    EXPECT_EQ(retiring.devices_[0].outstanding_event_count_, before_usage.devices_[0].outstanding_event_count_ + 1);
    context.Synchronize();
    const auto completed = runtime.GetStatistics();
    EXPECT_EQ(completed.pinned_memory_.pending_retirement_count_,
              before_usage.pinned_memory_.pending_retirement_count_);
    EXPECT_EQ(completed.devices_[0].outstanding_event_count_, before_usage.devices_[0].outstanding_event_count_);
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(RuntimeMemoryIntegrationTest, TreatsZeroBytePinnedBufferAsOwnedEmptyStorage) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  {
    auto buffer = runtime.AllocatePinned(0);
    EXPECT_EQ(buffer.GetData(), nullptr);
    EXPECT_EQ(buffer.GetSizeBytes(), 0);
    EXPECT_EQ(runtime.GetStatistics().pinned_memory_.outstanding_buffer_count_, 1);
    EXPECT_THROW(runtime.Shutdown(), InvalidArgumentError);
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(RuntimeMemoryIntegrationTest, PreservesAllocationAndRecoveryFailuresWhenPinnedTrimFails) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  internal::PinnedMemoryCache cache{sink, 4096, std::source_location::current()};
  const auto cached_capacity = internal::PinnedMemoryCache::GetSizeClass(1, std::source_location::current());
  const auto cached_allocation = cache.Acquire(1, cached_capacity, std::source_location::current());
  ASSERT_TRUE(cache.Release(cached_allocation, std::source_location::current()));

  auto cuda_api = internal::GetCudaApi();
  cuda_api.host_alloc_ = ReportPinnedAllocationOom;
  cuda_api.free_host_ = FailPinnedFree;
  {
    const internal::ScopedCudaApiOverride override{cuda_api};
    try {
      const auto requested_capacity = internal::PinnedMemoryCache::GetSizeClass(4097, std::source_location::current());
      static_cast<void>(cache.Acquire(4097, requested_capacity, std::source_location::current()));
      FAIL() << "expected pinned allocation recovery to fail";
    } catch (const CudaError &error) {
      const auto message = error.GetMessage();
      EXPECT_NE(message.find("cudaHostAlloc reported cudaErrorMemoryAllocation"), std::string_view::npos);
      EXPECT_NE(message.find("cudaFreeHost failed"), std::string_view::npos);
    }
  }

  const auto restored = cache.GetStats();
  EXPECT_EQ(restored.cached_bytes_, cached_capacity);
  EXPECT_EQ(restored.physical_bytes_, cached_capacity);
  cache.Trim(std::source_location::current());
  EXPECT_EQ(cache.GetStats().physical_bytes_, 0);
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(RuntimeMemoryIntegrationTest, QuarantinesPinnedRetirementWhenEventQueryFails) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  auto options = test::MakeRuntimeOptions({Device{0}}, sink);
  options.device_memory_.enable_maintenance_thread_ = false;
  Runtime runtime{options};
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    const auto baseline = runtime.GetStatistics().pinned_memory_;
    {
      auto buffer = runtime.AllocatePinned(1);
      buffer.RecordUsage(context.GetStream());
    }

    auto cuda_api = internal::GetCudaApi();
    cuda_api.query_event_ = FailEventQuery;
    {
      const internal::ScopedCudaApiOverride override{cuda_api};
      runtime.Poll();
    }
    const auto failed = runtime.GetStatistics().pinned_memory_;
    EXPECT_EQ(failed.quarantined_retirement_count_, baseline.quarantined_retirement_count_ + 1);
    EXPECT_GE(failed.quarantined_bytes_, baseline.quarantined_bytes_ + 4096);
  }
  runtime.Shutdown();
  EXPECT_FALSE(sink->GetRecords().empty());
}

TEST(RuntimeMemoryIntegrationTest, QuarantinesPinnedAllocationWhenNativeFreeFails) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  auto options = test::MakeRuntimeOptions({Device{0}}, sink);
  options.pinned_memory_.max_cached_bytes_ = 0;
  Runtime runtime{options};
  const auto baseline = runtime.GetStatistics().pinned_memory_;

  auto cuda_api = internal::GetCudaApi();
  cuda_api.free_host_ = FailPinnedFree;
  {
    const internal::ScopedCudaApiOverride override{cuda_api};
    auto buffer = runtime.AllocatePinned(1);
  }

  const auto failed = runtime.GetStatistics().pinned_memory_;
  EXPECT_EQ(failed.pending_retirement_count_, baseline.pending_retirement_count_ + 1);
  EXPECT_EQ(failed.quarantined_retirement_count_, baseline.quarantined_retirement_count_ + 1);
  EXPECT_GE(failed.quarantined_bytes_, baseline.quarantined_bytes_ + 4096);
  runtime.Shutdown();
  EXPECT_FALSE(sink->GetRecords().empty());
}

TEST(RuntimeMemoryIntegrationTest, WrapsBorrowedDeviceMemoryAndExternalStream) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  cudaStream_t native_stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&native_stream, cudaStreamNonBlocking), cudaSuccess);
  void *device_pointer = nullptr;
  ASSERT_EQ(cudaMalloc(&device_pointer, 4 * sizeof(float)), cudaSuccess);

  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  {
    auto context = runtime.WrapExternalStream(Device{0}, native_stream);
    EXPECT_TRUE(context.IsExternalStream());
    {
      auto tensor = runtime.FromBlob(context,
                                     ExternalDeviceMemory{
                                         .pointer_ = device_pointer,
                                         .capacity_bytes_ = 4 * sizeof(float),
                                         .device_ = Device{0},
                                         .owner_ = nullptr,
                                     },
                                     Shape{2, 2}, Strides{2, 1}, DType::FLOAT32);
      const std::vector<float> expected{1.0F, 2.0F, 4.0F, 8.0F};
      CopyFromHostBlocking(context, tensor, std::as_bytes(std::span{expected}));
      EXPECT_EQ(test::Download<float>(context, tensor), expected);
    }
    context.Synchronize();
  }
  runtime.Shutdown();
  EXPECT_EQ(cudaFree(device_pointer), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(native_stream), cudaSuccess);
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(RuntimeMemoryIntegrationTest, ShutdownCanResumeAfterOutstandingContextIsReleased) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    EXPECT_THROW(runtime.Shutdown(), InvalidArgumentError);
    EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::CLOSING);
    EXPECT_THROW(static_cast<void>(runtime.CreateExecutionContext(Device{0})), InvalidArgumentError);
  }
  runtime.Shutdown();
  EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::CLOSED);
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(RuntimeMemoryIntegrationTest, WaitsForEventsRecordedOnAnotherDevice) {
  if (!HasDevices(2)) {
    GTEST_SKIP() << "requires at least two CUDA devices";
  }
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}, Device{1}}, sink)};
  {
    auto producer = runtime.CreateExecutionContext(Device{0});
    auto consumer = runtime.CreateExecutionContext(Device{1});
    auto source = Full(producer, Shape{1024}, Scalar{7.0F}, DType::FLOAT32);
    const auto ready = producer.RecordEvent();
    consumer.Wait(ready);
    consumer.Synchronize();
    EXPECT_EQ(source.GetDevice(), Device{0});
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(RuntimeMemoryIntegrationTest, CopiesPeerTensorsInBothDirections) {
  if (!HasDevices(2) || !HasBidirectionalPeerAccess(Device{0}, Device{1})) {
    GTEST_SKIP() << "requires bidirectional peer access between two CUDA devices";
  }
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}, Device{1}}, sink)};
  {
    std::array contexts{
        runtime.CreateExecutionContext(Device{0}),
        runtime.CreateExecutionContext(Device{1}),
    };
    for (size_t source_index = 0; source_index < contexts.size(); ++source_index) {
      const auto destination_index = 1 - source_index;
      const std::vector<int32_t> expected{3, -5, 8, 13};
      auto source = test::Upload(contexts[source_index], Shape{4}, expected);
      auto destination = Empty(contexts[destination_index], Shape{4}, DType::INT32);
      const auto source_ready = contexts[source_index].RecordEvent();
      CopyPeerOut(contexts[destination_index], destination, source, source_ready);
      EXPECT_EQ(test::Download<int32_t>(contexts[destination_index], destination), expected);
    }
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

}  // namespace ttl
