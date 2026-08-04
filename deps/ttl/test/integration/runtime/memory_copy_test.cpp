#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "support/tensor_test_utils.hpp"
#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
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
                                     ExternalMemory{
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
