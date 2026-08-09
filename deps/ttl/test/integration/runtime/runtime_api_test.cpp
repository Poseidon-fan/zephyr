#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::test {
namespace {

[[nodiscard]] auto MakeRuntimeOptions(std::vector<Device> devices, std::shared_ptr<ErrorSink> sink) -> RuntimeOptions {
  RuntimeOptions options;
  options.devices_ = std::move(devices);
  options.error_sink_ = std::move(sink);
  return options;
}

TEST(RuntimeValidationTest, RejectsMissingSinkDevicesDuplicatesAndInvalidPoolConfiguration) {
  const auto devices = GetTestDevices(1);
  if (devices.empty()) {
    GTEST_SKIP() << "requires one CUDA device";
  }
  auto sink = std::make_shared<RecordingErrorSink>();

  RuntimeOptions missing_sink;
  missing_sink.devices_ = devices;
  EXPECT_THROW(static_cast<void>(Runtime{std::move(missing_sink)}), InvalidArgumentError);

  RuntimeOptions missing_devices;
  missing_devices.error_sink_ = sink;
  EXPECT_THROW(static_cast<void>(Runtime{std::move(missing_devices)}), InvalidArgumentError);

  auto duplicate_devices = devices;
  duplicate_devices.push_back(devices.front());
  EXPECT_THROW(static_cast<void>(Runtime{MakeRuntimeOptions(std::move(duplicate_devices), sink)}),
               InvalidArgumentError);

  auto bad_event_pool = MakeRuntimeOptions(devices, sink);
  bad_event_pool.event_pool_capacity_per_device_ = 1;
  bad_event_pool.event_pool_reserve_per_device_ = 2;
  EXPECT_THROW(static_cast<void>(Runtime{std::move(bad_event_pool)}), InvalidArgumentError);

  auto bad_blas_workspace = MakeRuntimeOptions(devices, sink);
  bad_blas_workspace.blas_workspace_bytes_ = 1024;
  EXPECT_THROW(static_cast<void>(Runtime{std::move(bad_blas_workspace)}), InvalidArgumentError);
}

TEST(RuntimeLifecycleTest, ShutdownClosesAdmissionAndCanBeRetriedAfterChildrenAreReleased) {
  auto devices = GetTestDevices(1);
  if (devices.empty()) {
    GTEST_SKIP() << "requires one CUDA device";
  }
  auto sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(devices, sink)};
  std::optional<ExecutionContext> context{runtime.CreateExecutionContext(devices.front())};

  EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::RUNNING);
  EXPECT_THROW(runtime.Shutdown(), InvalidArgumentError);
  EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::CLOSING);
  EXPECT_THROW(static_cast<void>(runtime.CreateExecutionContext(devices.front())), InvalidArgumentError);

  context->Synchronize();
  context.reset();
  EXPECT_NO_THROW(runtime.Shutdown());
  EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::CLOSED);
  EXPECT_NO_THROW(runtime.Shutdown());
}

TEST(RuntimeLifecycleTest, DestructorReportsAbandonedRuntimeWithoutThrowing) {
  auto devices = GetTestDevices(1);
  if (devices.empty()) {
    GTEST_SKIP() << "requires one CUDA device";
  }
  auto sink = std::make_shared<RecordingErrorSink>();
  {
    Runtime runtime{MakeRuntimeOptions(devices, sink)};
    EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::RUNNING);
  }
  const auto records = sink->GetRecords();
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records.front().code_, ErrorCode::INTERNAL);
  EXPECT_NE(records.front().message_.find("without a successful Shutdown"), std::string::npos);
}

TEST_F(SingleDeviceTest, ReportsDevicePropertiesContextIdentityAndStatistics) {
  const auto devices = GetRuntime().GetDevices();
  ASSERT_EQ(devices.size(), 1U);
  EXPECT_EQ(devices.front(), GetDevice());
  EXPECT_TRUE(GetRuntime().CanAccessPeer(GetDevice(), GetDevice()));
  EXPECT_THROW(static_cast<void>(GetRuntime().GetDeviceProperties(Device{9999})), InvalidArgumentError);

  const DeviceProperties &properties = GetRuntime().GetDeviceProperties(GetDevice());
  EXPECT_EQ(properties.device_, GetDevice());
  EXPECT_FALSE(properties.name_.empty());
  EXPECT_GE(properties.compute_capability_.GetSmVersion(), 80);
  EXPECT_GT(properties.multiprocessor_count_, 0);

  EXPECT_EQ(GetContext().GetDevice(), GetDevice());
  EXPECT_FALSE(GetContext().IsExternalStream());
  EXPECT_EQ(GetContext().GetAuxiliaryStreamCount(), 0U);
  EXPECT_EQ(GetContext().GetStream().GetDevice(), GetDevice());
  EXPECT_FALSE(GetContext().GetStream().IsExternal());
  EXPECT_NE(GetContext().GetStream().GetId(), 0U);

  const RuntimeStatistics statistics = GetRuntime().GetStatistics();
  EXPECT_EQ(statistics.status_, RuntimeStatus::RUNNING);
  EXPECT_EQ(statistics.execution_context_count_, 1U);
  ASSERT_EQ(statistics.devices_.size(), 1U);
  EXPECT_EQ(statistics.devices_.front().device_, GetDevice());
  EXPECT_GT(statistics.devices_.front().blas_workspace_bytes_, 0U);
}

TEST_F(SingleDeviceTest, ContextMoveLeavesSourceDetectablyInvalidAndPreservesStreamIdentity) {
  ExecutionContext source = GetRuntime().CreateExecutionContext(GetDevice(), {.max_auxiliary_stream_count_ = 2});
  const uint64_t stream_id = source.GetStream().GetId();
  EXPECT_EQ(source.GetAuxiliaryStreamCount(), 2U);

  ExecutionContext destination = std::move(source);
  EXPECT_EQ(destination.GetStream().GetId(), stream_id);
  EXPECT_EQ(destination.GetAuxiliaryStreamCount(), 2U);
  // TTL specifies that moved-from handles are detectably invalid.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(static_cast<void>(source.GetDevice()), InvalidArgumentError);
  destination.Synchronize();
}

TEST_F(SingleDeviceTest, EventsAreCopyableAndOrderWorkAcrossContexts) {
  Tensor source = TensorFromValues<int32_t>(GetContext(), Shape{4}, {1, 2, 3, 4});
  Event ready = GetContext().RecordEvent();
  Event copy = ready;  // NOLINT(performance-unnecessary-copy-initialization): exercise shared event ownership.

  ExecutionContext consumer = GetRuntime().CreateExecutionContext(GetDevice());
  consumer.Wait(copy);
  Tensor output = Empty(consumer, Shape{4}, DType::INT32);
  CopyOut(consumer, output, source);
  consumer.Synchronize();

  EXPECT_TRUE(ready.Query());
  EXPECT_TRUE(copy.Query());
  EXPECT_EQ(ready.GetDevice(), GetDevice());
  ExpectValues<int32_t>(consumer, output, {1, 2, 3, 4});
}

TEST_F(SingleDeviceTest, PinnedBuffersSupportZeroSizeCopiesMovesAndStatistics) {
  const auto baseline = GetRuntime().GetStatistics().pinned_memory_;
  {
    PinnedBuffer empty = GetRuntime().AllocatePinned(0);
    EXPECT_EQ(empty.GetData(), nullptr);
    EXPECT_TRUE(empty.AsBytes().empty());
    EXPECT_EQ(empty.GetSizeBytes(), 0U);
  }

  {
    PinnedBuffer buffer = GetRuntime().AllocatePinned(4 * sizeof(int32_t));
    const std::array<int32_t, 4> input{8, 6, 7, 5};
    std::memcpy(buffer.GetData(), input.data(), sizeof(input));
    PinnedBuffer shared = buffer;
    PinnedBuffer moved = std::move(buffer);
    // TTL specifies an empty moved-from pinned buffer.
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_EQ(buffer.GetData(), nullptr);
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_EQ(buffer.GetSizeBytes(), 0U);
    EXPECT_EQ(shared.GetData(), moved.GetData());

    Tensor tensor = Empty(GetContext(), Shape{4}, DType::INT32);
    CopyFromPinnedAsync(GetContext(), tensor, moved);
    GetContext().Synchronize();
    ExpectValues<int32_t>(GetContext(), tensor, {8, 6, 7, 5});

    const auto live = GetRuntime().GetStatistics().pinned_memory_;
    EXPECT_EQ(live.logical_live_bytes_, baseline.logical_live_bytes_ + sizeof(input));
    EXPECT_EQ(live.outstanding_buffer_count_, baseline.outstanding_buffer_count_ + 1U);
  }

  GetRuntime().Poll();
  GetRuntime().TrimPinnedMemory();
  const auto final = GetRuntime().GetStatistics().pinned_memory_;
  EXPECT_EQ(final.logical_live_bytes_, baseline.logical_live_bytes_);
  EXPECT_EQ(final.outstanding_buffer_count_, baseline.outstanding_buffer_count_);
}

TEST_F(SingleDeviceTest, WrapsBorrowedExternalStreamWithoutTakingNativeOwnership) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaSetDevice(GetDevice().GetOrdinal()), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  {
    ExecutionContext external = GetRuntime().WrapExternalStream(GetDevice(), stream);
    EXPECT_TRUE(external.IsExternalStream());
    EXPECT_TRUE(external.GetStream().IsExternal());
    EXPECT_EQ(external.GetDevice(), GetDevice());
    external.Synchronize();
  }
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(SingleDeviceTest, WrapsOwnedAndBorrowedExternalMemoryWithValidatedMetadata) {
  ASSERT_EQ(cudaSetDevice(GetDevice().GetOrdinal()), cudaSuccess);
  void *pointer = nullptr;
  ASSERT_EQ(cudaMalloc(&pointer, 4 * sizeof(int32_t)), cudaSuccess);
  auto owner = std::shared_ptr<void>{pointer, [](void *allocation) { static_cast<void>(cudaFree(allocation)); }};

  Tensor tensor = GetRuntime().FromBlob(
      GetContext(),
      ExternalDeviceMemory{
          .pointer_ = pointer, .capacity_bytes_ = 4 * sizeof(int32_t), .device_ = GetDevice(), .owner_ = owner},
      Shape{4}, Strides{1}, DType::INT32);
  owner.reset();
  const std::array<int32_t, 4> values{4, 3, 2, 1};
  CopyFromHostBlocking(GetContext(), tensor, AsBytes<int32_t>(values));
  ExpectValues<int32_t>(GetContext(), tensor, {4, 3, 2, 1});

  EXPECT_THROW(
      static_cast<void>(GetRuntime().FromBlob(
          GetContext(),
          ExternalDeviceMemory{.pointer_ = pointer, .capacity_bytes_ = 1, .device_ = GetDevice(), .owner_ = nullptr},
          Shape{4}, Strides{1}, DType::INT32)),
      InvalidArgumentError);

  Tensor empty = GetRuntime().FromBlob(
      GetContext(),
      ExternalDeviceMemory{.pointer_ = nullptr, .capacity_bytes_ = 0, .device_ = GetDevice(), .owner_ = nullptr},
      Shape{0}, Strides{1}, DType::FLOAT32);
  EXPECT_TRUE(empty.GetShape().IsEmpty());
  EXPECT_EQ(empty.GetData<float>(), nullptr);
}

TEST_F(SingleDeviceTest, DeviceAllocatorStatisticsTrackTensorLifetimeAndTrim) {
  const auto baseline = GetRuntime().GetStatistics().devices_.front();
  {
    Tensor tensor = Empty(GetContext(), Shape{1024}, DType::FLOAT32);
    const auto live = GetRuntime().GetStatistics().devices_.front();
    EXPECT_EQ(live.logical_live_bytes_, baseline.logical_live_bytes_ + 4096U);
    EXPECT_EQ(live.outstanding_storage_count_, baseline.outstanding_storage_count_ + 1U);
    FillOut(GetContext(), tensor, Scalar{1.0});
    GetContext().Synchronize();
  }
  GetRuntime().Poll();
  GetRuntime().TrimMemory(GetDevice(), 0);
  const auto final = GetRuntime().GetStatistics().devices_.front();
  EXPECT_EQ(final.logical_live_bytes_, baseline.logical_live_bytes_);
  EXPECT_EQ(final.outstanding_storage_count_, baseline.outstanding_storage_count_);
  EXPECT_GE(final.trim_count_, baseline.trim_count_ + 1U);
}

}  // namespace
}  // namespace ttl::test
