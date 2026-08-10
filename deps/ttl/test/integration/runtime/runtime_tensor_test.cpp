#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "support/runtime_session.hpp"
#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/runtime/event.hpp"
#include "ttl/runtime/generator.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/layout.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::test {
namespace {

class RuntimeTensorTest : public SingleDeviceTest {};
class RuntimeConstructionTest : public CudaDeviceTest {};

std::atomic<uint64_t> retirement_not_ready_clear_count{0};

auto ReportRetirementNotReady(cudaEvent_t event) -> cudaError_t {
  static_cast<void>(event);
  return cudaErrorNotReady;
}

auto ClearRetirementNotReady() -> cudaError_t {
  retirement_not_ready_clear_count.fetch_add(1, std::memory_order_relaxed);
  return cudaErrorNotReady;
}

}  // namespace

TEST(RuntimeTest, ValidatesOptionsBeforeCreatingNativeResources) {
  EXPECT_THROW(static_cast<void>(Runtime(RuntimeOptions{})), InvalidArgumentError);
  auto sink = std::make_shared<RecordingErrorSink>();
  EXPECT_THROW(static_cast<void>(Runtime(MakeRuntimeOptions({}, sink))), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Runtime(MakeRuntimeOptions({Device{0}, Device{0}}, sink))), InvalidArgumentError);
}

TEST_F(RuntimeTensorTest, OwnsDeviceContextAndRequiresExplicitCleanShutdown) {
  const auto &properties = GetRuntime().GetDeviceProperties(GetDevice());
  EXPECT_EQ(properties.device_, GetDevice());
  EXPECT_GE(properties.compute_capability_.GetSmVersion(), 80);
  EXPECT_EQ(GetRuntime().GetStatus(), RuntimeStatus::RUNNING);
  EXPECT_TRUE(GetRuntime().CanAccessPeer(GetDevice(), GetDevice()));
  ShutdownRuntime();
  EXPECT_EQ(GetRuntime().GetStatus(), RuntimeStatus::CLOSED);
}

TEST_F(RuntimeTensorTest, ExposesAllocatorAndLifecycleStatistics) {
  const auto before = GetRuntime().GetStatistics();
  ASSERT_EQ(before.devices_.size(), 1);
  EXPECT_EQ(before.status_, RuntimeStatus::RUNNING);
  EXPECT_EQ(before.execution_context_count_, 1);
  EXPECT_GE(before.devices_[0].outstanding_storage_count_, 1);

  auto tensor = Empty(GetContext(), Shape{1024}, DType::FLOAT32);
  const auto during = GetRuntime().GetStatistics();
  EXPECT_GE(during.devices_[0].logical_live_bytes_, before.devices_[0].logical_live_bytes_ + 4096);
  EXPECT_GE(during.devices_[0].allocation_count_, before.devices_[0].allocation_count_ + 1);
  EXPECT_GT(during.devices_[0].blas_workspace_bytes_, 0);
  static_cast<void>(tensor);
}

TEST_F(RuntimeConstructionTest, ReclaimsCompletedDeviceRetirementOnDemand) {
  auto sink = std::make_shared<RecordingErrorSink>();
  auto options = MakeRuntimeOptions({GetDevice()}, sink);
  options.device_memory_.max_live_bytes_ = 16384;
  Runtime runtime{options};
  {
    auto context = runtime.CreateExecutionContext(GetDevice());
    const auto baseline = runtime.GetStatistics().devices_[0].logical_live_bytes_;
    {
      auto tensor = Empty(context, Shape{2048}, DType::FLOAT32);
      EXPECT_EQ(runtime.GetStatistics().devices_[0].logical_live_bytes_, baseline + 8192);
    }

    const auto retiring = runtime.GetStatistics();
    EXPECT_EQ(retiring.devices_[0].logical_live_bytes_, baseline);
    EXPECT_EQ(retiring.devices_[0].pending_retirement_count_, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto replacement = Empty(context, Shape{2048}, DType::FLOAT32);
    EXPECT_EQ(runtime.GetStatistics().devices_[0].logical_live_bytes_, baseline + 8192);
    static_cast<void>(replacement);
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST_F(RuntimeConstructionTest, DeviceBudgetOomIsCountedAndRuntimeRecoversAfterRetirement) {
  auto sink = std::make_shared<RecordingErrorSink>();
  auto options = MakeRuntimeOptions({GetDevice()}, sink);
  options.device_memory_.max_live_bytes_ = 16384;
  Runtime runtime{options};
  {
    auto context = runtime.CreateExecutionContext(GetDevice());
    const uint64_t baseline_live_bytes = runtime.GetStatistics().devices_.front().logical_live_bytes_;
    {
      Tensor live = Empty(context, Shape{3072}, DType::FLOAT32);
      EXPECT_THROW(static_cast<void>(Empty(context, Shape{2048}, DType::FLOAT32)), OutOfMemoryError);
      const auto after_oom = runtime.GetStatistics().devices_.front();
      EXPECT_EQ(after_oom.logical_live_bytes_, baseline_live_bytes + 12288U);
      EXPECT_GE(after_oom.oom_count_, 1U);
      FillOut(context, live, Scalar{7.0F});
      context.Synchronize();
    }

    context.Synchronize();
    runtime.Poll();
    Tensor recovered = Empty(context, Shape{2048}, DType::FLOAT32);
    FillOut(context, recovered, Scalar{3.0F});
    ExpectFloatValues(context, recovered, std::vector<float>(2048, 3.0F));
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST_F(RuntimeConstructionTest, ClearsIncompleteDeviceRetirementBeforeCallerThreadAllocation) {
  auto sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions({GetDevice()}, sink)};
  {
    auto context = runtime.CreateExecutionContext(GetDevice());
    {
      auto tensor = Empty(context, Shape{1024}, DType::FLOAT32);
      static_cast<void>(tensor);
    }
    ASSERT_EQ(runtime.GetStatistics().devices_[0].pending_retirement_count_, 1);

    auto cuda_api = internal::GetCudaApi();
    cuda_api.query_event_ = ReportRetirementNotReady;
    cuda_api.get_last_error_ = ClearRetirementNotReady;
    retirement_not_ready_clear_count.store(0, std::memory_order_relaxed);
    {
      const internal::ScopedCudaApiOverride override{cuda_api};
      auto replacement = Empty(context, Shape{1024}, DType::FLOAT32);
      static_cast<void>(replacement);
    }
    EXPECT_EQ(retirement_not_ready_clear_count.load(std::memory_order_relaxed), 1);
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST_F(RuntimeConstructionTest, SerializesConcurrentContextCreationAndShutdown) {
  auto sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions({GetDevice()}, sink)};
  std::optional<ExecutionContext> context;
  std::exception_ptr creation_error;
  std::exception_ptr shutdown_error;

  std::thread creator{[&] {
    try {
      context.emplace(runtime.CreateExecutionContext(GetDevice()));
    } catch (...) {
      creation_error = std::current_exception();
    }
  }};
  std::thread shutdown{[&] {
    try {
      runtime.Shutdown();
    } catch (...) {
      shutdown_error = std::current_exception();
    }
  }};
  creator.join();
  shutdown.join();

  EXPECT_TRUE(context.has_value() || creation_error != nullptr);
  if (runtime.GetStatus() == RuntimeStatus::CLOSED) {
    EXPECT_FALSE(context.has_value());
  } else {
    EXPECT_NE(shutdown_error, nullptr);
    context.reset();
    runtime.Shutdown();
  }
}

TEST_F(RuntimeTensorTest, AllocatesCopiesAndClassifiesViews) {
  auto &context = GetContext();
  const std::vector<int32_t> host{0, 1, 2, 3, 4, 5};
  auto tensor = Upload(context, Shape{2, 3}, host);
  tensor.RecordUsage(context.GetStream());
  EXPECT_EQ(Download<int32_t>(context, tensor), host);
  EXPECT_TRUE(tensor.IsContiguous());
  EXPECT_TRUE(tensor.IsNonOverlappingDense());

  auto transposed = Transpose(tensor, 0, 1);
  EXPECT_FALSE(transposed.IsContiguous());
  EXPECT_TRUE(transposed.IsNonOverlappingDense());
  EXPECT_EQ(ClassifyAlias(tensor, transposed), AliasKind::MAY_OVERLAP);

  auto first_row = Narrow(tensor, 0, 0, 1);
  auto second_row = Narrow(tensor, 0, 1, 1);
  EXPECT_EQ(ClassifyAlias(first_row, second_row), AliasKind::DISJOINT);
  EXPECT_EQ(ClassifyAlias(tensor, tensor), AliasKind::EXACT);
}

TEST_F(RuntimeTensorTest, HandlesScalarAndEmptyStorage) {
  auto &context = GetContext();
  auto scalar = Full(context, Shape{}, Scalar{int64_t{7}}, DType::INT64);
  EXPECT_EQ(Download<int64_t>(context, scalar), (std::vector<int64_t>{7}));

  auto empty = Empty(context, Shape{2, 0, 4}, DType::FLOAT32);
  EXPECT_EQ(empty.GetNumElements(), 0);
  EXPECT_EQ(empty.GetData<float>(), nullptr);
  EXPECT_TRUE(Download<float>(context, empty).empty());
}

TEST_F(RuntimeConstructionTest, MovedFromHandlesFailDeterministically) {
  auto sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions({GetDevice()}, sink)};
  {
    auto context = runtime.CreateExecutionContext(GetDevice());

    auto tensor = Empty(context, Shape{1}, DType::FLOAT32);
    auto moved_tensor = std::move(tensor);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(static_cast<void>(tensor.GetDType()), InvalidArgumentError);

    auto stream = context.GetStream();
    auto moved_stream = std::move(stream);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(static_cast<void>(stream.GetDevice()), InvalidArgumentError);

    auto event = context.RecordEvent();
    auto moved_event = std::move(event);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(static_cast<void>(event.Query()), InvalidArgumentError);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(context.Wait(event), InvalidArgumentError);

    Generator generator{context, 7};
    auto moved_generator = std::move(generator);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(static_cast<void>(generator.GetSeed()), InvalidArgumentError);

    auto moved_context = std::move(context);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(static_cast<void>(context.GetDevice()), InvalidArgumentError);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(context.Synchronize(), InvalidArgumentError);
    moved_context.Synchronize();

    EXPECT_EQ(moved_tensor.GetDType(), DType::FLOAT32);
    EXPECT_EQ(moved_stream.GetDevice(), GetDevice());
    EXPECT_EQ(moved_event.GetDevice(), GetDevice());
    EXPECT_EQ(moved_generator.GetSeed(), 7);
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

}  // namespace ttl::test
