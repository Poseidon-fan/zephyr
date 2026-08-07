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

#include "support/tensor_test_utils.hpp"
#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/runtime/event.hpp"
#include "ttl/runtime/generator.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/layout.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

TEST(RuntimeTest, ValidatesOptionsBeforeCreatingNativeResources) {
  EXPECT_THROW(static_cast<void>(Runtime(RuntimeOptions{})), InvalidArgumentError);
  auto sink = std::make_shared<test::RecordingErrorSink>();
  EXPECT_THROW(static_cast<void>(Runtime(test::MakeRuntimeOptions({}, sink))), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Runtime(test::MakeRuntimeOptions({Device{0}, Device{0}}, sink))),
               InvalidArgumentError);
}

TEST(RuntimeTest, OwnsDeviceContextAndRequiresExplicitCleanShutdown) {
  test::RuntimeSession session;
  const auto &properties = session.GetRuntime().GetDeviceProperties(Device{0});
  EXPECT_EQ(properties.device_, Device{0});
  EXPECT_GE(properties.compute_capability_.GetSmVersion(), 80);
  EXPECT_EQ(session.GetRuntime().GetStatus(), RuntimeStatus::RUNNING);
  EXPECT_TRUE(session.GetRuntime().CanAccessPeer(Device{0}, Device{0}));
  session.Close();
  EXPECT_EQ(session.GetRuntime().GetStatus(), RuntimeStatus::CLOSED);
  EXPECT_TRUE(session.GetErrorSink()->GetRecords().empty());
}

TEST(RuntimeTest, ExposesAllocatorAndLifecycleStatistics) {
  test::RuntimeSession session;
  const auto before = session.GetRuntime().GetStatistics();
  ASSERT_EQ(before.devices_.size(), 1);
  EXPECT_EQ(before.status_, RuntimeStatus::RUNNING);
  EXPECT_EQ(before.execution_context_count_, 1);
  EXPECT_GE(before.devices_[0].outstanding_storage_count_, 1);

  auto tensor = Empty(session.GetContext(), Shape{1024}, DType::FLOAT32);
  const auto during = session.GetRuntime().GetStatistics();
  EXPECT_GE(during.devices_[0].logical_live_bytes_, before.devices_[0].logical_live_bytes_ + 4096);
  EXPECT_GE(during.devices_[0].allocation_count_, before.devices_[0].allocation_count_ + 1);
  EXPECT_GT(during.devices_[0].blas_workspace_bytes_, 0);
  static_cast<void>(tensor);
}

TEST(RuntimeTest, ReclaimsCompletedDeviceRetirementOnDemand) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  auto options = test::MakeRuntimeOptions({Device{0}}, sink);
  options.device_memory_.max_live_bytes_ = 16384;
  Runtime runtime{options};
  {
    auto context = runtime.CreateExecutionContext(Device{0});
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

TEST(RuntimeTest, SerializesConcurrentContextCreationAndShutdown) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  std::optional<ExecutionContext> context;
  std::exception_ptr creation_error;
  std::exception_ptr shutdown_error;

  std::thread creator{[&] {
    try {
      context.emplace(runtime.CreateExecutionContext(Device{0}));
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

TEST(TensorTest, AllocatesCopiesAndClassifiesViews) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  const std::vector<int32_t> host{0, 1, 2, 3, 4, 5};
  auto tensor = test::Upload(context, Shape{2, 3}, host);
  tensor.RecordUsage(context.GetStream());
  EXPECT_EQ(test::Download<int32_t>(context, tensor), host);
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

TEST(TensorTest, HandlesScalarAndEmptyStorage) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto scalar = Full(context, Shape{}, Scalar{int64_t{7}}, DType::INT64);
  EXPECT_EQ(test::Download<int64_t>(context, scalar), (std::vector<int64_t>{7}));

  auto empty = Empty(context, Shape{2, 0, 4}, DType::FLOAT32);
  EXPECT_EQ(empty.GetNumElements(), 0);
  EXPECT_EQ(empty.GetData<float>(), nullptr);
  EXPECT_TRUE(test::Download<float>(context, empty).empty());
}

TEST(RuntimeHandleTest, MovedFromHandlesFailDeterministically) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  {
    auto context = runtime.CreateExecutionContext(Device{0});

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
    EXPECT_EQ(moved_stream.GetDevice(), Device{0});
    EXPECT_EQ(moved_event.GetDevice(), Device{0});
    EXPECT_EQ(moved_generator.GetSeed(), 7);
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

}  // namespace ttl
