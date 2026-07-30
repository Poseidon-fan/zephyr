#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <utility>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/internal/execution_lane.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/parallel_op_scope.hpp"
#include "ttl/internal/scratch_arena.hpp"
#include "ttl/runtime.hpp"

namespace ttl::internal {
namespace {

class CountingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord /*error*/) noexcept override { count_.fetch_add(1, std::memory_order_relaxed); }

  [[nodiscard]] auto GetCount() const noexcept -> size_t { return count_.load(std::memory_order_relaxed); }

 private:
  std::atomic<size_t> count_{0};
};

[[nodiscard]] auto MakeRuntimeOptions(const std::shared_ptr<ErrorSink> &error_sink) -> RuntimeOptions {
  auto options = RuntimeOptions{};
  options.devices_ = {Device{0}};
  options.device_memory_.enable_maintenance_thread_ = false;
  options.event_pool_capacity_per_device_ = 64;
  options.error_sink_ = error_sink;
  return options;
}

class ScratchArenaTest : public testing::Test {
 protected:
  void SetUp() override {
    error_sink_ = std::make_shared<CountingErrorSink>();
    runtime_ = std::make_unique<Runtime>(MakeRuntimeOptions(error_sink_));
    context_.emplace(runtime_->CreateExecutionContext(Device{0}));
  }

  void TearDown() override {
    context_.reset();
    runtime_->Shutdown();
    runtime_.reset();
    EXPECT_EQ(error_sink_->GetCount(), 0);
  }

  [[nodiscard]] auto GetContext() -> ExecutionContext & { return *context_; }

  std::shared_ptr<CountingErrorSink> error_sink_;
  std::unique_ptr<Runtime> runtime_;
  std::optional<ExecutionContext> context_;
};

TEST_F(ScratchArenaTest, AllocatesAlignedRangesAndReusesReservedStorage) {
  void *base = nullptr;
  {
    OpGuard guard{GetContext(), "scratch allocation"};
    guard.ReserveScratch(512);
    EXPECT_EQ(guard.GetScratchCapacityBytes(), 512);

    {
      auto scope = guard.MakeScratchScope();
      const auto first = scope.AllocateBytes(13, 1);
      const auto second = scope.AllocateBytes(64, 64);
      const auto empty = scope.AllocateBytes(0);

      base = first.GetData();
      EXPECT_NE(base, nullptr);
      EXPECT_EQ(first.GetOffsetBytes(), 0);
      EXPECT_EQ(first.GetSizeBytes(), 13);
      EXPECT_EQ(second.GetOffsetBytes(), 64);
      EXPECT_EQ(second.GetSizeBytes(), 64);
      EXPECT_EQ(reinterpret_cast<uintptr_t>(second.GetData()) % 64, 0);
      EXPECT_EQ(empty.GetData(), nullptr);
      EXPECT_EQ(empty.GetOffsetBytes(), 128);
      EXPECT_EQ(empty.GetSizeBytes(), 0);
      EXPECT_EQ(cudaMemsetAsync(first.GetData(), 0x11, first.GetSizeBytes(), guard.GetNativeStream()), cudaSuccess);
      EXPECT_EQ(cudaMemsetAsync(second.GetData(), 0x22, second.GetSizeBytes(), guard.GetNativeStream()), cudaSuccess);
      guard.CheckLaunch();
    }
    EXPECT_EQ(guard.GetScratchHighWaterBytes(), 128);
  }

  {
    OpGuard guard{GetContext(), "scratch reuse"};
    auto scope = guard.MakeScratchScope();
    const auto allocation = scope.AllocateBytes(128);
    EXPECT_EQ(allocation.GetData(), base);
    EXPECT_EQ(allocation.GetOffsetBytes(), 0);
    EXPECT_EQ(guard.GetScratchCapacityBytes(), 512);
  }
  GetContext().Synchronize();
}

TEST_F(ScratchArenaTest, PreservesEarlierAllocationsAcrossGrowth) {
  {
    OpGuard guard{GetContext(), "scratch growth"};
    auto scope = guard.MakeScratchScope();
    const auto first = scope.AllocateBytes(64);
    EXPECT_EQ(guard.GetScratchCapacityBytes(), 64);
    EXPECT_EQ(cudaMemsetAsync(first.GetData(), 0x31, first.GetSizeBytes(), guard.GetNativeStream()), cudaSuccess);

    const auto second = scope.AllocateBytes(1024);
    EXPECT_NE(first.GetData(), second.GetData());
    EXPECT_EQ(second.GetOffsetBytes(), 256);
    EXPECT_EQ(guard.GetScratchCapacityBytes(), 2048);
    EXPECT_EQ(guard.GetScratchHighWaterBytes(), 1280);

    EXPECT_EQ(cudaMemsetAsync(second.GetData(), 0x42, second.GetSizeBytes(), guard.GetNativeStream()), cudaSuccess);
    EXPECT_EQ(cudaMemsetAsync(first.GetData(), 0x53, first.GetSizeBytes(), guard.GetNativeStream()), cudaSuccess);
    guard.CheckLaunch();
  }
  EXPECT_NO_THROW(GetContext().Synchronize());
}

TEST_F(ScratchArenaTest, EnforcesScopeAlignmentOverflowAndFixedCapacityContracts) {
  OpGuard guard{GetContext(), "scratch contracts"};
  guard.ReserveScratch(256);
  auto &lane = ContextAccess::GetPrimaryLane(GetContext(), std::source_location::current());

  {
    auto scope = lane.MakeScratchScope(ScratchGrowthPolicy::FIXED_CAPACITY);
    EXPECT_THROW([[maybe_unused]] auto nested = lane.MakeScratchScope(), InvalidArgumentError);
    EXPECT_THROW(lane.ReserveScratch(512), InvalidArgumentError);
    EXPECT_THROW([[maybe_unused]] const auto allocation = scope.AllocateBytes(1, 0), InvalidArgumentError);
    EXPECT_THROW([[maybe_unused]] const auto allocation = scope.AllocateBytes(1, 512), InvalidArgumentError);

    const auto full = scope.AllocateBytes(256);
    EXPECT_NE(full.GetData(), nullptr);
    EXPECT_THROW([[maybe_unused]] const auto allocation = scope.AllocateBytes(1, 1), CaptureError);
    EXPECT_EQ(lane.GetScratchCapacityBytes(), 256);
    EXPECT_EQ(lane.GetScratchHighWaterBytes(), 256);
  }

  {
    auto scope = lane.MakeScratchScope();
    [[maybe_unused]] const auto prefix = scope.AllocateBytes(1, 1);
    EXPECT_THROW([[maybe_unused]] const auto allocation = scope.AllocateBytes(std::numeric_limits<size_t>::max(), 256),
                 OverflowError);

    auto moved = std::move(scope);
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_THROW([[maybe_unused]] const auto allocation = scope.AllocateBytes(1), InvalidArgumentError);
    EXPECT_NE(moved.AllocateBytes(1).GetData(), nullptr);
  }
}

TEST_F(ScratchArenaTest, GivesEveryExecutionLaneIndependentStorage) {
  context_.reset();
  context_.emplace(runtime_->CreateExecutionContext(
      Device{0}, ExecutionContextOptions{.stream_priority_ = 0, .max_auxiliary_stream_count_ = 2}));

  {
    OpGuard guard{GetContext(), "parallel scratch"};
    guard.ReserveScratch(256);
    auto primary_scope = guard.MakeScratchScope();
    const auto primary = primary_scope.AllocateBytes(256);

    ParallelOpScope parallel{guard, 2};
    parallel.ReserveAuxiliaryScratch(0, 256);
    parallel.ReserveAuxiliaryScratch(1, 256);
    auto auxiliary_scope_0 = parallel.MakeAuxiliaryScratchScope(0);
    auto auxiliary_scope_1 = parallel.MakeAuxiliaryScratchScope(1);
    const auto auxiliary_0 = auxiliary_scope_0.AllocateBytes(256);
    const auto auxiliary_1 = auxiliary_scope_1.AllocateBytes(256);

    EXPECT_NE(primary.GetData(), auxiliary_0.GetData());
    EXPECT_NE(primary.GetData(), auxiliary_1.GetData());
    EXPECT_NE(auxiliary_0.GetData(), auxiliary_1.GetData());
    EXPECT_EQ(parallel.GetAuxiliaryScratchCapacityBytes(0), 256);
    EXPECT_EQ(parallel.GetAuxiliaryScratchCapacityBytes(1), 256);
    EXPECT_EQ(parallel.GetAuxiliaryScratchHighWaterBytes(0), 256);
    EXPECT_EQ(parallel.GetAuxiliaryScratchHighWaterBytes(1), 256);

    EXPECT_EQ(cudaMemsetAsync(primary.GetData(), 0x61, primary.GetSizeBytes(), guard.GetNativeStream()), cudaSuccess);
    EXPECT_EQ(
        cudaMemsetAsync(auxiliary_0.GetData(), 0x72, auxiliary_0.GetSizeBytes(), parallel.GetNativeAuxiliaryStream(0)),
        cudaSuccess);
    EXPECT_EQ(
        cudaMemsetAsync(auxiliary_1.GetData(), 0x83, auxiliary_1.GetSizeBytes(), parallel.GetNativeAuxiliaryStream(1)),
        cudaSuccess);
    parallel.CheckLaunch();
    parallel.Finish();
  }
  EXPECT_NO_THROW(GetContext().Synchronize());
}

}  // namespace
}  // namespace ttl::internal
