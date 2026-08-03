#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "support/tensor_test_utils.hpp"
#include "ttl/common/error.hpp"
#include "ttl/distributed/collective.hpp"
#include "ttl/distributed/communicator.hpp"
#include "ttl/distributed/nccl_launch.hpp"
#include "ttl/runtime/device.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

[[nodiscard]] auto RequireTwoDevices() -> bool {
  int device_count = 0;
  const auto status = cudaGetDeviceCount(&device_count);
  EXPECT_EQ(status, cudaSuccess);
  return status == cudaSuccess && device_count >= 2;
}

[[nodiscard]] auto MakeCalls(std::array<ExecutionContext, 2> &contexts, std::array<Tensor, 2> &outputs,
                             const std::array<Tensor, 2> &inputs, LocalCommunicatorGroup &group)
    -> std::array<LocalCollectiveCall, 2> {
  return {
      LocalCollectiveCall{
          .context_ = contexts.data(),
          .output_ = outputs.data(),
          .input_ = inputs.data(),
          .communicator_ = &group.GetCommunicator(0),
      },
      LocalCollectiveCall{
          .context_ = &contexts[1],
          .output_ = &outputs[1],
          .input_ = &inputs[1],
          .communicator_ = &group.GetCommunicator(1),
      },
  };
}

void Synchronize(std::array<ExecutionContext, 2> &contexts) {
  for (auto &context : contexts) {
    context.Synchronize();
  }
}

}  // namespace

TEST(CollectiveIntegrationTest, ExecutesProcessLocalCollectivesAndPointToPoint) {
  if (!RequireTwoDevices()) {
    GTEST_SKIP() << "requires at least two CUDA devices";
  }

  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}, Device{1}}, sink)};
  {
    std::array contexts{runtime.CreateExecutionContext(Device{0}), runtime.CreateExecutionContext(Device{1})};
    const std::array rank_order{Device{0}, Device{1}};
    auto group = LocalCommunicatorGroup::Create(runtime, rank_order);

    std::array inputs{
        test::Upload(contexts[0], Shape{2}, std::vector<float>{1, 2}),
        test::Upload(contexts[1], Shape{2}, std::vector<float>{10, 20}),
    };
    std::array outputs{Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    auto calls = MakeCalls(contexts, outputs, inputs, group);
    AllReduceLocal(calls, ReduceOp::SUM);
    Synchronize(contexts);
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), (std::vector<float>{11, 22}));
    EXPECT_EQ(test::Download<float>(contexts[1], outputs[1]), (std::vector<float>{11, 22}));

    outputs = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    calls = MakeCalls(contexts, outputs, inputs, group);
    ReduceLocal(calls, ReduceOp::SUM, 0);
    Synchronize(contexts);
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), (std::vector<float>{11, 22}));

    outputs = {Empty(contexts[0], Shape{4}, DType::FLOAT32), Empty(contexts[1], Shape{4}, DType::FLOAT32)};
    calls = MakeCalls(contexts, outputs, inputs, group);
    AllGatherLocal(calls);
    Synchronize(contexts);
    const auto gathered = std::vector<float>{1, 2, 10, 20};
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), gathered);
    EXPECT_EQ(test::Download<float>(contexts[1], outputs[1]), gathered);

    inputs = {
        test::Upload(contexts[0], Shape{4}, std::vector<float>{1, 2, 3, 4}),
        test::Upload(contexts[1], Shape{4}, std::vector<float>{10, 20, 30, 40}),
    };
    outputs = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    calls = MakeCalls(contexts, outputs, inputs, group);
    ReduceScatterLocal(calls, ReduceOp::SUM);
    Synchronize(contexts);
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), (std::vector<float>{11, 22}));
    EXPECT_EQ(test::Download<float>(contexts[1], outputs[1]), (std::vector<float>{33, 44}));

    inputs = {
        test::Upload(contexts[0], Shape{2}, std::vector<float>{3, 4}),
        test::Upload(contexts[1], Shape{2}, std::vector<float>{30, 40}),
    };
    outputs = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    calls = MakeCalls(contexts, outputs, inputs, group);
    BroadcastLocal(calls, 0);
    Synchronize(contexts);
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), (std::vector<float>{3, 4}));
    EXPECT_EQ(test::Download<float>(contexts[1], outputs[1]), (std::vector<float>{3, 4}));

    inputs = {
        test::Upload(contexts[0], Shape{4}, std::vector<float>{0, 1, 2, 3}),
        test::Upload(contexts[1], Shape{4}, std::vector<float>{10, 11, 12, 13}),
    };
    outputs = {Empty(contexts[0], Shape{4}, DType::FLOAT32), Empty(contexts[1], Shape{4}, DType::FLOAT32)};
    calls = MakeCalls(contexts, outputs, inputs, group);
    AllToAllLocal(calls);
    Synchronize(contexts);
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), (std::vector<float>{0, 1, 10, 11}));
    EXPECT_EQ(test::Download<float>(contexts[1], outputs[1]), (std::vector<float>{2, 3, 12, 13}));

    inputs = {
        test::Upload(contexts[0], Shape{4}, std::vector<float>{0, 10, 11, 12}),
        test::Upload(contexts[1], Shape{3}, std::vector<float>{20, 21, 1}),
    };
    outputs = {Empty(contexts[0], Shape{3}, DType::FLOAT32), Empty(contexts[1], Shape{4}, DType::FLOAT32)};
    const std::array<std::array<int64_t, 2>, 2> send_counts{{{1, 3}, {2, 1}}};
    const std::array<std::array<int64_t, 2>, 2> receive_counts{{{1, 2}, {3, 1}}};
    const std::array variable_calls{
        LocalAllToAllVCall{
            .context_ = contexts.data(),
            .output_ = outputs.data(),
            .input_ = inputs.data(),
            .send_counts_ = send_counts[0],
            .receive_counts_ = receive_counts[0],
            .communicator_ = &group.GetCommunicator(0),
        },
        LocalAllToAllVCall{
            .context_ = &contexts[1],
            .output_ = &outputs[1],
            .input_ = &inputs[1],
            .send_counts_ = send_counts[1],
            .receive_counts_ = receive_counts[1],
            .communicator_ = &group.GetCommunicator(1),
        },
    };
    AllToAllVLocal(variable_calls);
    Synchronize(contexts);
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), (std::vector<float>{0, 20, 21}));
    EXPECT_EQ(test::Download<float>(contexts[1], outputs[1]), (std::vector<float>{10, 11, 12, 1}));

    inputs = {
        test::Upload(contexts[0], Shape{2}, std::vector<float>{5, 6}),
        test::Upload(contexts[1], Shape{2}, std::vector<float>{50, 60}),
    };
    outputs = {Empty(contexts[0], Shape{4}, DType::FLOAT32), Empty(contexts[1], Shape{4}, DType::FLOAT32)};
    calls = MakeCalls(contexts, outputs, inputs, group);
    GatherLocal(calls, 0);
    Synchronize(contexts);
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), (std::vector<float>{5, 6, 50, 60}));

    inputs = {
        test::Upload(contexts[0], Shape{4}, std::vector<float>{7, 8, 70, 80}),
        test::Upload(contexts[1], Shape{4}, std::vector<float>{-1, -1, -1, -1}),
    };
    outputs = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    calls = MakeCalls(contexts, outputs, inputs, group);
    ScatterLocal(calls, 0);
    Synchronize(contexts);
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), (std::vector<float>{7, 8}));
    EXPECT_EQ(test::Download<float>(contexts[1], outputs[1]), (std::vector<float>{70, 80}));

    std::array send{
        test::Upload(contexts[0], Shape{2}, std::vector<int32_t>{100, 101}),
        test::Upload(contexts[1], Shape{2}, std::vector<int32_t>{200, 201}),
    };
    std::array receive{Empty(contexts[0], Shape{2}, DType::INT32), Empty(contexts[1], Shape{2}, DType::INT32)};
    const std::array point_to_point{
        LocalPointToPointCall{
            .context_ = contexts.data(),
            .send_ = send.data(),
            .send_peer_ = 1,
            .receive_ = receive.data(),
            .receive_peer_ = 1,
            .communicator_ = &group.GetCommunicator(0),
        },
        LocalPointToPointCall{
            .context_ = &contexts[1],
            .send_ = &send[1],
            .send_peer_ = 0,
            .receive_ = &receive[1],
            .receive_peer_ = 0,
            .communicator_ = &group.GetCommunicator(1),
        },
    };
    SendReceiveLocal(point_to_point);
    Synchronize(contexts);
    EXPECT_EQ(test::Download<int32_t>(contexts[0], receive[0]), (std::vector<int32_t>{200, 201}));
    EXPECT_EQ(test::Download<int32_t>(contexts[1], receive[1]), (std::vector<int32_t>{100, 101}));

    const std::array barriers{
        LocalBarrierCall{.context_ = contexts.data(), .communicator_ = &group.GetCommunicator(0)},
        LocalBarrierCall{.context_ = &contexts[1], .communicator_ = &group.GetCommunicator(1)},
    };
    BarrierLocal(barriers);
    Synchronize(contexts);
    group.Close();
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(CollectiveIntegrationTest, ValidatesEveryRankBeforeEnqueue) {
  if (!RequireTwoDevices()) {
    GTEST_SKIP() << "requires at least two CUDA devices";
  }

  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}, Device{1}}, sink)};
  {
    std::array contexts{runtime.CreateExecutionContext(Device{0}), runtime.CreateExecutionContext(Device{1})};
    const std::array rank_order{Device{0}, Device{1}};
    auto group = LocalCommunicatorGroup::Create(runtime, rank_order);
    std::array inputs{
        Empty(contexts[0], Shape{2}, DType::FLOAT32),
        Empty(contexts[1], Shape{3}, DType::FLOAT32),
    };
    std::array outputs{
        Empty(contexts[0], Shape{2}, DType::FLOAT32),
        Empty(contexts[1], Shape{3}, DType::FLOAT32),
    };
    auto calls = MakeCalls(contexts, outputs, inputs, group);
    EXPECT_THROW(AllReduceLocal(calls, ReduceOp::SUM), InvalidArgumentError);
    group.Close();
  }
  runtime.Shutdown();
}

TEST(CollectiveIntegrationTest, ExecutesCheckedCustomNcclSubmission) {
  if (!RequireTwoDevices()) {
    GTEST_SKIP() << "requires at least two CUDA devices";
  }

  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}, Device{1}}, sink)};
  {
    std::array contexts{runtime.CreateExecutionContext(Device{0}), runtime.CreateExecutionContext(Device{1})};
    const std::array rank_order{Device{0}, Device{1}};
    auto group = LocalCommunicatorGroup::Create(runtime, rank_order);
    std::array inputs{
        test::Upload(contexts[0], Shape{2}, std::vector<float>{1, 2}),
        test::Upload(contexts[1], Shape{2}, std::vector<float>{10, 20}),
    };
    std::array outputs{Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    std::array output_pointers{outputs.data(), &outputs[1]};
    const std::array calls{
        LocalNcclKernelCall{
            .context_ = contexts.data(),
            .communicator_ = &group.GetCommunicator(0),
            .inputs_ = std::span<const Tensor>{inputs.data(), 1},
            .outputs_ = std::span<Tensor *const>{output_pointers.data(), 1},
        },
        LocalNcclKernelCall{
            .context_ = &contexts[1],
            .communicator_ = &group.GetCommunicator(1),
            .inputs_ = std::span<const Tensor>{&inputs[1], 1},
            .outputs_ = std::span<Tensor *const>{&output_pointers[1], 1},
        },
    };

    SubmitNcclKernelsLocal(calls, "custom all-reduce", [&](size_t index, NcclKernelLaunch &launch) {
      auto &cuda_launch = launch.GetCudaLaunch();
      return ncclAllReduce(cuda_launch.GetInputDataAs<float>(inputs[index]),
                           cuda_launch.GetOutputDataAs<float>(outputs[index]), 2, ncclFloat32, ncclSum,
                           launch.GetCommunicator(), cuda_launch.GetStream());
    });
    Synchronize(contexts);
    EXPECT_EQ(test::Download<float>(contexts[0], outputs[0]), (std::vector<float>{11, 22}));
    EXPECT_EQ(test::Download<float>(contexts[1], outputs[1]), (std::vector<float>{11, 22}));
    group.Close();
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(CollectiveIntegrationTest, MovedFromCommunicatorHandlesFailDeterministically) {
  if (!RequireTwoDevices()) {
    GTEST_SKIP() << "requires at least two CUDA devices";
  }

  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}, Device{1}}, sink)};
  {
    const std::array rank_order{Device{0}, Device{1}};
    auto group = LocalCommunicatorGroup::Create(runtime, rank_order);

    auto communicator = std::move(group.GetCommunicator(0));
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(static_cast<void>(group.GetCommunicator(0).GetDevice()), InvalidArgumentError);
    EXPECT_EQ(communicator.GetRank(), 0);

    auto moved_group = std::move(group);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(static_cast<void>(group.GetWorldSize()), InvalidArgumentError);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(static_cast<void>(group.GetStatus()), InvalidArgumentError);
    // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
    EXPECT_THROW(group.Poll(), InvalidArgumentError);
    moved_group.Close();
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

}  // namespace ttl
