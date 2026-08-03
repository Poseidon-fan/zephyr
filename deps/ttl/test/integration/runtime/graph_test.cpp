#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "support/tensor_test_utils.hpp"
#include "ttl/common/error.hpp"
#include "ttl/distributed/collective.hpp"
#include "ttl/distributed/communicator.hpp"
#include "ttl/distributed/nccl_launch.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/runtime/device.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/graph.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/scalar.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

[[nodiscard]] auto HasTwoDevices() -> bool {
  int device_count = 0;
  const auto status = cudaGetDeviceCount(&device_count);
  EXPECT_EQ(status, cudaSuccess);
  return status == cudaSuccess && device_count >= 2;
}

}  // namespace

TEST(GraphIntegrationTest, RejectsWrongReplayStreamAndRetainsRuntimeRegistration) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  std::optional<CapturedGraph> graph;
  {
    auto capture_context = runtime.CreateExecutionContext(Device{0});
    auto other_context = runtime.CreateExecutionContext(Device{0});
    auto output = Empty(capture_context, Shape{4}, DType::FLOAT32);
    FillOut(capture_context, output, Scalar{1.0F});
    capture_context.Synchronize();

    auto session = capture_context.BeginCapture(GraphCaptureOptions{.name_ = "fill"});
    FillOut(capture_context, output, Scalar{2.0F});
    graph.emplace(session.Finish());
    EXPECT_THROW(graph->Launch(other_context), InvalidArgumentError);
    graph->Launch(capture_context);
    capture_context.Synchronize();
    EXPECT_EQ(test::Download<float>(capture_context, output), (std::vector<float>{2, 2, 2, 2}));
  }

  EXPECT_THROW(runtime.Shutdown(), InvalidArgumentError);
  graph.reset();
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(GraphIntegrationTest, CaptureTransactionOwnsContextStateAfterPublicHandleDestruction) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  std::optional<CaptureSession> capture;
  std::optional<Tensor> output;
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    output.emplace(Empty(context, Shape{4}, DType::FLOAT32));
    FillOut(context, *output, Scalar{1.0F});
    context.Synchronize();
    capture.emplace(context.BeginCapture(GraphCaptureOptions{.name_ = "detached context"}));
    FillOut(context, *output, Scalar{2.0F});
  }

  {
    auto graph = capture->Finish();
    capture.reset();
    const auto statistics = runtime.GetStatistics();
    EXPECT_EQ(statistics.execution_context_count_, 0);
    EXPECT_EQ(statistics.captured_graph_count_, 1);
    EXPECT_EQ(statistics.active_capture_count_, 0);
    EXPECT_EQ(graph.GetNodeCount(), 1);
    EXPECT_THROW(runtime.Shutdown(), InvalidArgumentError);
  }
  output.reset();
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(GraphIntegrationTest, CaptureTransactionCanAbortOnAnotherHostThread) {
  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}}, sink)};
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    auto session = context.BeginCapture(GraphCaptureOptions{.name_ = "cross-thread abort"});
    std::thread abort_thread{[capture = std::move(session)]() mutable { capture.Abort(); }};
    abort_thread.join();

    auto output = Full(context, Shape{2}, Scalar{3.0F}, DType::FLOAT32);
    EXPECT_EQ(test::Download<float>(context, output), (std::vector<float>{3, 3}));
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

TEST(GraphIntegrationTest, CapturesAndReplaysCheckedNcclExtensionOnFixedRankWorkers) {
  if (!HasTwoDevices()) {
    GTEST_SKIP() << "requires at least two CUDA devices";
  }

  auto sink = std::make_shared<test::RecordingErrorSink>();
  Runtime runtime{test::MakeRuntimeOptions({Device{0}, Device{1}}, sink)};
  const std::array rank_order{Device{0}, Device{1}};
  auto communicator_group = LocalCommunicatorGroup::Create(runtime, rank_order);
  {
    std::vector<ExecutionContext> contexts;
    contexts.reserve(2);
    contexts.push_back(runtime.CreateExecutionContext(Device{0}));
    contexts.push_back(runtime.CreateExecutionContext(Device{1}));
    std::array inputs{
        test::Upload(contexts[0], Shape{2}, std::vector<float>{1, 2}),
        test::Upload(contexts[1], Shape{2}, std::vector<float>{10, 20}),
    };
    std::array outputs{Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    const std::array warmup_calls{
        LocalCollectiveCall{
            .context_ = contexts.data(),
            .output_ = outputs.data(),
            .input_ = inputs.data(),
            .communicator_ = &communicator_group.GetCommunicator(0),
        },
        LocalCollectiveCall{
            .context_ = &contexts[1],
            .output_ = &outputs[1],
            .input_ = &inputs[1],
            .communicator_ = &communicator_group.GetCommunicator(1),
        },
    };
    AllReduceLocal(warmup_calls, ReduceOp::SUM);
    for (auto &context : contexts) {
      context.Synchronize();
    }

    {
      auto graph_group = CapturedGraphGroup::Capture(
          std::move(contexts),
          [&](size_t rank, ExecutionContext &context) {
            const std::array rank_inputs{inputs[rank]};
            const std::array rank_outputs{&outputs[rank]};
            SubmitNcclKernel(
                context, communicator_group.GetCommunicator(rank), "captured custom all-reduce", rank_inputs,
                rank_outputs,
                [&](NcclKernelLaunch &launch) {
                  auto &cuda_launch = launch.GetCudaLaunch();
                  return ncclAllReduce(cuda_launch.GetInputDataAs<float>(inputs[rank]),
                                       cuda_launch.GetOutputDataAs<float>(outputs[rank]), 2, ncclFloat32, ncclSum,
                                       launch.GetCommunicator(), cuda_launch.GetStream());
                },
                CudaKernelLaunchOptions{.capture_policy_ = CudaCapturePolicy::SAFE});
          },
          GraphGroupCaptureOptions{.name_ = "two-rank all-reduce"});
      EXPECT_EQ(graph_group.GetWorldSize(), 2);
      EXPECT_THROW(communicator_group.Close(), InvalidArgumentError);

      graph_group.Launch();
      graph_group.Synchronize();
      EXPECT_EQ(test::Download<float>(graph_group.GetContext(0), outputs[0]), (std::vector<float>{11, 22}));
      EXPECT_EQ(test::Download<float>(graph_group.GetContext(1), outputs[1]), (std::vector<float>{11, 22}));
    }
  }
  communicator_group.Close();
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

}  // namespace ttl
