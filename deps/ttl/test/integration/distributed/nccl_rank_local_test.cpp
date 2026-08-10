#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <nccl.h>

#include "support/runtime_session.hpp"
#include "ttl/common/error.hpp"
#include "ttl/distributed/collective.hpp"
#include "ttl/distributed/communicator.hpp"
#include "ttl/distributed/nccl_launch.hpp"

namespace ttl::test {
namespace {

template <typename Function>
void RunRanks(std::array<ExecutionContext, 2> &contexts, std::array<NcclCommunicator *, 2> communicators,
              Function &&function) {
  std::array<std::exception_ptr, 2> failures{};
  std::array<std::thread, 2> workers;
  for (size_t rank = 0; rank < 2; ++rank) {
    workers[rank] = std::thread([&, rank] {
      try {
        std::invoke(function, rank, contexts[rank], *communicators[rank]);
      } catch (...) {
        failures[rank] = std::current_exception();
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  for (const auto &failure : failures) {
    if (failure != nullptr) {
      std::rethrow_exception(failure);
    }
  }
}

void Synchronize(std::array<ExecutionContext, 2> &contexts) {
  for (auto &context : contexts) {
    context.Synchronize();
  }
}

TEST(NcclRankLocalCollectiveTest, ExercisesEveryPublicRankLocalCollective) {
  const auto devices = GetTestDevices(2);
  if (devices.size() < 2) {
    GTEST_SKIP() << "requires at least two CUDA devices";
  }

  auto sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions({devices[0], devices[1]}, sink)};
  {
    std::array contexts{runtime.CreateExecutionContext(devices[0]), runtime.CreateExecutionContext(devices[1])};
    auto group = LocalCommunicatorGroup::Create(runtime, std::array{devices[0], devices[1]});
    std::array communicators{&group.GetCommunicator(0), &group.GetCommunicator(1)};

    std::array inputs{Upload(contexts[0], Shape{2}, std::vector<float>{1, 2}),
                      Upload(contexts[1], Shape{2}, std::vector<float>{10, 20})};
    std::array outputs{Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};

    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      AllReduceOut(context, outputs[rank], inputs[rank], communicator, ReduceOp::SUM);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], outputs[0]), (std::vector<float>{11, 22}));
    EXPECT_EQ(Download<float>(contexts[1], outputs[1]), (std::vector<float>{11, 22}));

    std::array extension_outputs{Empty(contexts[0], Shape{2}, DType::FLOAT32),
                                 Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      const std::array<Tensor, 1> rank_inputs{inputs[rank]};
      const std::array<Tensor *, 1> rank_outputs{&extension_outputs[rank]};
      SubmitNcclKernel(context, communicator, "rank-local all-reduce", rank_inputs, rank_outputs,
                       [&](NcclKernelLaunch &launch) {
                         auto &cuda_launch = launch.GetCudaLaunch();
                         return ncclAllReduce(cuda_launch.GetInputDataAs<float>(inputs[rank]),
                                              cuda_launch.GetOutputDataAs<float>(extension_outputs[rank]), 2,
                                              ncclFloat32, ncclSum, launch.GetCommunicator(), cuda_launch.GetStream());
                       });
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], extension_outputs[0]), (std::vector<float>{11, 22}));
    EXPECT_EQ(Download<float>(contexts[1], extension_outputs[1]), (std::vector<float>{11, 22}));

    outputs = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      ReduceOut(context, outputs[rank], inputs[rank], communicator, ReduceOp::SUM, 0);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], outputs[0]), (std::vector<float>{11, 22}));

    outputs = {Empty(contexts[0], Shape{4}, DType::FLOAT32), Empty(contexts[1], Shape{4}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      AllGatherOut(context, outputs[rank], inputs[rank], communicator);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], outputs[0]), (std::vector<float>{1, 2, 10, 20}));
    EXPECT_EQ(Download<float>(contexts[1], outputs[1]), (std::vector<float>{1, 2, 10, 20}));

    inputs = {Upload(contexts[0], Shape{4}, std::vector<float>{1, 2, 3, 4}),
              Upload(contexts[1], Shape{4}, std::vector<float>{10, 20, 30, 40})};
    outputs = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      ReduceScatterOut(context, outputs[rank], inputs[rank], communicator, ReduceOp::SUM);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], outputs[0]), (std::vector<float>{11, 22}));
    EXPECT_EQ(Download<float>(contexts[1], outputs[1]), (std::vector<float>{33, 44}));

    inputs = {Upload(contexts[0], Shape{2}, std::vector<float>{3, 4}),
              Upload(contexts[1], Shape{2}, std::vector<float>{30, 40})};
    outputs = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      BroadcastOut(context, outputs[rank], inputs[rank], communicator, 0);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], outputs[0]), (std::vector<float>{3, 4}));
    EXPECT_EQ(Download<float>(contexts[1], outputs[1]), (std::vector<float>{3, 4}));

    inputs = {Upload(contexts[0], Shape{4}, std::vector<float>{0, 1, 2, 3}),
              Upload(contexts[1], Shape{4}, std::vector<float>{10, 11, 12, 13})};
    outputs = {Empty(contexts[0], Shape{4}, DType::FLOAT32), Empty(contexts[1], Shape{4}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      AllToAllOut(context, outputs[rank], inputs[rank], communicator);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], outputs[0]), (std::vector<float>{0, 1, 10, 11}));
    EXPECT_EQ(Download<float>(contexts[1], outputs[1]), (std::vector<float>{2, 3, 12, 13}));

    inputs = {Upload(contexts[0], Shape{2}, std::vector<float>{0, 1}),
              Upload(contexts[1], Shape{2}, std::vector<float>{10, 11})};
    outputs = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    constexpr std::array<int64_t, 2> counts{1, 1};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      AllToAllVOut(context, outputs[rank], inputs[rank], counts, counts, communicator);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], outputs[0]), (std::vector<float>{0, 10}));
    EXPECT_EQ(Download<float>(contexts[1], outputs[1]), (std::vector<float>{1, 11}));

    inputs = {Upload(contexts[0], Shape{2}, std::vector<float>{5, 6}),
              Upload(contexts[1], Shape{2}, std::vector<float>{50, 60})};
    outputs = {Empty(contexts[0], Shape{4}, DType::FLOAT32), Empty(contexts[1], Shape{4}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      GatherOut(context, outputs[rank], inputs[rank], communicator, 0);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], outputs[0]), (std::vector<float>{5, 6, 50, 60}));

    inputs = {Upload(contexts[0], Shape{4}, std::vector<float>{7, 8, 70, 80}),
              Upload(contexts[1], Shape{4}, std::vector<float>{-1, -1, -1, -1})};
    outputs = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      ScatterOut(context, outputs[rank], inputs[rank], communicator, 0);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], outputs[0]), (std::vector<float>{7, 8}));
    EXPECT_EQ(Download<float>(contexts[1], outputs[1]), (std::vector<float>{70, 80}));

    std::array send{Upload(contexts[0], Shape{2}, std::vector<float>{100, 101}),
                    Upload(contexts[1], Shape{2}, std::vector<float>{200, 201})};
    std::array receive{Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      if (rank == 0) {
        Send(context, send[rank], communicator, 1);
      } else {
        ReceiveOut(context, receive[rank], communicator, 0);
      }
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[1], receive[1]), (std::vector<float>{100, 101}));

    receive = {Empty(contexts[0], Shape{2}, DType::FLOAT32), Empty(contexts[1], Shape{2}, DType::FLOAT32)};
    RunRanks(contexts, communicators, [&](size_t rank, ExecutionContext &context, NcclCommunicator &communicator) {
      SendReceiveOut(context, send[rank], static_cast<int32_t>(1 - rank), receive[rank], static_cast<int32_t>(1 - rank),
                     communicator);
    });
    Synchronize(contexts);
    EXPECT_EQ(Download<float>(contexts[0], receive[0]), (std::vector<float>{200, 201}));
    EXPECT_EQ(Download<float>(contexts[1], receive[1]), (std::vector<float>{100, 101}));

    RunRanks(contexts, communicators,
             [](size_t, ExecutionContext &context, NcclCommunicator &communicator) { Barrier(context, communicator); });
    Synchronize(contexts);
    for (NcclCommunicator *communicator : communicators) {
      communicator->PollAsyncError();
    }
    group.Close();
  }
  runtime.Shutdown();
  EXPECT_TRUE(sink->GetRecords().empty());
}

}  // namespace
}  // namespace ttl::test
