#include "ttl/ops/collective.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ttl/communicator.hpp"
#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/runtime.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override {
    try {
      std::scoped_lock lock{latch_};
      errors_.push_back(std::move(error));
    } catch (...) {
      return;
    }
  }

  [[nodiscard]] auto IsEmpty() const noexcept -> bool {
    std::scoped_lock lock{latch_};
    return errors_.empty();
  }

 private:
  mutable std::mutex latch_;
  std::vector<ErrorRecord> errors_;
};

template <typename T, size_t SIZE>
void Upload(ExecutionContext &context, Tensor &tensor, const std::array<T, SIZE> &values) {
  CopyFromHostBlocking(context, tensor, std::as_bytes(std::span{values}));
}

template <typename T, size_t SIZE>
[[nodiscard]] auto Download(ExecutionContext &context, const Tensor &tensor) -> std::array<T, SIZE> {
  std::array<T, SIZE> values{};
  CopyToHostBlocking(context, std::as_writable_bytes(std::span{values}), tensor);
  return values;
}

class CollectiveTest : public testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
      GTEST_SKIP() << "collective integration tests require two CUDA devices";
    }
    error_sink_ = std::make_shared<RecordingErrorSink>();
    runtime_ = std::make_unique<Runtime>(RuntimeOptions{
        .devices_ = {Device{0}, Device{1}},
        .device_memory_ =
            {
                .enable_maintenance_thread_ = false,
            },
        .error_sink_ = error_sink_,
        .pinned_memory_ = {},
    });
  }

  void TearDown() override {
    if (runtime_ != nullptr) {
      runtime_->Shutdown();
    }
    EXPECT_TRUE(error_sink_ == nullptr || error_sink_->IsEmpty());
  }

  std::shared_ptr<RecordingErrorSink> error_sink_;
  std::unique_ptr<Runtime> runtime_;
};

TEST_F(CollectiveTest, ExecutesProcessLocalCollectivesAndPointToPoint) {
  std::vector<ExecutionContext> contexts;
  contexts.reserve(2);
  contexts.push_back(runtime_->CreateExecutionContext(Device{0}));
  contexts.push_back(runtime_->CreateExecutionContext(Device{1}));
  auto group = LocalCommunicatorGroup::Create(*runtime_, std::array{Device{0}, Device{1}});

  auto RunCollective = [&](auto operation, std::span<Tensor> outputs, std::span<const Tensor> inputs) {
    std::array<LocalCollectiveCall, 2> calls{};
    for (size_t rank = 0; rank < calls.size(); rank++) {
      calls[rank] = LocalCollectiveCall{
          .context_ = &contexts[rank],
          .output_ = &outputs[rank],
          .input_ = &inputs[rank],
          .communicator_ = &group.GetCommunicator(rank),
      };
    }
    operation(std::span<const LocalCollectiveCall>{calls});
  };

  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  inputs.reserve(2);
  outputs.reserve(2);
  for (auto &context : contexts) {
    inputs.push_back(Empty(context, Shape{4}, DType::INT32));
    outputs.push_back(Empty(context, Shape{4}, DType::INT32));
  }
  Upload(contexts[0], inputs[0], std::array<int32_t, 4>{1, 2, 3, 4});
  Upload(contexts[1], inputs[1], std::array<int32_t, 4>{10, 20, 30, 40});
  RunCollective([&](auto calls) { AllReduceLocal(calls, ReduceOp::SUM); }, outputs, inputs);
  EXPECT_EQ((Download<int32_t, 4>(contexts[0], outputs[0])), (std::array<int32_t, 4>{11, 22, 33, 44}));
  EXPECT_EQ((Download<int32_t, 4>(contexts[1], outputs[1])), (std::array<int32_t, 4>{11, 22, 33, 44}));

  inputs.clear();
  outputs.clear();
  for (auto &context : contexts) {
    inputs.push_back(Empty(context, Shape{2}, DType::INT32));
    outputs.push_back(Empty(context, Shape{4}, DType::INT32));
  }
  Upload(contexts[0], inputs[0], std::array<int32_t, 2>{1, 2});
  Upload(contexts[1], inputs[1], std::array<int32_t, 2>{3, 4});
  RunCollective([](auto calls) { AllGatherLocal(calls); }, outputs, inputs);
  EXPECT_EQ((Download<int32_t, 4>(contexts[0], outputs[0])), (std::array<int32_t, 4>{1, 2, 3, 4}));
  EXPECT_EQ((Download<int32_t, 4>(contexts[1], outputs[1])), (std::array<int32_t, 4>{1, 2, 3, 4}));

  inputs.clear();
  outputs.clear();
  for (auto &context : contexts) {
    inputs.push_back(Empty(context, Shape{4}, DType::INT32));
    outputs.push_back(Empty(context, Shape{2}, DType::INT32));
  }
  Upload(contexts[0], inputs[0], std::array<int32_t, 4>{1, 2, 3, 4});
  Upload(contexts[1], inputs[1], std::array<int32_t, 4>{10, 20, 30, 40});
  RunCollective([&](auto calls) { ReduceScatterLocal(calls, ReduceOp::SUM); }, outputs, inputs);
  EXPECT_EQ((Download<int32_t, 2>(contexts[0], outputs[0])), (std::array<int32_t, 2>{11, 22}));
  EXPECT_EQ((Download<int32_t, 2>(contexts[1], outputs[1])), (std::array<int32_t, 2>{33, 44}));

  inputs.clear();
  outputs.clear();
  for (auto &context : contexts) {
    inputs.push_back(Empty(context, Shape{2}, DType::INT32));
    outputs.push_back(Empty(context, Shape{2}, DType::INT32));
  }
  Upload(contexts[0], inputs[0], std::array<int32_t, 2>{1, 2});
  Upload(contexts[1], inputs[1], std::array<int32_t, 2>{10, 20});
  RunCollective([&](auto calls) { ReduceLocal(calls, ReduceOp::SUM, 0); }, outputs, inputs);
  EXPECT_EQ((Download<int32_t, 2>(contexts[0], outputs[0])), (std::array<int32_t, 2>{11, 22}));

  Upload(contexts[0], inputs[0], std::array<int32_t, 2>{1, 2});
  Upload(contexts[1], inputs[1], std::array<int32_t, 2>{7, 8});
  RunCollective([](auto calls) { BroadcastLocal(calls, 1); }, outputs, inputs);
  EXPECT_EQ((Download<int32_t, 2>(contexts[0], outputs[0])), (std::array<int32_t, 2>{7, 8}));
  EXPECT_EQ((Download<int32_t, 2>(contexts[1], outputs[1])), (std::array<int32_t, 2>{7, 8}));

  outputs.clear();
  for (auto &context : contexts) {
    outputs.push_back(Empty(context, Shape{4}, DType::INT32));
  }
  Upload(contexts[0], inputs[0], std::array<int32_t, 2>{1, 2});
  Upload(contexts[1], inputs[1], std::array<int32_t, 2>{3, 4});
  RunCollective([](auto calls) { GatherLocal(calls, 0); }, outputs, inputs);
  EXPECT_EQ((Download<int32_t, 4>(contexts[0], outputs[0])), (std::array<int32_t, 4>{1, 2, 3, 4}));

  inputs.clear();
  outputs.clear();
  for (auto &context : contexts) {
    inputs.push_back(Empty(context, Shape{4}, DType::INT32));
    outputs.push_back(Empty(context, Shape{2}, DType::INT32));
  }
  Upload(contexts[0], inputs[0], std::array<int32_t, 4>{5, 6, 7, 8});
  Upload(contexts[1], inputs[1], std::array<int32_t, 4>{0, 0, 0, 0});
  RunCollective([](auto calls) { ScatterLocal(calls, 0); }, outputs, inputs);
  EXPECT_EQ((Download<int32_t, 2>(contexts[0], outputs[0])), (std::array<int32_t, 2>{5, 6}));
  EXPECT_EQ((Download<int32_t, 2>(contexts[1], outputs[1])), (std::array<int32_t, 2>{7, 8}));

  inputs.clear();
  outputs.clear();
  for (auto &context : contexts) {
    inputs.push_back(Empty(context, Shape{4}, DType::INT32));
    outputs.push_back(Empty(context, Shape{4}, DType::INT32));
  }
  Upload(contexts[0], inputs[0], std::array<int32_t, 4>{0, 1, 2, 3});
  Upload(contexts[1], inputs[1], std::array<int32_t, 4>{10, 11, 12, 13});
  RunCollective([](auto calls) { AllToAllLocal(calls); }, outputs, inputs);
  EXPECT_EQ((Download<int32_t, 4>(contexts[0], outputs[0])), (std::array<int32_t, 4>{0, 1, 10, 11}));
  EXPECT_EQ((Download<int32_t, 4>(contexts[1], outputs[1])), (std::array<int32_t, 4>{2, 3, 12, 13}));

  std::array<LocalBarrierCall, 2> barrier_calls{};
  for (size_t rank = 0; rank < barrier_calls.size(); rank++) {
    barrier_calls[rank] = {
        .context_ = &contexts[rank],
        .communicator_ = &group.GetCommunicator(rank),
    };
  }
  BarrierLocal(barrier_calls);
  {
    std::array<ExecutionContext, 2> secondary_contexts{
        runtime_->CreateExecutionContext(Device{0}),
        runtime_->CreateExecutionContext(Device{1}),
    };
    const std::array<LocalBarrierCall, 2> secondary_calls{
        LocalBarrierCall{
            .context_ = &secondary_contexts[0],
            .communicator_ = &group.GetCommunicator(0),
        },
        LocalBarrierCall{
            .context_ = &secondary_contexts[1],
            .communicator_ = &group.GetCommunicator(1),
        },
    };
    BarrierLocal(secondary_calls);
  }
  BarrierLocal(barrier_calls);

  std::vector<Tensor> sends;
  std::vector<Tensor> receives;
  sends.reserve(2);
  receives.reserve(2);
  for (auto &context : contexts) {
    sends.push_back(Empty(context, Shape{2}, DType::INT32));
    receives.push_back(Empty(context, Shape{2}, DType::INT32));
  }
  Upload(contexts[0], sends[0], std::array<int32_t, 2>{5, 6});
  Upload(contexts[1], sends[1], std::array<int32_t, 2>{7, 8});
  const std::array<LocalPointToPointCall, 2> point_to_point_calls{
      LocalPointToPointCall{
          .context_ = &contexts[0],
          .send_ = &sends[0],
          .send_peer_ = 1,
          .receive_ = &receives[0],
          .receive_peer_ = 1,
          .communicator_ = &group.GetCommunicator(0),
      },
      LocalPointToPointCall{
          .context_ = &contexts[1],
          .send_ = &sends[1],
          .send_peer_ = 0,
          .receive_ = &receives[1],
          .receive_peer_ = 0,
          .communicator_ = &group.GetCommunicator(1),
      },
  };
  SendReceiveLocal(point_to_point_calls);
  EXPECT_EQ((Download<int32_t, 2>(contexts[0], receives[0])), (std::array<int32_t, 2>{7, 8}));
  EXPECT_EQ((Download<int32_t, 2>(contexts[1], receives[1])), (std::array<int32_t, 2>{5, 6}));

  group.Close();
  contexts.clear();
}

TEST_F(CollectiveTest, SupportsOneHostThreadPerRankSubmission) {
  std::array<ExecutionContext, 2> contexts{
      runtime_->CreateExecutionContext(Device{0}),
      runtime_->CreateExecutionContext(Device{1}),
  };
  auto group = LocalCommunicatorGroup::Create(*runtime_, std::array{Device{0}, Device{1}});
  std::array<Tensor, 2> inputs{
      Empty(contexts[0], Shape{2}, DType::INT32),
      Empty(contexts[1], Shape{2}, DType::INT32),
  };
  std::array<Tensor, 2> outputs{
      Empty(contexts[0], Shape{2}, DType::INT32),
      Empty(contexts[1], Shape{2}, DType::INT32),
  };
  std::array<Tensor, 2> receives{
      Empty(contexts[0], Shape{2}, DType::INT32),
      Empty(contexts[1], Shape{2}, DType::INT32),
  };
  Upload(contexts[0], inputs[0], std::array<int32_t, 2>{1, 2});
  Upload(contexts[1], inputs[1], std::array<int32_t, 2>{10, 20});

  std::array<std::exception_ptr, 2> errors{};
  std::array<std::jthread, 2> workers{
      std::jthread{[&] {
        try {
          AllReduceOut(contexts[0], outputs[0], inputs[0], group.GetCommunicator(0), ReduceOp::SUM);
          SendReceiveOut(contexts[0], inputs[0], 1, receives[0], 1, group.GetCommunicator(0));
        } catch (...) {
          errors[0] = std::current_exception();
        }
      }},
      std::jthread{[&] {
        try {
          AllReduceOut(contexts[1], outputs[1], inputs[1], group.GetCommunicator(1), ReduceOp::SUM);
          SendReceiveOut(contexts[1], inputs[1], 0, receives[1], 0, group.GetCommunicator(1));
        } catch (...) {
          errors[1] = std::current_exception();
        }
      }},
  };
  for (auto &worker : workers) {
    worker.join();
  }
  ASSERT_EQ(errors[0], nullptr);
  ASSERT_EQ(errors[1], nullptr);
  EXPECT_EQ((Download<int32_t, 2>(contexts[0], outputs[0])), (std::array<int32_t, 2>{11, 22}));
  EXPECT_EQ((Download<int32_t, 2>(contexts[1], outputs[1])), (std::array<int32_t, 2>{11, 22}));
  EXPECT_EQ((Download<int32_t, 2>(contexts[0], receives[0])), (std::array<int32_t, 2>{10, 20}));
  EXPECT_EQ((Download<int32_t, 2>(contexts[1], receives[1])), (std::array<int32_t, 2>{1, 2}));

  group.Close();
}

TEST_F(CollectiveTest, MaterializesNonContiguousInputAndOutput) {
  std::array<ExecutionContext, 2> contexts{
      runtime_->CreateExecutionContext(Device{0}),
      runtime_->CreateExecutionContext(Device{1}),
  };
  auto group = LocalCommunicatorGroup::Create(*runtime_, std::array{Device{0}, Device{1}});
  std::array<Tensor, 2> contiguous_inputs{
      Empty(contexts[0], Shape{2, 2}, DType::INT32),
      Empty(contexts[1], Shape{2, 2}, DType::INT32),
  };
  std::array<Tensor, 2> inputs{
      EmptyStrided(contexts[0], Shape{2, 2}, Strides{1, 2}, DType::INT32),
      EmptyStrided(contexts[1], Shape{2, 2}, Strides{1, 2}, DType::INT32),
  };
  std::array<Tensor, 2> outputs{
      EmptyStrided(contexts[0], Shape{2, 2}, Strides{1, 2}, DType::INT32),
      EmptyStrided(contexts[1], Shape{2, 2}, Strides{1, 2}, DType::INT32),
  };
  Upload(contexts[0], contiguous_inputs[0], std::array<int32_t, 4>{1, 2, 3, 4});
  Upload(contexts[1], contiguous_inputs[1], std::array<int32_t, 4>{10, 20, 30, 40});
  CopyOut(contexts[0], inputs[0], contiguous_inputs[0]);
  CopyOut(contexts[1], inputs[1], contiguous_inputs[1]);

  const std::array<LocalCollectiveCall, 2> calls{
      LocalCollectiveCall{
          .context_ = &contexts[0],
          .output_ = &outputs[0],
          .input_ = &inputs[0],
          .communicator_ = &group.GetCommunicator(0),
      },
      LocalCollectiveCall{
          .context_ = &contexts[1],
          .output_ = &outputs[1],
          .input_ = &inputs[1],
          .communicator_ = &group.GetCommunicator(1),
      },
  };
  AllReduceLocal(calls, ReduceOp::SUM);
  auto contiguous_output_0 = Contiguous(contexts[0], outputs[0]);
  auto contiguous_output_1 = Contiguous(contexts[1], outputs[1]);
  EXPECT_EQ((Download<int32_t, 4>(contexts[0], contiguous_output_0)), (std::array<int32_t, 4>{11, 22, 33, 44}));
  EXPECT_EQ((Download<int32_t, 4>(contexts[1], contiguous_output_1)), (std::array<int32_t, 4>{11, 22, 33, 44}));

  group.Close();
}

TEST_F(CollectiveTest, RuntimeShutdownRejectsAnOpenCommunicatorGroup) {
  auto group = LocalCommunicatorGroup::Create(*runtime_, std::array{Device{0}, Device{1}});

  EXPECT_THROW(runtime_->Shutdown(), InvalidArgumentError);

  group.Close();
  runtime_->Shutdown();
}

TEST_F(CollectiveTest, RejectsInvalidGroupConfigurationBeforeNcclInitialization) {
  const std::array duplicate_devices{Device{0}, Device{0}};
  EXPECT_THROW(LocalCommunicatorGroup::Create(*runtime_, duplicate_devices), InvalidArgumentError);

  auto options = NcclOptions{};
  options.enqueue_timeout_ = std::chrono::milliseconds{0};
  const std::array devices{Device{0}, Device{1}};
  EXPECT_THROW(LocalCommunicatorGroup::Create(*runtime_, devices, options), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl
