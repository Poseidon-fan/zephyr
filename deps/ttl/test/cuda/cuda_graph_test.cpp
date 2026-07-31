#include "ttl/cuda_graph.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
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
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/parallel_op_scope.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/ops/collective.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/runtime.hpp"
#include "ttl/scalar.hpp"
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

[[nodiscard]] auto MakeRuntimeOptions(const std::shared_ptr<ErrorSink> &error_sink, std::vector<Device> devices)
    -> RuntimeOptions {
  return RuntimeOptions{
      .devices_ = std::move(devices),
      .device_memory_ =
          {
              .enable_maintenance_thread_ = false,
          },
      .event_pool_capacity_per_device_ = 64,
      .error_sink_ = error_sink,
      .pinned_memory_ = {},
  };
}

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

void SubmitAuxiliaryMemset(ExecutionContext &context, Tensor &output, uint8_t value) {
  internal::OpGuard guard{context, "AuxiliaryMemset", std::source_location::current(), internal::CapturePolicy::SAFE};
  internal::ParallelOpScope parallel{guard, 1};
  parallel.RecordTensor(output, 0);
  internal::CheckCuda(
      cudaMemsetAsync(internal::TensorAccess::GetMutableData(output), value,
                      static_cast<size_t>(output.GetNumElements()), parallel.GetNativeAuxiliaryStream(0)),
      "cudaMemsetAsync (CUDA graph test)");
  parallel.CheckLaunch();
  parallel.Finish();
}

auto FailGraphInstantiation(cudaGraphExec_t * /*executable*/, cudaGraph_t /*graph*/, unsigned long long /*flags*/)
    -> cudaError_t {
  return cudaErrorInvalidValue;
}

class CudaGraphTest : public testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count == 0) {
      GTEST_SKIP() << "CUDA graph tests require a CUDA device";
    }
    error_sink_ = std::make_shared<RecordingErrorSink>();
    runtime_ = std::make_unique<Runtime>(MakeRuntimeOptions(error_sink_, {Device{0}}));
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

TEST_F(CudaGraphTest, CapturesAndReplaysFixedAddressOutOperations) {
  auto context = runtime_->CreateExecutionContext(Device{0});
  auto lhs = Empty(context, Shape{4}, DType::INT32);
  auto rhs = Empty(context, Shape{4}, DType::INT32);
  auto output = Empty(context, Shape{4}, DType::INT32);

  AddOut(context, output, lhs, rhs);
  context.Synchronize();

  auto session = context.BeginCapture(GraphCaptureOptions{.name_ = "elementwise"});
  AddOut(context, output, lhs, rhs);
  auto graph = session.Finish();
  auto foreign_context = runtime_->CreateExecutionContext(Device{0});

  EXPECT_EQ(graph.GetDevice(), Device{0});
  EXPECT_EQ(graph.GetStreamId(), context.GetStream().GetId());
  EXPECT_GT(graph.GetNodeCount(), 0);
  EXPECT_EQ(graph.GetLaunchCount(), 0);
  EXPECT_EQ(graph.GetName(), std::string_view{"elementwise"});
  EXPECT_THROW(graph.Launch(foreign_context), InvalidArgumentError);

  Upload(context, lhs, std::array<int32_t, 4>{1, 2, 3, 4});
  Upload(context, rhs, std::array<int32_t, 4>{10, 20, 30, 40});
  graph.Launch(context);
  EXPECT_EQ((Download<int32_t, 4>(context, output)), (std::array<int32_t, 4>{11, 22, 33, 44}));
  EXPECT_EQ(graph.GetLaunchCount(), 1);

  Upload(context, lhs, std::array<int32_t, 4>{5, 6, 7, 8});
  graph.Launch(context);
  EXPECT_EQ((Download<int32_t, 4>(context, output)), (std::array<int32_t, 4>{15, 26, 37, 48}));
  EXPECT_EQ(graph.GetLaunchCount(), 2);
}

TEST_F(CudaGraphTest, RejectsAllocationDuringCaptureAndRecoversByAbort) {
  auto context = runtime_->CreateExecutionContext(Device{0});
  const auto input = Ones(context, Shape{4}, DType::FLOAT32);

  auto session = context.BeginCapture();
  EXPECT_THROW([[maybe_unused]] auto output = Add(context, input, input), CaptureError);
  EXPECT_TRUE(session.IsActive());
  session.Abort();

  auto output = Empty(context, Shape{4}, DType::FLOAT32);
  AddOut(context, output, input, input);
  context.Synchronize();
}

TEST_F(CudaGraphTest, LeavesContextReadyWhenGraphInstantiationFails) {
  auto context = runtime_->CreateExecutionContext(Device{0});
  auto output = Empty(context, Shape{4}, DType::FLOAT32);
  FillOut(context, output, Scalar{1.0});
  context.Synchronize();

  auto session = context.BeginCapture();
  FillOut(context, output, Scalar{2.0});
  auto cuda_api = internal::GetCudaApi();
  cuda_api.instantiate_graph_ = FailGraphInstantiation;
  {
    internal::ScopedCudaApiOverride override{cuda_api};
    EXPECT_THROW(session.Finish(), CaptureError);
  }

  EXPECT_FALSE(session.IsActive());
  FillOut(context, output, Scalar{3.0});
  context.Synchronize();
}

TEST_F(CudaGraphTest, LiveGraphPreventsRuntimeShutdownAfterContextDestruction) {
  std::optional<CapturedGraph> graph;
  {
    auto context = runtime_->CreateExecutionContext(Device{0});
    auto output = Empty(context, Shape{4}, DType::FLOAT32);
    FillOut(context, output, Scalar{1.0});
    context.Synchronize();

    auto session = context.BeginCapture();
    FillOut(context, output, Scalar{2.0});
    graph.emplace(session.Finish());
  }

  EXPECT_THROW(runtime_->Shutdown(), InvalidArgumentError);
  graph.reset();
  runtime_->Shutdown();
}

TEST_F(CudaGraphTest, CapturesStructuredAuxiliaryStreams) {
  auto context = runtime_->CreateExecutionContext(Device{0}, ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
  auto output = Empty(context, Shape{128}, DType::UINT8);

  SubmitAuxiliaryMemset(context, output, uint8_t{0});
  context.Synchronize();

  auto session = context.BeginCapture(GraphCaptureOptions{.name_ = "multi_stream"});
  SubmitAuxiliaryMemset(context, output, uint8_t{0x5A});
  auto graph = session.Finish();
  graph.Launch(context);

  const auto values = Download<uint8_t, 128>(context, output);
  EXPECT_TRUE(std::ranges::all_of(values, [](uint8_t value) { return value == uint8_t{0x5A}; }));
}

TEST(CudaGraphGroupTest, CapturesAndReplaysNcclOnEveryLocalDevice) {
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2) {
    GTEST_SKIP() << "multi-GPU CUDA graph tests require two CUDA devices";
  }

  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}, Device{1}})};
  auto communicators = LocalCommunicatorGroup::Create(runtime, std::array{Device{0}, Device{1}});
  std::vector<ExecutionContext> contexts;
  contexts.reserve(2);
  contexts.push_back(runtime.CreateExecutionContext(Device{0}));
  contexts.push_back(runtime.CreateExecutionContext(Device{1}));
  std::array<Tensor, 2> staging_inputs{
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
  std::array<Tensor, 2> receives{
      Empty(contexts[0], Shape{2, 2}, DType::INT32),
      Empty(contexts[1], Shape{2, 2}, DType::INT32),
  };

  Upload(contexts[0], staging_inputs[0], std::array<int32_t, 4>{1, 2, 3, 4});
  Upload(contexts[1], staging_inputs[1], std::array<int32_t, 4>{10, 20, 30, 40});
  CopyOut(contexts[0], inputs[0], staging_inputs[0]);
  CopyOut(contexts[1], inputs[1], staging_inputs[1]);
  std::array<std::exception_ptr, 2> warmup_errors{};
  {
    std::array<std::jthread, 2> workers{
        std::jthread{[&] {
          try {
            AllReduceOut(contexts[0], outputs[0], inputs[0], communicators.GetCommunicator(0), ReduceOp::SUM);
            SendReceiveOut(contexts[0], staging_inputs[0], 1, receives[0], 1, communicators.GetCommunicator(0));
            Barrier(contexts[0], communicators.GetCommunicator(0));
          } catch (...) {
            warmup_errors[0] = std::current_exception();
          }
        }},
        std::jthread{[&] {
          try {
            AllReduceOut(contexts[1], outputs[1], inputs[1], communicators.GetCommunicator(1), ReduceOp::SUM);
            SendReceiveOut(contexts[1], staging_inputs[1], 0, receives[1], 0, communicators.GetCommunicator(1));
            Barrier(contexts[1], communicators.GetCommunicator(1));
          } catch (...) {
            warmup_errors[1] = std::current_exception();
          }
        }},
    };
  }
  ASSERT_EQ(warmup_errors[0], nullptr);
  ASSERT_EQ(warmup_errors[1], nullptr);
  contexts[0].Synchronize();
  contexts[1].Synchronize();

  {
    auto graph_group = CapturedGraphGroup::Capture(
        std::move(contexts),
        [&](size_t rank, ExecutionContext &context) {
          AllReduceOut(context, outputs[rank], inputs[rank], communicators.GetCommunicator(rank), ReduceOp::SUM);
          const auto peer = static_cast<int32_t>(rank == 0 ? 1 : 0);
          SendReceiveOut(context, staging_inputs[rank], peer, receives[rank], peer,
                         communicators.GetCommunicator(rank));
          Barrier(context, communicators.GetCommunicator(rank));
        },
        GraphGroupCaptureOptions{.name_ = "tensor_parallel"});

    EXPECT_EQ(graph_group.GetWorldSize(), 2);
    EXPECT_EQ(graph_group.GetDevice(0), Device{0});
    EXPECT_EQ(graph_group.GetDevice(1), Device{1});
    EXPECT_GT(graph_group.GetNodeCount(0), 0);
    EXPECT_GT(graph_group.GetNodeCount(1), 0);

    Upload(graph_group.GetContext(0), staging_inputs[0], std::array<int32_t, 4>{2, 4, 6, 8});
    Upload(graph_group.GetContext(1), staging_inputs[1], std::array<int32_t, 4>{1, 3, 5, 7});
    CopyOut(graph_group.GetContext(0), inputs[0], staging_inputs[0]);
    CopyOut(graph_group.GetContext(1), inputs[1], staging_inputs[1]);
    graph_group.Launch();
    const auto output_0 = Contiguous(graph_group.GetContext(0), outputs[0]);
    const auto output_1 = Contiguous(graph_group.GetContext(1), outputs[1]);
    EXPECT_EQ((Download<int32_t, 4>(graph_group.GetContext(0), output_0)), (std::array<int32_t, 4>{3, 7, 11, 15}));
    EXPECT_EQ((Download<int32_t, 4>(graph_group.GetContext(1), output_1)), (std::array<int32_t, 4>{3, 7, 11, 15}));
    EXPECT_EQ((Download<int32_t, 4>(graph_group.GetContext(0), receives[0])), (std::array<int32_t, 4>{1, 3, 5, 7}));
    EXPECT_EQ((Download<int32_t, 4>(graph_group.GetContext(1), receives[1])), (std::array<int32_t, 4>{2, 4, 6, 8}));
    EXPECT_EQ(graph_group.GetLaunchCount(), 1);
    EXPECT_THROW(communicators.Close(), InvalidArgumentError);
  }

  communicators.Close();
  EXPECT_TRUE(error_sink->IsEmpty());
}

}  // namespace
}  // namespace ttl
