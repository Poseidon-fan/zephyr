#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "support/tensor_test_utils.hpp"
#include "ttl/common/error.hpp"
#include "ttl/runtime/graph.hpp"
#include "ttl/runtime/kernel_launch.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

__global__ void AddOneKernel(const float *input, float *output, size_t count) {
  const auto index = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index < count) {
    output[index] = input[index] + 1.0F;
  }
}

void SubmitAddOne(ExecutionContext &context, const Tensor &input, Tensor &output, CudaCapturePolicy capture_policy) {
  const std::array inputs{input};
  const std::array outputs{&output};
  SubmitCudaKernel(
      context, "external AddOne", inputs, outputs,
      [&](CudaKernelLaunch &launch) {
        constexpr uint32_t block_size = 128;
        const auto count = static_cast<size_t>(input.GetNumElements());
        const auto block_count = static_cast<uint32_t>((count + block_size - 1) / block_size);
        AddOneKernel<<<block_count, block_size, 0, launch.GetStream()>>>(launch.GetInputDataAs<float>(input),
                                                                         launch.GetOutputDataAs<float>(output), count);
      },
      CudaKernelLaunchOptions{
          .workspace_bytes_ = 1024,
          .workspace_alignment_ = 256,
          .capture_policy_ = capture_policy,
      });
}

}  // namespace

TEST(CudaKernelLaunchTest, RunsExternalKernelUsingOnlyPublicApi) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{5}, std::vector<float>{-2.0F, -0.0F, 1.0F, 4.0F, 9.0F});
  auto output = Empty(context, Shape{5}, DType::FLOAT32);

  SubmitAddOne(context, input, output, CudaCapturePolicy::FORBIDDEN);
  EXPECT_EQ(test::Download<float>(context, output), (std::vector<float>{-1.0F, 1.0F, 2.0F, 5.0F, 10.0F}));
}

TEST(CudaKernelLaunchTest, RejectsUnregisteredAndWronglyTypedPointers) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = Empty(context, Shape{1}, DType::FLOAT32);
  auto output = Empty(context, Shape{1}, DType::FLOAT32);
  auto unrelated = Empty(context, Shape{1}, DType::FLOAT32);
  const std::array inputs{input};
  const std::array outputs{&output};

  SubmitCudaKernel(context, "pointer validation", inputs, outputs, [&](CudaKernelLaunch &launch) {
    EXPECT_THROW(static_cast<void>(launch.GetInputData(unrelated)), InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(launch.GetOutputData(input)), InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(launch.GetInputDataAs<int32_t>(input)), InvalidArgumentError);
    EXPECT_THROW(static_cast<void>(launch.GetOutputDataAs<int32_t>(output)), InvalidArgumentError);
  });
}

TEST(CudaKernelLaunchTest, RejectsInvalidWorkspaceAndCaptureOptions) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto output = Empty(context, Shape{1}, DType::FLOAT32);
  const std::array<Tensor, 0> inputs{};
  const std::array outputs{&output};
  const auto no_op = [](CudaKernelLaunch &) {};

  EXPECT_THROW(SubmitCudaKernel(context, "invalid alignment", inputs, outputs, no_op,
                                CudaKernelLaunchOptions{.workspace_bytes_ = 1, .workspace_alignment_ = 3}),
               InvalidArgumentError);
  EXPECT_THROW(SubmitCudaKernel(context, "invalid capture policy", inputs, outputs, no_op,
                                CudaKernelLaunchOptions{
                                    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): invalid input test.
                                    .capture_policy_ = static_cast<CudaCapturePolicy>(255),
                                }),
               InvalidArgumentError);
}

TEST(CudaKernelLaunchTest, ClearsLaunchErrorBeforeTheNextSubmission) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{1}, std::vector<float>{4.0F});
  auto output = Empty(context, Shape{1}, DType::FLOAT32);
  const std::array inputs{input};
  const std::array outputs{&output};

  EXPECT_THROW(SubmitCudaKernel(context, "invalid launch", inputs, outputs,
                                [&](CudaKernelLaunch &launch) {
                                  AddOneKernel<<<0, 128, 0, launch.GetStream()>>>(
                                      launch.GetInputDataAs<float>(input), launch.GetOutputDataAs<float>(output), 1);
                                }),
               CudaError);

  SubmitAddOne(context, input, output, CudaCapturePolicy::FORBIDDEN);
  EXPECT_EQ(test::Download<float>(context, output), (std::vector<float>{5.0F}));
}

TEST(CudaKernelLaunchTest, ReplaysCaptureSafeExternalKernel) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{4}, std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F});
  auto output = Empty(context, Shape{4}, DType::FLOAT32);

  SubmitAddOne(context, input, output, CudaCapturePolicy::SAFE);
  context.Synchronize();
  auto capture = context.BeginCapture(GraphCaptureOptions{.name_ = "external kernel"});
  SubmitAddOne(context, input, output, CudaCapturePolicy::SAFE);
  auto graph = capture.Finish();
  graph.Launch(context);
  graph.Launch(context);
  context.Synchronize();

  EXPECT_EQ(graph.GetLaunchCount(), 2);
  EXPECT_EQ(test::Download<float>(context, output), (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}));
}

TEST(CudaKernelLaunchTest, MovedFromCapturedGraphFailsDeterministically) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{1}, std::vector<float>{1.0F});
  auto output = Empty(context, Shape{1}, DType::FLOAT32);

  SubmitAddOne(context, input, output, CudaCapturePolicy::SAFE);
  context.Synchronize();
  auto capture = context.BeginCapture();
  SubmitAddOne(context, input, output, CudaCapturePolicy::SAFE);
  auto graph = capture.Finish();
  auto moved_graph = std::move(graph);

  // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
  EXPECT_THROW(static_cast<void>(graph.GetDevice()), InvalidArgumentError);
  // NOLINTNEXTLINE(bugprone-use-after-move, clang-analyzer-cplusplus.Move): moved-from is the contract under test.
  EXPECT_THROW(graph.Launch(context), InvalidArgumentError);
  moved_graph.Launch(context);
  context.Synchronize();
}

TEST(CudaKernelLaunchTest, RejectsDefaultPolicyDuringCapture) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = Empty(context, Shape{1}, DType::FLOAT32);
  auto output = Empty(context, Shape{1}, DType::FLOAT32);
  auto capture = context.BeginCapture();
  EXPECT_THROW(SubmitAddOne(context, input, output, CudaCapturePolicy::FORBIDDEN), CaptureError);
  capture.Abort();
}

TEST(CudaKernelLaunchTest, RecordsStorageUsageAcrossExplicitlyOrderedStreams) {
  test::RuntimeSession session;
  auto &producer = session.GetContext();
  auto consumer = session.GetRuntime().CreateExecutionContext(Device{0});
  auto input = test::Upload(producer, Shape{4}, std::vector<float>{1, 2, 3, 4});
  auto intermediate = Empty(producer, Shape{4}, DType::FLOAT32);
  auto output = Empty(consumer, Shape{4}, DType::FLOAT32);

  SubmitAddOne(producer, input, intermediate, CudaCapturePolicy::FORBIDDEN);
  const auto ready = producer.RecordEvent();
  consumer.Wait(ready);
  SubmitAddOne(consumer, intermediate, output, CudaCapturePolicy::FORBIDDEN);
  intermediate = Empty(producer, Shape{0}, DType::FLOAT32);

  EXPECT_EQ(test::Download<float>(consumer, output), (std::vector<float>{3, 4, 5, 6}));
}

}  // namespace ttl
