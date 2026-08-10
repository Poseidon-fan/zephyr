#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "support/tensor_test_utils.hpp"
#include "ttl/common/error.hpp"
#include "ttl/runtime/device_error.cuh"
#include "ttl/runtime/generator.hpp"
#include "ttl/runtime/graph.hpp"
#include "ttl/runtime/kernel_launch.hpp"
#include "ttl/runtime/philox.cuh"
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

__global__ void DelayedAddOneKernel(const float *input, float *output, size_t count, uint64_t delay_cycles) {
  const auto start = static_cast<uint64_t>(clock64());
  while (static_cast<uint64_t>(clock64()) - start < delay_cycles) {
  }
  const auto index = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index < count) {
    output[index] = input[index] + 1.0F;
  }
}

__global__ void DelayedSetKernel(float *output, size_t index, float value, uint64_t delay_cycles) {
  const auto start = static_cast<uint64_t>(clock64());
  while (static_cast<uint64_t>(clock64()) - start < delay_cycles) {
  }
  if (threadIdx.x == 0) {
    output[index] = value;
  }
}

__global__ void ReportInvalidValueKernel(CudaDeviceErrorContext error_context) {
  if (threadIdx.x == 0) {
    ReportCudaDeviceError(error_context, CudaDeviceErrorCode::INVALID_VALUE, 7, 42, 11);
  }
}

__global__ void WritePhiloxKernel(CudaPhiloxReservation reservation, int64_t *output) {
  if (threadIdx.x == 0) {
    const auto result = GenerateCudaPhilox(reservation, 0);
    for (size_t index = 0; index < 4; ++index) {
      output[index] = static_cast<int64_t>(result.values_[index]);
    }
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
          .workspace_ = {.size_bytes_ = 1024, .alignment_ = 256},
          .capture_policy_ = capture_policy,
      });
}

void SubmitAddOneParallel(ExecutionContext &context, const Tensor &input, Tensor &output,
                          CudaCapturePolicy capture_policy) {
  const std::array inputs{input};
  const std::array outputs{&output};
  constexpr std::array auxiliary_workspaces{
      CudaWorkspaceRequest{.size_bytes_ = 512, .alignment_ = 128},
  };
  SubmitCudaKernel(
      context, "external parallel AddOne", inputs, outputs,
      [&](CudaKernelLaunch &launch) {
        const auto count = static_cast<size_t>(input.GetNumElements());
        const auto primary_count = count / 2;
        const auto auxiliary_count = count - primary_count;
        constexpr uint32_t block_size = 128;
        const auto *input_data = launch.GetInputDataAs<float>(input);
        auto *output_data = launch.GetOutputDataAs<float>(output);
        const auto primary_blocks = static_cast<uint32_t>((primary_count + block_size - 1) / block_size);
        const auto auxiliary_blocks = static_cast<uint32_t>((auxiliary_count + block_size - 1) / block_size);
        AddOneKernel<<<primary_blocks, block_size, 0, launch.GetStream()>>>(input_data, output_data, primary_count);
        AddOneKernel<<<auxiliary_blocks, block_size, 0, launch.GetAuxiliaryStream(0)>>>(
            input_data + primary_count, output_data + primary_count, auxiliary_count);
      },
      CudaKernelLaunchOptions{
          .workspace_ = {.size_bytes_ = 1024, .alignment_ = 256},
          .auxiliary_stream_count_ = 1,
          .auxiliary_workspaces_ = auxiliary_workspaces,
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

TEST(CudaKernelLaunchTest, RunsOneSubmissionAcrossPrimaryAndAuxiliaryStreams) {
  test::RuntimeSession session;
  auto context = session.GetRuntime().CreateExecutionContext(session.GetDevice(),
                                                             ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
  auto input = test::Upload(context, Shape{6}, std::vector<float>{-2.0F, -0.0F, 1.0F, 4.0F, 9.0F, 16.0F});
  auto output = Empty(context, Shape{6}, DType::FLOAT32);
  const std::array inputs{input};
  const std::array outputs{&output};
  constexpr std::array auxiliary_workspaces{
      CudaWorkspaceRequest{.size_bytes_ = 512, .alignment_ = 128},
  };

  SubmitCudaKernel(
      context, "parallel public API", inputs, outputs,
      [&](CudaKernelLaunch &launch) {
        EXPECT_EQ(launch.GetAuxiliaryStreamCount(), 1);
        EXPECT_NE(launch.GetStream(), launch.GetAuxiliaryStream(0));
        EXPECT_EQ(launch.GetWorkspace().size_bytes_, 1024);
        EXPECT_EQ(launch.GetAuxiliaryWorkspace(0).size_bytes_, 512);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(launch.GetWorkspace().data_) % 256, 0);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(launch.GetAuxiliaryWorkspace(0).data_) % 128, 0);
        EXPECT_THROW(static_cast<void>(launch.GetAuxiliaryStream(1)), InvalidArgumentError);
        EXPECT_THROW(static_cast<void>(launch.GetAuxiliaryWorkspace(1)), InvalidArgumentError);

        constexpr uint32_t block_size = 128;
        const auto *input_data = launch.GetInputDataAs<float>(input);
        auto *output_data = launch.GetOutputDataAs<float>(output);
        AddOneKernel<<<1, block_size, 0, launch.GetStream()>>>(input_data, output_data, 3);
        DelayedAddOneKernel<<<1, block_size, 0, launch.GetAuxiliaryStream(0)>>>(input_data + 3, output_data + 3, 3,
                                                                                5'000'000);
      },
      CudaKernelLaunchOptions{
          .workspace_ = {.size_bytes_ = 1024, .alignment_ = 256},
          .auxiliary_stream_count_ = 1,
          .auxiliary_workspaces_ = auxiliary_workspaces,
      });

  EXPECT_EQ(test::Download<float>(context, output), (std::vector<float>{-1.0F, 1.0F, 2.0F, 5.0F, 10.0F, 17.0F}));
}

TEST(CudaKernelLaunchTest, PublishesDependenciesBetweenPrimaryAndAuxiliaryStreams) {
  test::RuntimeSession session;
  auto context = session.GetRuntime().CreateExecutionContext(session.GetDevice(),
                                                             ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
  auto output = Empty(context, Shape{2}, DType::FLOAT32);
  const std::array<Tensor, 0> inputs{};
  const std::array outputs{&output};

  SubmitCudaKernel(
      context, "reusable lane dependencies", inputs, outputs,
      [&](CudaKernelLaunch &launch) {
        auto *output_data = launch.GetOutputDataAs<float>(output);
        DelayedSetKernel<<<1, 1, 0, launch.GetStream()>>>(output_data, 0, 41.0F, 5'000'000);
        launch.PublishPrimaryToAuxiliary();
        AddOneKernel<<<1, 1, 0, launch.GetAuxiliaryStream(0)>>>(output_data, output_data, 1);

        DelayedSetKernel<<<1, 1, 0, launch.GetAuxiliaryStream(0)>>>(output_data, 1, 41.0F, 5'000'000);
        launch.PublishAuxiliaryToPrimary();
        AddOneKernel<<<1, 1, 0, launch.GetStream()>>>(output_data + 1, output_data + 1, 1);
      },
      CudaKernelLaunchOptions{.auxiliary_stream_count_ = 1});

  EXPECT_EQ(test::Download<float>(context, output), (std::vector<float>{42.0F, 42.0F}));
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
                                CudaKernelLaunchOptions{.workspace_ = {.size_bytes_ = 1, .alignment_ = 3}}),
               InvalidArgumentError);
  EXPECT_THROW(SubmitCudaKernel(context, "invalid capture policy", inputs, outputs, no_op,
                                CudaKernelLaunchOptions{
                                    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): invalid input test.
                                    .capture_policy_ = static_cast<CudaCapturePolicy>(255),
                                }),
               InvalidArgumentError);

  EXPECT_THROW(SubmitCudaKernel(context, "unavailable auxiliary stream", inputs, outputs, no_op,
                                CudaKernelLaunchOptions{.auxiliary_stream_count_ = 1}),
               InvalidArgumentError);

  auto auxiliary_context = session.GetRuntime().CreateExecutionContext(
      session.GetDevice(), ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
  constexpr std::array invalid_auxiliary_workspaces{
      CudaWorkspaceRequest{.size_bytes_ = 1, .alignment_ = 3},
  };
  EXPECT_THROW(SubmitCudaKernel(auxiliary_context, "invalid auxiliary alignment", inputs, outputs, no_op,
                                CudaKernelLaunchOptions{
                                    .auxiliary_stream_count_ = 1,
                                    .auxiliary_workspaces_ = invalid_auxiliary_workspaces,
                                }),
               InvalidArgumentError);
  EXPECT_THROW(SubmitCudaKernel(auxiliary_context, "excess auxiliary workspaces", inputs, outputs, no_op,
                                CudaKernelLaunchOptions{
                                    .auxiliary_workspaces_ = invalid_auxiliary_workspaces,
                                }),
               InvalidArgumentError);
}

TEST(CudaKernelLaunchTest, PoisonsContextAfterLaunchError) {
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

  EXPECT_THROW(SubmitAddOne(context, input, output, CudaCapturePolicy::FORBIDDEN), InvalidArgumentError);
  context.Synchronize();
}

TEST(CudaKernelLaunchTest, ReportsExternalDeviceSemanticErrorsThroughContext) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  const std::array<Tensor, 0> inputs{};
  const std::array<Tensor *, 0> outputs{};

  SubmitCudaKernel(context, "external semantic error", inputs, outputs, [&](CudaKernelLaunch &launch) {
    const auto error_context = launch.GetDeviceErrorContext(DType::INT64, DType::INT64);
    ReportInvalidValueKernel<<<1, 1, 0, launch.GetStream()>>>(error_context);
  });
  EXPECT_THROW(context.CheckAsyncErrors(), DeviceError);

  auto input = test::Upload(context, Shape{1}, std::vector<float>{4.0F});
  auto output = Empty(context, Shape{1}, DType::FLOAT32);
  SubmitAddOne(context, input, output, CudaCapturePolicy::FORBIDDEN);
  EXPECT_EQ(test::Download<float>(context, output), (std::vector<float>{5.0F}));
}

TEST(CudaKernelLaunchTest, ReservesGraphSafePhiloxBlocksForExternalKernel) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  Generator generator{context, 1234};
  auto first = Empty(context, Shape{4}, DType::INT64);
  auto second = Empty(context, Shape{4}, DType::INT64);
  const std::array<Tensor, 0> inputs{};

  const auto submit = [&](Tensor &output) {
    const std::array outputs{&output};
    SubmitCudaKernel(context, "external Philox", inputs, outputs, [&](CudaKernelLaunch &launch) {
      const auto reservation = launch.ReservePhilox(generator, 1);
      WritePhiloxKernel<<<1, 1, 0, launch.GetStream()>>>(reservation, launch.GetOutputDataAs<int64_t>(output));
    });
  };

  submit(first);
  submit(second);
  const auto first_values = test::Download<int64_t>(context, first);
  const auto second_values = test::Download<int64_t>(context, second);
  EXPECT_NE(first_values, second_values);

  generator.SetSeed(context, 1234);
  submit(second);
  EXPECT_EQ(test::Download<int64_t>(context, second), first_values);
}

TEST(CudaKernelLaunchTest, PublishesPhiloxReservationToAuxiliaryStreams) {
  test::RuntimeSession session;
  auto context = session.GetRuntime().CreateExecutionContext(session.GetDevice(),
                                                             ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
  Generator generator{context, 1234};
  auto expected = Empty(context, Shape{4}, DType::INT64);
  auto actual = Empty(context, Shape{4}, DType::INT64);
  const std::array<Tensor, 0> inputs{};

  const auto submit = [&](Tensor &output, bool use_auxiliary) {
    const std::array outputs{&output};
    SubmitCudaKernel(
        context, "external auxiliary Philox", inputs, outputs,
        [&](CudaKernelLaunch &launch) {
          const auto reservation = launch.ReservePhilox(generator, 1);
          const auto stream = use_auxiliary ? launch.GetAuxiliaryStream(0) : launch.GetStream();
          WritePhiloxKernel<<<1, 1, 0, stream>>>(reservation, launch.GetOutputDataAs<int64_t>(output));
        },
        CudaKernelLaunchOptions{.auxiliary_stream_count_ = 1});
  };

  submit(expected, false);
  generator.SetSeed(context, 1234);
  submit(actual, true);
  EXPECT_EQ(test::Download<int64_t>(context, actual), test::Download<int64_t>(context, expected));
}

TEST(CudaKernelLaunchTest, CapturedPhiloxReservationReadsCurrentDeviceSeedOnReplay) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  Generator generator{context, 1234};
  auto output = Empty(context, Shape{4}, DType::INT64);
  const std::array<Tensor, 0> inputs{};
  const std::array outputs{&output};
  const auto submit = [&] {
    SubmitCudaKernel(
        context, "captured external Philox", inputs, outputs,
        [&](CudaKernelLaunch &launch) {
          const auto reservation = launch.ReservePhilox(generator, 1);
          WritePhiloxKernel<<<1, 1, 0, launch.GetStream()>>>(reservation, launch.GetOutputDataAs<int64_t>(output));
        },
        CudaKernelLaunchOptions{
            .workspace_ = {.size_bytes_ = sizeof(uint64_t), .alignment_ = alignof(uint64_t)},
            .capture_policy_ = CudaCapturePolicy::SAFE,
        });
  };

  submit();
  context.Synchronize();
  auto capture = context.BeginCapture(GraphCaptureOptions{.name_ = "external Philox seed"});
  submit();
  auto graph = capture.Finish();

  generator.SetSeed(context, 1234);
  graph.Launch(context);
  const auto first_seed_values = test::Download<int64_t>(context, output);
  generator.SetSeed(context, 5678);
  graph.Launch(context);
  const auto second_seed_values = test::Download<int64_t>(context, output);
  EXPECT_NE(second_seed_values, first_seed_values);

  generator.SetSeed(context, 1234);
  graph.Launch(context);
  EXPECT_EQ(test::Download<int64_t>(context, output), first_seed_values);
}

TEST(CudaKernelLaunchTest, FailsContextAndPreservesExceptionAfterPartialSubmission) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{1}, std::vector<float>{4.0F});
  auto output = Empty(context, Shape{1}, DType::FLOAT32);
  const std::array inputs{input};
  const std::array outputs{&output};

  EXPECT_THROW(SubmitCudaKernel(context, "throwing submission", inputs, outputs,
                                [&](CudaKernelLaunch &launch) {
                                  AddOneKernel<<<1, 1, 0, launch.GetStream()>>>(
                                      launch.GetInputDataAs<float>(input), launch.GetOutputDataAs<float>(output), 1);
                                  throw std::runtime_error{"submission callback failed"};
                                }),
               std::runtime_error);
  EXPECT_THROW(SubmitAddOne(context, input, output, CudaCapturePolicy::FORBIDDEN), InvalidArgumentError);
  context.Synchronize();
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

TEST(CudaKernelLaunchTest, ReplaysCaptureSafeExternalKernelAcrossAuxiliaryStream) {
  test::RuntimeSession session;
  auto context = session.GetRuntime().CreateExecutionContext(session.GetDevice(),
                                                             ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
  auto input = test::Upload(context, Shape{6}, std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  auto output = Empty(context, Shape{6}, DType::FLOAT32);

  SubmitAddOneParallel(context, input, output, CudaCapturePolicy::SAFE);
  context.Synchronize();
  auto capture = context.BeginCapture(GraphCaptureOptions{.name_ = "external parallel kernel"});
  SubmitAddOneParallel(context, input, output, CudaCapturePolicy::SAFE);
  auto graph = capture.Finish();
  graph.Launch(context);
  graph.Launch(context);
  context.Synchronize();

  EXPECT_EQ(graph.GetLaunchCount(), 2);
  EXPECT_EQ(test::Download<float>(context, output), (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));
}

TEST(CudaKernelLaunchTest, FailsContextAfterParallelCallbackException) {
  test::RuntimeSession session;
  auto context = session.GetRuntime().CreateExecutionContext(session.GetDevice(),
                                                             ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
  auto input = test::Upload(context, Shape{1}, std::vector<float>{4.0F});
  auto output = Empty(context, Shape{1}, DType::FLOAT32);
  const std::array inputs{input};
  const std::array outputs{&output};

  EXPECT_THROW(SubmitCudaKernel(
                   context, "throwing parallel submission", inputs, outputs,
                   [&](CudaKernelLaunch &launch) {
                     AddOneKernel<<<1, 1, 0, launch.GetAuxiliaryStream(0)>>>(launch.GetInputDataAs<float>(input),
                                                                             launch.GetOutputDataAs<float>(output), 1);
                     throw std::runtime_error{"parallel submission callback failed"};
                   },
                   CudaKernelLaunchOptions{.auxiliary_stream_count_ = 1}),
               std::runtime_error);
  EXPECT_THROW(SubmitAddOneParallel(context, input, output, CudaCapturePolicy::FORBIDDEN), InvalidArgumentError);
  context.Synchronize();
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
  auto consumer = session.GetRuntime().CreateExecutionContext(session.GetDevice());
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
