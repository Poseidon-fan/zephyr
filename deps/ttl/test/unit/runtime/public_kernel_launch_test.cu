#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/runtime/device_error.cuh"
#include "ttl/runtime/graph.hpp"
#include "ttl/runtime/kernel_launch.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::test {
namespace {

__global__ void AddOneKernel(const float *input, float *output, size_t count) {
  const size_t index = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index < count) {
    output[index] = input[index] + 1.0F;
  }
}

__global__ void ReportErrorKernel(CudaDeviceErrorContext context) {
  ReportCudaDeviceError(context, CudaDeviceErrorCode::USER_DEFINED, 7, 42, 9);
}

__global__ void MaybeReportErrorKernel(CudaDeviceErrorContext context, bool report) {
  if (report) {
    ReportCudaDeviceError(context, CudaDeviceErrorCode::USER_DEFINED, 11, 73, 5);
  }
}

TEST_F(SingleDeviceTest, CheckedLaunchProvidesTypedPointersStreamAndAlignedWorkspace) {
  ExecutionContext launch_context =
      GetRuntime().CreateExecutionContext(GetDevice(), {.max_auxiliary_stream_count_ = 1});
  Tensor input = FloatingTensorFromValues(launch_context, Shape{4}, DType::FLOAT32, {1, 2, 3, 4});
  Tensor output = Empty(launch_context, Shape{4}, DType::FLOAT32);
  const std::array<Tensor, 1> inputs{input};
  const std::array<Tensor *, 1> outputs{&output};
  const std::array<CudaWorkspaceRequest, 1> auxiliary_workspaces{{{.size_bytes_ = 256, .alignment_ = 256}}};
  const CudaKernelLaunchOptions options{
      .workspace_ = {.size_bytes_ = 1024, .alignment_ = 256},
      .auxiliary_stream_count_ = 1,
      .auxiliary_workspaces_ = auxiliary_workspaces,
      .capture_policy_ = CudaCapturePolicy::SAFE,
  };

  SubmitCudaKernel(
      launch_context, "test_add_one", inputs, outputs,
      [&](CudaKernelLaunch &launch) {
        EXPECT_NE(launch.GetStream(), nullptr);
        EXPECT_FALSE(launch.IsCapturing());
        ASSERT_NE(launch.GetWorkspace().data_, nullptr);
        EXPECT_EQ(launch.GetWorkspace().size_bytes_, 1024U);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(launch.GetWorkspace().data_) % 256U, 0U);
        ASSERT_EQ(launch.GetAuxiliaryStreamCount(), 1U);
        EXPECT_NE(launch.GetAuxiliaryStream(0), nullptr);
        EXPECT_EQ(launch.GetAuxiliaryWorkspace(0).size_bytes_, 256U);
        EXPECT_THROW(static_cast<void>(launch.GetAuxiliaryStream(1)), InvalidArgumentError);
        EXPECT_THROW(static_cast<void>(launch.GetInputDataAs<int32_t>(input)), InvalidArgumentError);

        AddOneKernel<<<1, 32, 0, launch.GetStream()>>>(launch.GetInputDataAs<float>(input),
                                                       launch.GetOutputDataAs<float>(output), 4);
      },
      options);

  launch_context.Synchronize();
  ExpectFloatValues(launch_context, output, {2, 3, 4, 5});
}

TEST_F(SingleDeviceTest, DeviceSemanticErrorIsStickyAndSurfacedAtExplicitBoundary) {
  const std::array<Tensor, 0> inputs{};
  const std::array<Tensor *, 0> outputs{};
  SubmitCudaKernel(GetContext(), "test_device_error", inputs, outputs, [](CudaKernelLaunch &launch) {
    ReportErrorKernel<<<1, 1, 0, launch.GetStream()>>>(launch.GetDeviceErrorContext(DType::INT64, DType::INT32));
  });

  try {
    GetContext().CheckAsyncErrors();
    FAIL() << "expected a DeviceError";
  } catch (const DeviceError &error) {
    EXPECT_EQ(error.GetCode(), ErrorCode::ASYNC_EXECUTION);
    EXPECT_NE(error.GetMessage().find("external CUDA operation"), std::string_view::npos);
    EXPECT_NE(error.GetMessage().find("linear index 7"), std::string_view::npos);
    EXPECT_NE(error.GetMessage().find("value bits 42"), std::string_view::npos);
  }

  EXPECT_NO_THROW(GetContext().Synchronize());
}

TEST_F(SingleDeviceTest, CapturedDeviceSemanticErrorSurfacesAfterGraphReplay) {
  const std::array<Tensor, 0> inputs{};
  const std::array<Tensor *, 0> outputs{};
  const auto submit = [&](bool report, CudaCapturePolicy capture_policy) {
    SubmitCudaKernel(GetContext(), "captured_device_error", inputs, outputs,
                     [report](CudaKernelLaunch &launch) {
                       MaybeReportErrorKernel<<<1, 1, 0, launch.GetStream()>>>(
                           launch.GetDeviceErrorContext(DType::INT64, DType::INT32), report);
                     },
                     {.capture_policy_ = capture_policy});
  };

  submit(false, CudaCapturePolicy::FORBIDDEN);
  GetContext().Synchronize();
  CaptureSession capture = GetContext().BeginCapture({.name_ = "captured device error"});
  submit(true, CudaCapturePolicy::SAFE);
  CapturedGraph graph = capture.Finish();
  graph.Launch(GetContext());

  try {
    GetContext().CheckAsyncErrors();
    FAIL() << "expected a DeviceError after graph replay";
  } catch (const DeviceError &error) {
    EXPECT_NE(error.GetMessage().find("linear index 11"), std::string_view::npos);
    EXPECT_NE(error.GetMessage().find("value bits 73"), std::string_view::npos);
  }
  EXPECT_NO_THROW(GetContext().Synchronize());
}

TEST_F(SingleDeviceTest, RejectsInvalidWorkspaceAndLaneRequestsBeforeCallingCallback) {
  const std::array<Tensor, 0> inputs{};
  const std::array<Tensor *, 0> outputs{};
  bool called = false;
  EXPECT_THROW(SubmitCudaKernel(
                   GetContext(), "bad_workspace", inputs, outputs, [&](CudaKernelLaunch &) { called = true; },
                   CudaKernelLaunchOptions{.workspace_ = {.size_bytes_ = 16, .alignment_ = 3}}),
               InvalidArgumentError);
  EXPECT_FALSE(called);

  EXPECT_THROW(SubmitCudaKernel(
                   GetContext(), "too_many_lanes", inputs, outputs, [&](CudaKernelLaunch &) { called = true; },
                   CudaKernelLaunchOptions{.auxiliary_stream_count_ = 1}),
               InvalidArgumentError);
  EXPECT_FALSE(called);
}

}  // namespace
}  // namespace ttl::test
