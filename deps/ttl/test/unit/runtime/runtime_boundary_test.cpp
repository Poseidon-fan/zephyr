#include <cstring>
#include <source_location>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "ttl/common/error.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/device_properties.hpp"
#include "ttl/internal/runtime/error_report.hpp"

namespace ttl::internal {
namespace {

int g_device_count = 1;
cudaError_t g_properties_status = cudaSuccess;
cudaDeviceProp g_properties{};

auto FakeGetDeviceCount(int *count) -> cudaError_t {
  *count = g_device_count;
  return cudaSuccess;
}

auto FakeGetDeviceProperties(cudaDeviceProp *properties, int ordinal) -> cudaError_t {
  static_cast<void>(ordinal);
  if (g_properties_status != cudaSuccess) {
    return g_properties_status;
  }
  *properties = g_properties;
  return cudaSuccess;
}

void SetValidProperties() {
  g_device_count = 1;
  g_properties_status = cudaSuccess;
  g_properties = cudaDeviceProp{};
  std::strncpy(g_properties.name, "test-a100", sizeof(g_properties.name) - 1);
  g_properties.major = 8;
  g_properties.minor = 0;
  g_properties.warpSize = 32;
  g_properties.memoryPoolsSupported = 1;

  g_properties.multiProcessorCount = 108;
  g_properties.sharedMemPerBlockOptin = 98304;
  g_properties.clusterLaunch = 1;
}

auto MakeApiOverride() -> CudaApi {
  auto api = GetCudaApi();
  api.get_device_count_ = FakeGetDeviceCount;
  api.get_device_properties_ = FakeGetDeviceProperties;
  return api;
}

class DevicePropertiesBoundaryTest : public ::testing::Test {
 protected:
  void SetUp() override { SetValidProperties(); }
};

TEST_F(DevicePropertiesBoundaryTest, ConvertsValidNativeProperties) {
  const auto api = MakeApiOverride();
  const ScopedCudaApiOverride override{api};
  const auto properties = QueryDeviceProperties(Device{0}, std::source_location::current());
  EXPECT_EQ(properties.device_, Device{0});
  EXPECT_EQ(properties.name_, "test-a100");
  EXPECT_EQ(properties.compute_capability_, (ComputeCapability{.major_ = 8, .minor_ = 0}));
  EXPECT_EQ(properties.multiprocessor_count_, 108);
  EXPECT_EQ(properties.max_dynamic_shared_memory_per_block_bytes_, 98304);
  EXPECT_TRUE(properties.supports_cluster_launch_);
}

TEST_F(DevicePropertiesBoundaryTest, RejectsOutOfRangeOrdinalsWithCorrectPluralization) {
  g_device_count = 1;
  const auto api = MakeApiOverride();
  const ScopedCudaApiOverride override{api};
  try {
    static_cast<void>(QueryDeviceProperties(Device{1}, std::source_location::current()));
    FAIL();
  } catch (const InvalidArgumentError &error) {
    EXPECT_NE(std::strstr(error.what(), "1 visible CUDA device"), nullptr);
  }

  g_device_count = 2;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{2}, std::source_location::current())),
               InvalidArgumentError);
}

TEST_F(DevicePropertiesBoundaryTest, RejectsCudaPropertyQueryFailure) {
  g_properties_status = cudaErrorInvalidDevice;
  const auto api = MakeApiOverride();
  const ScopedCudaApiOverride override{api};
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, std::source_location::current())), CudaError);
}

TEST_F(DevicePropertiesBoundaryTest, RejectsInvalidNativeComputeCapability) {
  const auto api = MakeApiOverride();
  const ScopedCudaApiOverride override{api};
  g_properties.major = -1;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, std::source_location::current())), InternalError);
  g_properties.major = 8;
  g_properties.minor = 10;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, std::source_location::current())), InternalError);
}

TEST_F(DevicePropertiesBoundaryTest, RejectsUnsupportedHardwareConfiguration) {
  const auto api = MakeApiOverride();
  const ScopedCudaApiOverride override{api};
  g_properties.major = 7;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, std::source_location::current())), NotSupportedError);
  SetValidProperties();
  g_properties.warpSize = 64;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, std::source_location::current())), NotSupportedError);
  SetValidProperties();
  g_properties.memoryPoolsSupported = 0;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, std::source_location::current())), NotSupportedError);
}

TEST_F(DevicePropertiesBoundaryTest, RejectsInvalidResourceLimits) {
  const auto api = MakeApiOverride();
  const ScopedCudaApiOverride override{api};
  g_properties.multiProcessorCount = 0;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, std::source_location::current())), InternalError);
  SetValidProperties();
  g_properties.sharedMemPerBlockOptin = 0;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, std::source_location::current())), InternalError);
}

TEST(ErrorReportTest, PreservesCodeMessageAndOptionalExecutionContext) {
  test::RecordingErrorSink sink;
  const auto location = std::source_location::current();
  ReportErrorNoexcept(ErrorCode::INTERNAL, "background failure", sink,
                      ErrorReportContext{.location_ = location, .device_ = Device{2}, .stream_id_ = 41});
  const auto records = sink.GetRecords();
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0].code_, ErrorCode::INTERNAL);
  EXPECT_EQ(records[0].message_, "background failure");
  ASSERT_TRUE(records[0].device_.has_value());
  EXPECT_EQ(*records[0].device_, Device{2});
  ASSERT_TRUE(records[0].stream_id_.has_value());
  EXPECT_EQ(*records[0].stream_id_, 41);
  EXPECT_EQ(records[0].location_.line(), location.line());
}

TEST(ErrorReportTest, PreservesProcessWideContext) {
  test::RecordingErrorSink sink;
  ReportErrorNoexcept(
      ErrorCode::CUDA, "process-wide failure", sink,
      ErrorReportContext{
          .location_ = std::source_location::current(), .device_ = std::nullopt, .stream_id_ = std::nullopt});
  const auto records = sink.GetRecords();
  ASSERT_EQ(records.size(), 1);
  EXPECT_FALSE(records[0].device_.has_value());
  EXPECT_FALSE(records[0].stream_id_.has_value());
}

}  // namespace
}  // namespace ttl::internal
