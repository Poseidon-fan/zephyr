#include "ttl/internal/device_properties.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/device_properties.hpp"
#include "ttl/error.hpp"
#include "ttl/internal/cuda_api.hpp"

namespace ttl::internal {
namespace {

using testing::HasSubstr;

struct FakeCudaState final {
  int device_count_{2};
  cudaDeviceProp properties_{};
  cudaError_t get_device_count_status_{cudaSuccess};
  cudaError_t get_device_properties_status_{cudaSuccess};
  int get_device_count_call_count_{0};
  int get_device_properties_call_count_{0};
  int get_device_call_count_{0};
  int set_device_call_count_{0};
  int last_queried_device_{-1};
};

thread_local FakeCudaState fake_cuda_state;

auto FakeGetDeviceCount(int *device_count) -> cudaError_t {
  fake_cuda_state.get_device_count_call_count_++;
  if (fake_cuda_state.get_device_count_status_ == cudaSuccess) {
    *device_count = fake_cuda_state.device_count_;
  }
  return fake_cuda_state.get_device_count_status_;
}

auto FakeGetDeviceProperties(cudaDeviceProp *properties, int device) -> cudaError_t {
  fake_cuda_state.get_device_properties_call_count_++;
  fake_cuda_state.last_queried_device_ = device;
  if (fake_cuda_state.get_device_properties_status_ == cudaSuccess) {
    *properties = fake_cuda_state.properties_;
  }
  return fake_cuda_state.get_device_properties_status_;
}

auto FakeGetDevice(int *device) -> cudaError_t {
  fake_cuda_state.get_device_call_count_++;
  *device = 0;
  return cudaSuccess;
}

auto FakeSetDevice(int /* device */) -> cudaError_t {
  fake_cuda_state.set_device_call_count_++;
  return cudaSuccess;
}

const CudaApi FAKE_CUDA_API{
    .get_device_count_ = FakeGetDeviceCount,
    .get_device_properties_ = FakeGetDeviceProperties,
    .get_device_ = FakeGetDevice,
    .set_device_ = FakeSetDevice,
    .get_stream_priority_range_ = cudaDeviceGetStreamPriorityRange,
    .create_stream_with_priority_ = cudaStreamCreateWithPriority,
    .destroy_stream_ = cudaStreamDestroy,
};

[[nodiscard]] auto MakeSupportedProperties() -> cudaDeviceProp {
  cudaDeviceProp properties{};
  constexpr std::string_view DEVICE_NAME{"NVIDIA Test GPU"};
  std::copy(DEVICE_NAME.begin(), DEVICE_NAME.end(), properties.name);
  properties.warpSize = 32;
  properties.major = 8;
  properties.minor = 0;
  properties.multiProcessorCount = 108;
  properties.computeMode = cudaComputeModeDefault;
  properties.sharedMemPerBlockOptin = size_t{163} << 10;
  properties.memoryPoolsSupported = 1;
  properties.clusterLaunch = 0;
  return properties;
}

class DevicePropertiesTest : public testing::Test {
 protected:
  void SetUp() override {
    fake_cuda_state = FakeCudaState{};
    fake_cuda_state.properties_ = MakeSupportedProperties();
  }
};

TEST(ComputeCapabilityTest, EncodesSmVersionWithoutNarrowing) {
  EXPECT_EQ((ComputeCapability{8, 0}.GetSmVersion()), 80);
  EXPECT_EQ((ComputeCapability{9, 0}.GetSmVersion()), 90);
  EXPECT_EQ((ComputeCapability{std::numeric_limits<int32_t>::max(), 9}.GetSmVersion()), 21474836479LL);
}

TEST(ComputeCapabilityTest, ComparesMajorBeforeMinor) {
  EXPECT_LT((ComputeCapability{8, 0}), (ComputeCapability{8, 6}));
  EXPECT_LT((ComputeCapability{8, 9}), (ComputeCapability{9, 0}));
  EXPECT_EQ((ComputeCapability{9, 0}), (ComputeCapability{9, 0}));
}

TEST_F(DevicePropertiesTest, QueriesAndMapsValidatedPropertiesWithoutChangingCurrentDevice) {
  fake_cuda_state.properties_.major = 12;
  fake_cuda_state.properties_.minor = 0;
  fake_cuda_state.properties_.clusterLaunch = 1;

  const auto properties = QueryDeviceProperties(Device{1}, FAKE_CUDA_API);

  EXPECT_EQ(properties.device_, Device{1});
  EXPECT_EQ(properties.name_, "NVIDIA Test GPU");
  EXPECT_EQ(properties.compute_capability_, (ComputeCapability{12, 0}));
  EXPECT_EQ(properties.multiprocessor_count_, 108);
  EXPECT_EQ(properties.max_dynamic_shared_memory_per_block_bytes_, size_t{163} << 10);
  EXPECT_TRUE(properties.supports_cluster_launch_);
  EXPECT_EQ(fake_cuda_state.get_device_count_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.get_device_properties_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.last_queried_device_, 1);
  EXPECT_EQ(fake_cuda_state.get_device_call_count_, 0);
  EXPECT_EQ(fake_cuda_state.set_device_call_count_, 0);
}

TEST_F(DevicePropertiesTest, AcceptsNonNullTerminatedCudaDeviceName) {
  std::fill(std::begin(fake_cuda_state.properties_.name), std::end(fake_cuda_state.properties_.name), 'x');

  const auto properties = QueryDeviceProperties(Device{0}, FAKE_CUDA_API);

  EXPECT_EQ(properties.name_, std::string(sizeof(fake_cuda_state.properties_.name), 'x'));
}

TEST_F(DevicePropertiesTest, ReportsDeviceCountFailureAtCallSite) {
  fake_cuda_state.get_device_count_status_ = cudaErrorInitializationError;
  const auto location = std::source_location::current();

  try {
    static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API, location));
    FAIL() << "QueryDeviceProperties did not throw";
  } catch (const CudaError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("cudaGetDeviceCount"));
    EXPECT_EQ(error.GetLocation().line(), location.line());
    return;
  }

  FAIL() << "QueryDeviceProperties threw the wrong exception type";
}

TEST_F(DevicePropertiesTest, RejectsNegativeDeviceCountAsInternalError) {
  fake_cuda_state.device_count_ = -1;

  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API)), InternalError);
  EXPECT_EQ(fake_cuda_state.get_device_properties_call_count_, 0);
}

TEST_F(DevicePropertiesTest, RejectsOrdinalOutsideVisibleDeviceRange) {
  fake_cuda_state.device_count_ = 1;
  const auto location = std::source_location::current();

  try {
    static_cast<void>(QueryDeviceProperties(Device{1}, FAKE_CUDA_API, location));
    FAIL() << "QueryDeviceProperties did not throw";
  } catch (const InvalidArgumentError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("device ordinal 1"));
    EXPECT_THAT(error.GetMessage(), HasSubstr("1 visible CUDA device"));
    EXPECT_EQ(error.GetLocation().line(), location.line());
    EXPECT_EQ(fake_cuda_state.get_device_properties_call_count_, 0);
    return;
  }

  FAIL() << "QueryDeviceProperties threw the wrong exception type";
}

TEST_F(DevicePropertiesTest, ReportsDevicePropertiesFailure) {
  fake_cuda_state.get_device_properties_status_ = cudaErrorInvalidDevice;

  try {
    static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API));
    FAIL() << "QueryDeviceProperties did not throw";
  } catch (const CudaError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("cudaGetDeviceProperties"));
    return;
  }

  FAIL() << "QueryDeviceProperties threw the wrong exception type";
}

TEST_F(DevicePropertiesTest, RejectsDevicesBelowSm80) {
  fake_cuda_state.properties_.major = 7;
  fake_cuda_state.properties_.minor = 9;

  try {
    static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API));
    FAIL() << "QueryDeviceProperties did not throw";
  } catch (const NotSupportedError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("7.9"));
    EXPECT_THAT(error.GetMessage(), HasSubstr("required 8.0"));
    return;
  }

  FAIL() << "QueryDeviceProperties threw the wrong exception type";
}

TEST_F(DevicePropertiesTest, RejectsInvalidComputeCapabilityFromCuda) {
  fake_cuda_state.properties_.minor = 10;

  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API)), InternalError);
}

TEST_F(DevicePropertiesTest, RejectsUnsupportedExecutionCapabilities) {
  fake_cuda_state.properties_.warpSize = 16;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API)), NotSupportedError);

  fake_cuda_state.properties_ = MakeSupportedProperties();
  fake_cuda_state.properties_.memoryPoolsSupported = 0;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API)), NotSupportedError);

  fake_cuda_state.properties_ = MakeSupportedProperties();
  fake_cuda_state.properties_.computeMode = cudaComputeModeProhibited;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API)), NotSupportedError);
}

TEST_F(DevicePropertiesTest, RejectsInvalidLaunchResourceProperties) {
  fake_cuda_state.properties_.multiProcessorCount = 0;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API)), InternalError);

  fake_cuda_state.properties_ = MakeSupportedProperties();
  fake_cuda_state.properties_.sharedMemPerBlockOptin = 0;
  EXPECT_THROW(static_cast<void>(QueryDeviceProperties(Device{0}, FAKE_CUDA_API)), InternalError);
}

}  // namespace
}  // namespace ttl::internal
