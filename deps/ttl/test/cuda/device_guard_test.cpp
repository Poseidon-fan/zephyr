#include "ttl/internal/device_guard.hpp"

#include <cstddef>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/cuda_api.hpp"

namespace ttl::internal {
namespace {

struct FakeCudaState final {
  int current_device_{0};
  cudaError_t get_status_{cudaSuccess};
  cudaError_t set_status_{cudaSuccess};
  int get_count_{0};
  int set_count_{0};
  int last_set_device_{-1};
};

thread_local FakeCudaState fake_cuda_state;

auto FakeGetDevice(int *device) -> cudaError_t {
  fake_cuda_state.get_count_++;
  if (fake_cuda_state.get_status_ == cudaSuccess) {
    *device = fake_cuda_state.current_device_;
  }
  return fake_cuda_state.get_status_;
}

auto FakeSetDevice(int device) -> cudaError_t {
  fake_cuda_state.set_count_++;
  fake_cuda_state.last_set_device_ = device;
  if (fake_cuda_state.set_status_ == cudaSuccess) {
    fake_cuda_state.current_device_ = device;
  }
  return fake_cuda_state.set_status_;
}

[[nodiscard]] auto MakeFakeCudaApi() -> CudaApi {
  auto cuda_api = GetCudaApi();
  cuda_api.get_device_ = FakeGetDevice;
  cuda_api.set_device_ = FakeSetDevice;
  return cuda_api;
}

const CudaApi FAKE_CUDA_API = MakeFakeCudaApi();

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override {
    record_.emplace(std::move(error));
    report_count_++;
  }

  [[nodiscard]] auto GetRecord() const noexcept -> const std::optional<ErrorRecord> & { return record_; }
  [[nodiscard]] auto GetReportCount() const noexcept -> size_t { return report_count_; }

 private:
  std::optional<ErrorRecord> record_;
  size_t report_count_{0};
};

class DeviceGuardTest : public testing::Test {
 protected:
  void SetUp() override { fake_cuda_state = FakeCudaState{}; }

 private:
  ScopedCudaApiOverride cuda_api_override_{FAKE_CUDA_API};
};

TEST_F(DeviceGuardTest, DoesNotSetOrRestoreWhenTargetIsAlreadyCurrent) {
  fake_cuda_state.current_device_ = 2;
  RecordingErrorSink error_sink;

  {
    DeviceGuard guard{Device{2}, error_sink};
    EXPECT_EQ(fake_cuda_state.current_device_, 2);
    EXPECT_EQ(fake_cuda_state.get_count_, 1);
    EXPECT_EQ(fake_cuda_state.set_count_, 0);
  }

  EXPECT_EQ(fake_cuda_state.set_count_, 0);
  EXPECT_EQ(error_sink.GetReportCount(), 0);
}

TEST_F(DeviceGuardTest, SwitchesToTargetAndRestoresPreviousDevice) {
  fake_cuda_state.current_device_ = 1;
  RecordingErrorSink error_sink;

  {
    DeviceGuard guard{Device{3}, error_sink};
    EXPECT_EQ(fake_cuda_state.current_device_, 3);
    EXPECT_EQ(fake_cuda_state.set_count_, 1);
  }

  EXPECT_EQ(fake_cuda_state.current_device_, 1);
  EXPECT_EQ(fake_cuda_state.set_count_, 2);
  EXPECT_EQ(fake_cuda_state.last_set_device_, 1);
  EXPECT_EQ(error_sink.GetReportCount(), 0);
}

TEST_F(DeviceGuardTest, GetDeviceFailureThrowsAtConstructionCallSite) {
  fake_cuda_state.get_status_ = cudaErrorInitializationError;
  RecordingErrorSink error_sink;
  const auto location = std::source_location::current();

  try {
    DeviceGuard guard{Device{1}, error_sink, location};
    FAIL() << "DeviceGuard did not throw";
  } catch (const CudaError &error) {
    EXPECT_NE(error.GetMessage().find("cudaGetDevice"), std::string_view::npos);
    EXPECT_EQ(error.GetLocation().line(), location.line());
  }

  EXPECT_EQ(fake_cuda_state.set_count_, 0);
  EXPECT_EQ(error_sink.GetReportCount(), 0);
}

TEST_F(DeviceGuardTest, SetDeviceFailureDoesNotAttemptRestore) {
  fake_cuda_state.current_device_ = 0;
  fake_cuda_state.set_status_ = cudaErrorInvalidDevice;
  RecordingErrorSink error_sink;

  EXPECT_THROW((DeviceGuard{Device{4}, error_sink}), CudaError);

  EXPECT_EQ(fake_cuda_state.current_device_, 0);
  EXPECT_EQ(fake_cuda_state.set_count_, 1);
  EXPECT_EQ(error_sink.GetReportCount(), 0);
}

TEST_F(DeviceGuardTest, RestoreFailureIsReportedWithoutEscapingDestructor) {
  fake_cuda_state.current_device_ = 0;
  RecordingErrorSink error_sink;
  const auto location = std::source_location::current();

  {
    DeviceGuard guard{Device{2}, error_sink, location};
    ASSERT_EQ(fake_cuda_state.current_device_, 2);
    fake_cuda_state.set_status_ = cudaErrorInvalidDevice;
  }

  EXPECT_EQ(fake_cuda_state.current_device_, 2);
  EXPECT_EQ(fake_cuda_state.set_count_, 2);
  ASSERT_TRUE(error_sink.GetRecord().has_value());
  const auto &record = *error_sink.GetRecord();
  EXPECT_EQ(record.code_, ErrorCode::CUDA);
  EXPECT_EQ(record.device_, Device{2});
  EXPECT_FALSE(record.stream_id_.has_value());
  EXPECT_NE(record.message_.find("restore previous device"), std::string::npos);
  EXPECT_EQ(record.location_.line(), location.line());
}

TEST_F(DeviceGuardTest, CleanupGuardSwitchesAndRestoresWithoutThrowing) {
  fake_cuda_state.current_device_ = 1;
  RecordingErrorSink error_sink;
  const ErrorReportContext context{
      .location_ = std::source_location::current(),
      .device_ = Device{2},
      .stream_id_ = std::nullopt,
  };

  {
    CleanupDeviceGuard guard{Device{2}, error_sink, context, "destroy test resource",
                             "restore after test resource destruction"};
    EXPECT_TRUE(guard);
    EXPECT_EQ(fake_cuda_state.current_device_, 2);
  }

  EXPECT_EQ(fake_cuda_state.current_device_, 1);
  EXPECT_EQ(fake_cuda_state.set_count_, 2);
  EXPECT_EQ(error_sink.GetReportCount(), 0);
}

TEST_F(DeviceGuardTest, CleanupGuardReportsGetFailureAndRemainsInactive) {
  fake_cuda_state.get_status_ = cudaErrorInitializationError;
  RecordingErrorSink error_sink;
  const ErrorReportContext context{
      .location_ = std::source_location::current(),
      .device_ = Device{2},
      .stream_id_ = std::nullopt,
  };

  CleanupDeviceGuard guard{Device{2}, error_sink, context, "destroy test resource",
                           "restore after test resource destruction"};

  EXPECT_FALSE(guard);
  EXPECT_EQ(fake_cuda_state.set_count_, 0);
  ASSERT_TRUE(error_sink.GetRecord().has_value());
  EXPECT_NE(error_sink.GetRecord()->message_.find("cudaGetDevice (destroy test resource)"), std::string::npos);
}

TEST_F(DeviceGuardTest, CleanupGuardReportsSetFailureAndRemainsInactive) {
  fake_cuda_state.current_device_ = 0;
  fake_cuda_state.set_status_ = cudaErrorInvalidDevice;
  RecordingErrorSink error_sink;
  const ErrorReportContext context{
      .location_ = std::source_location::current(),
      .device_ = Device{2},
      .stream_id_ = std::nullopt,
  };

  CleanupDeviceGuard guard{Device{2}, error_sink, context, "destroy test resource",
                           "restore after test resource destruction"};

  EXPECT_FALSE(guard);
  EXPECT_EQ(fake_cuda_state.set_count_, 1);
  ASSERT_TRUE(error_sink.GetRecord().has_value());
  EXPECT_NE(error_sink.GetRecord()->message_.find("cudaSetDevice (destroy test resource)"), std::string::npos);
}

}  // namespace
}  // namespace ttl::internal
