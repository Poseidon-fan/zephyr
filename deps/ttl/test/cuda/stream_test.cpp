#include "ttl/internal/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {
namespace {

using testing::HasSubstr;

int fake_stream_storage;

struct FakeCudaState final {
  int current_device_{0};
  int least_priority_{0};
  int greatest_priority_{-3};
  cudaStream_t stream_to_create_{nullptr};
  cudaError_t get_device_status_{cudaSuccess};
  cudaError_t get_priority_range_status_{cudaSuccess};
  cudaError_t create_stream_status_{cudaSuccess};
  cudaError_t destroy_stream_status_{cudaSuccess};
  std::optional<int> fail_set_device_;
  int get_device_call_count_{0};
  int set_device_call_count_{0};
  int get_priority_range_call_count_{0};
  int create_stream_call_count_{0};
  int destroy_stream_call_count_{0};
  int last_set_device_{-1};
  int last_priority_{0};
  unsigned int last_create_flags_{0};
  cudaStream_t last_destroyed_stream_{nullptr};
};

thread_local FakeCudaState fake_cuda_state;

auto FakeGetDevice(int *device) -> cudaError_t {
  fake_cuda_state.get_device_call_count_++;
  if (fake_cuda_state.get_device_status_ == cudaSuccess) {
    *device = fake_cuda_state.current_device_;
  }
  return fake_cuda_state.get_device_status_;
}

auto FakeSetDevice(int device) -> cudaError_t {
  fake_cuda_state.set_device_call_count_++;
  fake_cuda_state.last_set_device_ = device;
  if (fake_cuda_state.fail_set_device_ == device) {
    return cudaErrorInvalidDevice;
  }
  fake_cuda_state.current_device_ = device;
  return cudaSuccess;
}

auto FakeGetStreamPriorityRange(int *least_priority, int *greatest_priority) -> cudaError_t {
  fake_cuda_state.get_priority_range_call_count_++;
  if (fake_cuda_state.get_priority_range_status_ == cudaSuccess) {
    *least_priority = fake_cuda_state.least_priority_;
    *greatest_priority = fake_cuda_state.greatest_priority_;
  }
  return fake_cuda_state.get_priority_range_status_;
}

auto FakeCreateStreamWithPriority(cudaStream_t *stream, unsigned int flags, int priority) -> cudaError_t {
  fake_cuda_state.create_stream_call_count_++;
  fake_cuda_state.last_create_flags_ = flags;
  fake_cuda_state.last_priority_ = priority;
  if (fake_cuda_state.create_stream_status_ == cudaSuccess) {
    *stream = fake_cuda_state.stream_to_create_;
  }
  return fake_cuda_state.create_stream_status_;
}

auto FakeDestroyStream(cudaStream_t stream) -> cudaError_t {
  fake_cuda_state.destroy_stream_call_count_++;
  fake_cuda_state.last_destroyed_stream_ = stream;
  return fake_cuda_state.destroy_stream_status_;
}

const CudaApi FAKE_CUDA_API{
    .get_device_count_ = cudaGetDeviceCount,
    .get_device_properties_ = cudaGetDeviceProperties,
    .get_device_ = FakeGetDevice,
    .set_device_ = FakeSetDevice,
    .get_last_error_ = cudaGetLastError,
    .get_stream_priority_range_ = FakeGetStreamPriorityRange,
    .create_stream_with_priority_ = FakeCreateStreamWithPriority,
    .destroy_stream_ = FakeDestroyStream,
    .create_event_with_flags_ = cudaEventCreateWithFlags,
    .record_event_ = cudaEventRecord,
    .query_event_ = cudaEventQuery,
    .synchronize_event_ = cudaEventSynchronize,
    .destroy_event_ = cudaEventDestroy,
    .stream_wait_event_ = cudaStreamWaitEvent,
};

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

class StreamTest : public testing::Test {
 protected:
  void SetUp() override {
    fake_cuda_state = FakeCudaState{};
    fake_cuda_state.stream_to_create_ = reinterpret_cast<cudaStream_t>(&fake_stream_storage);
  }

 private:
  ScopedCudaApiOverride cuda_api_override_{FAKE_CUDA_API};
};

TEST_F(StreamTest, CreatesNonBlockingOwnedStreamAndSharesItsIdentity) {
  fake_cuda_state.current_device_ = 1;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  uint64_t stream_id = 0;

  {
    const auto stream = StreamAccess::CreateOwned(Device{2}, -2, error_sink);
    const auto stream_copy = stream;
    stream_id = stream.GetId();

    EXPECT_EQ(stream.GetDevice(), Device{2});
    EXPECT_EQ(stream_copy.GetId(), stream_id);
    EXPECT_FALSE(stream.IsExternal());
    EXPECT_EQ(StreamAccess::GetNative(stream), fake_cuda_state.stream_to_create_);
    EXPECT_EQ(fake_cuda_state.current_device_, 1);
    EXPECT_EQ(fake_cuda_state.get_priority_range_call_count_, 1);
    EXPECT_EQ(fake_cuda_state.create_stream_call_count_, 1);
    EXPECT_EQ(fake_cuda_state.last_create_flags_, cudaStreamNonBlocking);
    EXPECT_EQ(fake_cuda_state.last_priority_, -2);
    EXPECT_EQ(fake_cuda_state.destroy_stream_call_count_, 0);
  }

  EXPECT_NE(stream_id, 0);
  EXPECT_EQ(fake_cuda_state.destroy_stream_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.last_destroyed_stream_, fake_cuda_state.stream_to_create_);
  EXPECT_EQ(fake_cuda_state.current_device_, 1);
  EXPECT_EQ(error_sink->GetReportCount(), 0);
}

TEST_F(StreamTest, AssignsDistinctProcessWideIds) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  fake_cuda_state.current_device_ = 0;

  const auto first = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
  const auto second = StreamAccess::CreateOwned(Device{0}, 0, error_sink);

  EXPECT_NE(first.GetId(), second.GetId());
}

TEST_F(StreamTest, KeepsCleanupErrorSinkAliveUntilTheLastStreamOwner) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  const std::weak_ptr<RecordingErrorSink> weak_error_sink = error_sink;

  {
    const auto stream = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
    error_sink.reset();
    EXPECT_FALSE(weak_error_sink.expired());
  }

  EXPECT_TRUE(weak_error_sink.expired());
  EXPECT_EQ(fake_cuda_state.destroy_stream_call_count_, 1);
}

TEST_F(StreamTest, KeepsNativeStreamAliveThroughInternalStateLease) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  std::shared_ptr<StreamState> state_lease;

  {
    const auto stream = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
    state_lease = StreamAccess::GetState(stream);
  }

  EXPECT_EQ(fake_cuda_state.destroy_stream_call_count_, 0);
  state_lease.reset();
  EXPECT_EQ(fake_cuda_state.destroy_stream_call_count_, 1);
}

TEST_F(StreamTest, RejectsPriorityOutsideDeviceRangeAtCallSite) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto location = std::source_location::current();

  try {
    static_cast<void>(StreamAccess::CreateOwned(Device{0}, -4, error_sink, location));
    FAIL() << "CreateOwned did not throw";
  } catch (const InvalidArgumentError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("priority -4"));
    EXPECT_THAT(error.GetMessage(), HasSubstr("[-3, 0]"));
    EXPECT_EQ(error.GetLocation().line(), location.line());
    EXPECT_EQ(fake_cuda_state.create_stream_call_count_, 0);
    return;
  }

  FAIL() << "CreateOwned threw the wrong exception type";
}

TEST_F(StreamTest, ReportsPriorityQueryAndCreationFailures) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  fake_cuda_state.get_priority_range_status_ = cudaErrorInitializationError;
  EXPECT_THROW(static_cast<void>(StreamAccess::CreateOwned(Device{0}, 0, error_sink)), CudaError);
  EXPECT_EQ(fake_cuda_state.create_stream_call_count_, 0);

  fake_cuda_state.get_priority_range_status_ = cudaSuccess;
  fake_cuda_state.create_stream_status_ = cudaErrorDevicesUnavailable;
  EXPECT_THROW(static_cast<void>(StreamAccess::CreateOwned(Device{0}, 0, error_sink)), CudaError);
}

TEST_F(StreamTest, RejectsInvalidSuccessfulCudaResults) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  fake_cuda_state.least_priority_ = -3;
  fake_cuda_state.greatest_priority_ = 0;
  EXPECT_THROW(static_cast<void>(StreamAccess::CreateOwned(Device{0}, 0, error_sink)), InternalError);
  EXPECT_EQ(fake_cuda_state.create_stream_call_count_, 0);

  fake_cuda_state.least_priority_ = 0;
  fake_cuda_state.greatest_priority_ = -3;
  fake_cuda_state.stream_to_create_ = nullptr;
  EXPECT_THROW(static_cast<void>(StreamAccess::CreateOwned(Device{0}, 0, error_sink)), InternalError);
}

TEST_F(StreamTest, RejectsNullErrorSinkBeforeCallingCuda) {
  EXPECT_THROW(static_cast<void>(StreamAccess::CreateOwned(Device{0}, 0, std::shared_ptr<ErrorSink>{})),
               InvalidArgumentError);
  EXPECT_EQ(fake_cuda_state.get_device_call_count_, 0);
  EXPECT_EQ(fake_cuda_state.create_stream_call_count_, 0);
}

TEST_F(StreamTest, ReportsDestroyFailureWithDeviceStreamAndCallSite) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto location = std::source_location::current();
  uint64_t stream_id = 0;

  {
    const auto stream = StreamAccess::CreateOwned(Device{0}, 0, error_sink, location);
    stream_id = stream.GetId();
    fake_cuda_state.destroy_stream_status_ = cudaErrorInvalidResourceHandle;
  }

  ASSERT_TRUE(error_sink->GetRecord().has_value());
  const auto &record = *error_sink->GetRecord();
  EXPECT_EQ(record.code_, ErrorCode::CUDA);
  EXPECT_EQ(record.device_, Device{0});
  EXPECT_EQ(record.stream_id_, stream_id);
  EXPECT_EQ(record.location_.line(), location.line());
  EXPECT_THAT(record.message_, HasSubstr("cudaStreamDestroy"));
}

TEST_F(StreamTest, DoesNotDestroyWhenCleanupCannotSelectTheOwningDevice) {
  fake_cuda_state.current_device_ = 1;
  const auto error_sink = std::make_shared<RecordingErrorSink>();

  {
    const auto stream = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
    fake_cuda_state.fail_set_device_ = 0;
  }

  EXPECT_EQ(fake_cuda_state.destroy_stream_call_count_, 0);
  ASSERT_TRUE(error_sink->GetRecord().has_value());
  EXPECT_THAT(error_sink->GetRecord()->message_, HasSubstr("cudaSetDevice (destroy stream)"));
}

TEST_F(StreamTest, DoesNotDestroyWhenCleanupCannotReadTheCurrentDevice) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();

  {
    const auto stream = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
    fake_cuda_state.get_device_status_ = cudaErrorInitializationError;
  }

  EXPECT_EQ(fake_cuda_state.destroy_stream_call_count_, 0);
  ASSERT_TRUE(error_sink->GetRecord().has_value());
  EXPECT_THAT(error_sink->GetRecord()->message_, HasSubstr("cudaGetDevice (destroy stream)"));
}

TEST_F(StreamTest, ReportsRestoreFailureAfterDestroyingOwnedStream) {
  fake_cuda_state.current_device_ = 1;
  const auto error_sink = std::make_shared<RecordingErrorSink>();

  {
    const auto stream = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
    fake_cuda_state.fail_set_device_ = 1;
  }

  EXPECT_EQ(fake_cuda_state.destroy_stream_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.current_device_, 0);
  ASSERT_TRUE(error_sink->GetRecord().has_value());
  EXPECT_THAT(error_sink->GetRecord()->message_, HasSubstr("restore after stream destruction"));
}

TEST_F(StreamTest, WrapsExternalStreamWithoutCudaCallsOrDestruction) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto external_owner = std::make_shared<int>(42);
  const std::weak_ptr<int> weak_owner = external_owner;

  {
    const auto stream =
        StreamAccess::WrapExternal(Device{3}, fake_cuda_state.stream_to_create_, external_owner, error_sink);
    external_owner.reset();

    EXPECT_EQ(stream.GetDevice(), Device{3});
    EXPECT_TRUE(stream.IsExternal());
    EXPECT_EQ(StreamAccess::GetNative(stream), fake_cuda_state.stream_to_create_);
    EXPECT_FALSE(weak_owner.expired());
  }

  EXPECT_TRUE(weak_owner.expired());
  EXPECT_EQ(fake_cuda_state.get_device_call_count_, 0);
  EXPECT_EQ(fake_cuda_state.destroy_stream_call_count_, 0);
  EXPECT_EQ(error_sink->GetReportCount(), 0);
}

TEST_F(StreamTest, RejectsCudaDefaultStreamHandles) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();

  EXPECT_THROW(static_cast<void>(StreamAccess::WrapExternal(Device{0}, nullptr, nullptr, error_sink)),
               InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(StreamAccess::WrapExternal(Device{0}, cudaStreamLegacy, nullptr, error_sink)),
               InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(StreamAccess::WrapExternal(Device{0}, cudaStreamPerThread, nullptr, error_sink)),
               InvalidArgumentError);
}

}  // namespace
}  // namespace ttl::internal
