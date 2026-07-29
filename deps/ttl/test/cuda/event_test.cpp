#include "ttl/internal/event.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <thread>
#include <utility>

#include <cuda_runtime_api.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/event.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {
namespace {

using testing::HasSubstr;

int fake_stream_storage[2];
int fake_event_storage;

void CUDART_CB HoldStreamUntilReleased(void *data) {
  const auto *gate = static_cast<const std::atomic<bool> *>(data);
  while (!gate->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

class StreamGate final {
 public:
  explicit StreamGate(cudaStream_t stream) noexcept : stream_(stream) {}
  StreamGate(const StreamGate &) = delete;
  auto operator=(const StreamGate &) -> StreamGate & = delete;

  ~StreamGate() noexcept {
    Release();
    cudaStreamSynchronize(stream_);
  }

  [[nodiscard]] auto GetFlag() noexcept -> std::atomic<bool> * { return &gate_; }
  void Release() noexcept { gate_.store(true, std::memory_order_release); }

 private:
  cudaStream_t stream_;
  std::atomic<bool> gate_{false};
};

struct FakeCudaState final {
  int current_device_{0};
  int least_priority_{0};
  int greatest_priority_{-3};
  cudaStream_t stream_to_create_{nullptr};
  cudaEvent_t event_to_create_{nullptr};
  cudaError_t get_device_status_{cudaSuccess};
  cudaError_t create_event_status_{cudaSuccess};
  cudaError_t record_event_status_{cudaSuccess};
  cudaError_t query_event_status_{cudaSuccess};
  cudaError_t synchronize_event_status_{cudaSuccess};
  cudaError_t destroy_event_status_{cudaSuccess};
  cudaError_t wait_event_status_{cudaSuccess};
  cudaError_t last_error_status_{cudaErrorNotReady};
  std::optional<int> fail_set_device_;
  int get_device_call_count_{0};
  int set_device_call_count_{0};
  int get_last_error_call_count_{0};
  int create_event_call_count_{0};
  int record_event_call_count_{0};
  int query_event_call_count_{0};
  int synchronize_event_call_count_{0};
  int destroy_event_call_count_{0};
  int wait_event_call_count_{0};
  int destroy_stream_call_count_{0};
  int last_set_device_{-1};
  unsigned int last_event_flags_{0};
  unsigned int last_wait_flags_{0};
  cudaStream_t last_record_stream_{nullptr};
  cudaStream_t last_wait_stream_{nullptr};
  cudaEvent_t last_recorded_event_{nullptr};
  cudaEvent_t last_waited_event_{nullptr};
  cudaEvent_t last_destroyed_event_{nullptr};
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

auto FakeGetLastError() -> cudaError_t {
  fake_cuda_state.get_last_error_call_count_++;
  return std::exchange(fake_cuda_state.last_error_status_, cudaSuccess);
}

auto FakeGetStreamPriorityRange(int *least_priority, int *greatest_priority) -> cudaError_t {
  *least_priority = fake_cuda_state.least_priority_;
  *greatest_priority = fake_cuda_state.greatest_priority_;
  return cudaSuccess;
}

auto FakeCreateStreamWithPriority(cudaStream_t *stream, unsigned int /* flags */, int /* priority */) -> cudaError_t {
  *stream = fake_cuda_state.stream_to_create_;
  return cudaSuccess;
}

auto FakeDestroyStream(cudaStream_t /* stream */) -> cudaError_t {
  fake_cuda_state.destroy_stream_call_count_++;
  return cudaSuccess;
}

auto FakeCreateEventWithFlags(cudaEvent_t *event, unsigned int flags) -> cudaError_t {
  fake_cuda_state.create_event_call_count_++;
  fake_cuda_state.last_event_flags_ = flags;
  if (fake_cuda_state.create_event_status_ == cudaSuccess) {
    *event = fake_cuda_state.event_to_create_;
  }
  return fake_cuda_state.create_event_status_;
}

auto FakeRecordEvent(cudaEvent_t event, cudaStream_t stream) -> cudaError_t {
  fake_cuda_state.record_event_call_count_++;
  fake_cuda_state.last_recorded_event_ = event;
  fake_cuda_state.last_record_stream_ = stream;
  return fake_cuda_state.record_event_status_;
}

auto FakeQueryEvent(cudaEvent_t /* event */) -> cudaError_t {
  fake_cuda_state.query_event_call_count_++;
  return fake_cuda_state.query_event_status_;
}

auto FakeSynchronizeEvent(cudaEvent_t /* event */) -> cudaError_t {
  fake_cuda_state.synchronize_event_call_count_++;
  return fake_cuda_state.synchronize_event_status_;
}

auto FakeDestroyEvent(cudaEvent_t event) -> cudaError_t {
  fake_cuda_state.destroy_event_call_count_++;
  fake_cuda_state.last_destroyed_event_ = event;
  return fake_cuda_state.destroy_event_status_;
}

auto FakeStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) -> cudaError_t {
  fake_cuda_state.wait_event_call_count_++;
  fake_cuda_state.last_wait_stream_ = stream;
  fake_cuda_state.last_waited_event_ = event;
  fake_cuda_state.last_wait_flags_ = flags;
  return fake_cuda_state.wait_event_status_;
}

const CudaApi FAKE_CUDA_API{
    .get_device_count_ = cudaGetDeviceCount,
    .get_device_properties_ = cudaGetDeviceProperties,
    .get_device_ = FakeGetDevice,
    .set_device_ = FakeSetDevice,
    .get_last_error_ = FakeGetLastError,
    .get_stream_priority_range_ = FakeGetStreamPriorityRange,
    .create_stream_with_priority_ = FakeCreateStreamWithPriority,
    .destroy_stream_ = FakeDestroyStream,
    .create_event_with_flags_ = FakeCreateEventWithFlags,
    .record_event_ = FakeRecordEvent,
    .query_event_ = FakeQueryEvent,
    .synchronize_event_ = FakeSynchronizeEvent,
    .destroy_event_ = FakeDestroyEvent,
    .stream_wait_event_ = FakeStreamWaitEvent,
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

class EventTest : public testing::Test {
 protected:
  void SetUp() override {
    fake_cuda_state = FakeCudaState{};
    fake_cuda_state.stream_to_create_ = reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]);
    fake_cuda_state.event_to_create_ = reinterpret_cast<cudaEvent_t>(&fake_event_storage);
  }

  [[nodiscard]] static auto MakeStream(Device device, cudaStream_t stream, const std::shared_ptr<ErrorSink> &error_sink)
      -> Stream {
    fake_cuda_state.stream_to_create_ = stream;
    return StreamAccess::CreateOwned(device, 0, error_sink);
  }

 private:
  ScopedCudaApiOverride cuda_api_override_{FAKE_CUDA_API};
};

TEST_F(EventTest, RecordsTimingDisabledEventAndSharesItsLifetime) {
  fake_cuda_state.current_device_ = 1;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = MakeStream(Device{2}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);

  {
    const auto event = EventAccess::Record(stream);
    const auto event_copy = event;

    EXPECT_EQ(event.GetDevice(), Device{2});
    EXPECT_EQ(event_copy.GetDevice(), Device{2});
    EXPECT_EQ(EventAccess::GetNative(event), fake_cuda_state.event_to_create_);
    EXPECT_EQ(fake_cuda_state.last_event_flags_, cudaEventDisableTiming);
    EXPECT_EQ(fake_cuda_state.last_recorded_event_, fake_cuda_state.event_to_create_);
    EXPECT_EQ(fake_cuda_state.last_record_stream_, StreamAccess::GetNative(stream));
    EXPECT_EQ(fake_cuda_state.current_device_, 1);
    EXPECT_EQ(fake_cuda_state.destroy_event_call_count_, 0);
  }

  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.last_destroyed_event_, fake_cuda_state.event_to_create_);
  EXPECT_EQ(fake_cuda_state.current_device_, 1);
  EXPECT_EQ(error_sink->GetReportCount(), 0);
}

TEST_F(EventTest, KeepsTheCleanupSinkAliveAfterTheRecordingStreamIsReleased) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  const std::weak_ptr<RecordingErrorSink> weak_error_sink = error_sink;
  std::optional<Event> event;

  {
    const auto stream = MakeStream(Device{0}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);
    event.emplace(EventAccess::Record(stream));
    error_sink.reset();
    EXPECT_FALSE(weak_error_sink.expired());
  }

  EXPECT_FALSE(weak_error_sink.expired());
  event.reset();
  EXPECT_TRUE(weak_error_sink.expired());
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_, 1);
}

TEST_F(EventTest, QueryDistinguishesCompletionNotReadyAndFailure) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = MakeStream(Device{0}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);
  const auto event = EventAccess::Record(stream);

  EXPECT_TRUE(event.Query());
  EXPECT_EQ(fake_cuda_state.get_last_error_call_count_, 0);

  fake_cuda_state.query_event_status_ = cudaErrorNotReady;
  EXPECT_FALSE(event.Query());
  EXPECT_EQ(fake_cuda_state.get_last_error_call_count_, 1);

  fake_cuda_state.query_event_status_ = cudaErrorInvalidResourceHandle;
  const auto location = std::source_location::current();
  try {
    static_cast<void>(event.Query(location));
    FAIL() << "Query did not throw";
  } catch (const CudaError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("cudaEventQuery"));
    EXPECT_EQ(error.GetLocation().line(), location.line());
  }
}

TEST_F(EventTest, QueryAndSynchronizeDoNotChangeTheCurrentDevice) {
  fake_cuda_state.current_device_ = 1;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = MakeStream(Device{2}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);
  const auto event = EventAccess::Record(stream);
  fake_cuda_state.get_device_call_count_ = 0;
  fake_cuda_state.set_device_call_count_ = 0;

  EXPECT_TRUE(event.Query());
  EXPECT_NO_THROW(event.Synchronize());

  EXPECT_EQ(fake_cuda_state.query_event_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.synchronize_event_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.get_device_call_count_, 0);
  EXPECT_EQ(fake_cuda_state.set_device_call_count_, 0);
  EXPECT_EQ(fake_cuda_state.current_device_, 1);
}

TEST_F(EventTest, SynchronizeReportsFailureAtTheCallSite) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = MakeStream(Device{0}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);
  const auto event = EventAccess::Record(stream);
  fake_cuda_state.synchronize_event_status_ = cudaErrorLaunchFailure;
  const auto location = std::source_location::current();

  try {
    event.Synchronize(location);
    FAIL() << "Synchronize did not throw";
  } catch (const CudaError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("cudaEventSynchronize"));
    EXPECT_EQ(error.GetLocation().line(), location.line());
  }
}

TEST_F(EventTest, WaitSkipsTheRecordingStreamAndSupportsCrossDeviceConsumers) {
  fake_cuda_state.current_device_ = 0;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto producer = MakeStream(Device{1}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);
  const auto event = EventAccess::Record(producer);

  EventAccess::Wait(producer, event);
  EXPECT_EQ(fake_cuda_state.wait_event_call_count_, 0);

  const auto consumer = MakeStream(Device{2}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[1]), error_sink);
  fake_cuda_state.get_device_call_count_ = 0;
  fake_cuda_state.set_device_call_count_ = 0;
  EventAccess::Wait(consumer, event);

  EXPECT_EQ(fake_cuda_state.wait_event_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.last_wait_stream_, StreamAccess::GetNative(consumer));
  EXPECT_EQ(fake_cuda_state.last_waited_event_, EventAccess::GetNative(event));
  EXPECT_EQ(fake_cuda_state.last_wait_flags_, cudaEventWaitDefault);
  EXPECT_EQ(fake_cuda_state.get_device_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.set_device_call_count_, 2);
  EXPECT_EQ(fake_cuda_state.current_device_, 0);
}

TEST_F(EventTest, WaitFailureRestoresTheConsumerDeviceAndPreservesTheCallSite) {
  fake_cuda_state.current_device_ = 0;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto producer = MakeStream(Device{1}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);
  const auto consumer = MakeStream(Device{2}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[1]), error_sink);
  const auto event = EventAccess::Record(producer);
  fake_cuda_state.wait_event_status_ = cudaErrorInvalidResourceHandle;
  const auto location = std::source_location::current();

  try {
    EventAccess::Wait(consumer, event, location);
    FAIL() << "Wait did not throw";
  } catch (const CudaError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("cudaStreamWaitEvent"));
    EXPECT_EQ(error.GetLocation().line(), location.line());
    EXPECT_EQ(fake_cuda_state.current_device_, 0);
  }
}

TEST_F(EventTest, RollsBackAHandleWhenRecordFails) {
  fake_cuda_state.current_device_ = 1;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = MakeStream(Device{2}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);
  fake_cuda_state.record_event_status_ = cudaErrorLaunchFailure;

  EXPECT_THROW(static_cast<void>(EventAccess::Record(stream)), CudaError);
  EXPECT_EQ(fake_cuda_state.create_event_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.record_event_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.current_device_, 1);
}

TEST_F(EventTest, RejectsInvalidCreationResultsWithoutRecording) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = MakeStream(Device{0}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);

  fake_cuda_state.create_event_status_ = cudaErrorInitializationError;
  EXPECT_THROW(static_cast<void>(EventAccess::Record(stream)), CudaError);
  EXPECT_EQ(fake_cuda_state.record_event_call_count_, 0);
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_, 0);

  fake_cuda_state.create_event_status_ = cudaSuccess;
  fake_cuda_state.event_to_create_ = nullptr;
  EXPECT_THROW(static_cast<void>(EventAccess::Record(stream)), InternalError);
  EXPECT_EQ(fake_cuda_state.record_event_call_count_, 0);
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_, 0);
}

TEST_F(EventTest, ReportsCleanupFailureWithProducerContext) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = MakeStream(Device{0}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);
  const auto location = std::source_location::current();
  uint64_t stream_id = 0;

  {
    const auto event = EventAccess::Record(stream, location);
    stream_id = stream.GetId();
    fake_cuda_state.destroy_event_status_ = cudaErrorInvalidResourceHandle;
  }

  ASSERT_TRUE(error_sink->GetRecord().has_value());
  const auto &record = *error_sink->GetRecord();
  EXPECT_EQ(record.code_, ErrorCode::CUDA);
  EXPECT_EQ(record.device_, Device{0});
  EXPECT_EQ(record.stream_id_, stream_id);
  EXPECT_EQ(record.location_.line(), location.line());
  EXPECT_THAT(record.message_, HasSubstr("cudaEventDestroy"));
}

TEST_F(EventTest, DoesNotDestroyWhenCleanupCannotReadOrSelectTheOwningDevice) {
  fake_cuda_state.current_device_ = 1;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = MakeStream(Device{2}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);

  {
    const auto event = EventAccess::Record(stream);
    fake_cuda_state.get_device_status_ = cudaErrorInitializationError;
  }
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_, 0);
  ASSERT_TRUE(error_sink->GetRecord().has_value());
  EXPECT_THAT(error_sink->GetRecord()->message_, HasSubstr("cudaGetDevice (destroy event)"));

  fake_cuda_state.get_device_status_ = cudaSuccess;
  {
    const auto event = EventAccess::Record(stream);
    fake_cuda_state.fail_set_device_ = 2;
  }
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_, 0);
  ASSERT_TRUE(error_sink->GetRecord().has_value());
  EXPECT_THAT(error_sink->GetRecord()->message_, HasSubstr("cudaSetDevice (destroy event)"));

  fake_cuda_state.fail_set_device_.reset();
}

TEST_F(EventTest, ReportsRestoreFailureAfterDestroyingTheEvent) {
  fake_cuda_state.current_device_ = 1;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = MakeStream(Device{2}, reinterpret_cast<cudaStream_t>(&fake_stream_storage[0]), error_sink);

  {
    const auto event = EventAccess::Record(stream);
    fake_cuda_state.fail_set_device_ = 1;
  }

  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_, 1);
  EXPECT_EQ(fake_cuda_state.current_device_, 2);
  ASSERT_TRUE(error_sink->GetRecord().has_value());
  EXPECT_THAT(error_sink->GetRecord()->message_, HasSubstr("restore after event destruction"));
  fake_cuda_state.fail_set_device_.reset();
}

TEST(EventIntegrationTest, ObservesIncompleteAndCompletedCudaWork) {
  int device_count = 0;
  const auto count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    GTEST_SKIP() << "CUDA device unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto stream = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
  StreamGate gate{StreamAccess::GetNative(stream)};
  ASSERT_EQ(cudaLaunchHostFunc(StreamAccess::GetNative(stream), HoldStreamUntilReleased, gate.GetFlag()), cudaSuccess);
  const auto event = EventAccess::Record(stream);

  EXPECT_FALSE(event.Query());
  gate.Release();
  event.Synchronize();
  EXPECT_TRUE(event.Query());
  EXPECT_EQ(error_sink->GetReportCount(), 0);
}

TEST(EventIntegrationTest, WaitsAcrossCudaDevices) {
  int device_count = 0;
  const auto count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count < 2) {
    cudaGetLastError();
    GTEST_SKIP() << "two CUDA devices unavailable";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  const auto error_sink = std::make_shared<RecordingErrorSink>();
  const auto producer = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
  const auto consumer = StreamAccess::CreateOwned(Device{1}, 0, error_sink);
  StreamGate gate{StreamAccess::GetNative(producer)};
  ASSERT_EQ(cudaLaunchHostFunc(StreamAccess::GetNative(producer), HoldStreamUntilReleased, gate.GetFlag()),
            cudaSuccess);
  const auto producer_ready = EventAccess::Record(producer);
  EventAccess::Wait(consumer, producer_ready);
  const auto consumer_ready = EventAccess::Record(consumer);

  EXPECT_FALSE(consumer_ready.Query());
  gate.Release();
  consumer_ready.Synchronize();
  EXPECT_TRUE(consumer_ready.Query());
  EXPECT_EQ(error_sink->GetReportCount(), 0);
}

}  // namespace
}  // namespace ttl::internal
