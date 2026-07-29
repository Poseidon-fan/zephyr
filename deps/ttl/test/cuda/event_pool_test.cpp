#include "ttl/internal/event_pool.hpp"

#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/device_guard.hpp"

namespace ttl::internal {
namespace {

constexpr size_t MAX_FAKE_EVENTS = 4096;
std::array<int, MAX_FAKE_EVENTS> fake_event_storage;
thread_local int fake_current_device = 0;

struct FakeCudaState final {
  std::atomic<cudaError_t> get_device_status_{cudaSuccess};
  std::atomic<cudaError_t> set_device_status_{cudaSuccess};
  std::atomic<cudaError_t> create_event_status_{cudaSuccess};
  std::atomic<cudaError_t> destroy_event_status_{cudaSuccess};
  std::atomic<size_t> get_device_call_count_{0};
  std::atomic<size_t> set_device_call_count_{0};
  std::atomic<size_t> create_event_call_count_{0};
  std::atomic<size_t> destroy_event_call_count_{0};
  std::atomic<size_t> next_event_index_{0};
  std::atomic<int> last_create_device_{-1};
  std::atomic<unsigned int> last_create_flags_{0};
  std::atomic<bool> return_null_event_{false};
  std::atomic<bool> block_create_{false};
  std::atomic<bool> create_entered_{false};
  std::atomic<bool> allow_create_{false};
};

FakeCudaState fake_cuda_state;

void ResetFakeCudaState() {
  fake_current_device = 0;
  fake_cuda_state.get_device_status_.store(cudaSuccess);
  fake_cuda_state.set_device_status_.store(cudaSuccess);
  fake_cuda_state.create_event_status_.store(cudaSuccess);
  fake_cuda_state.destroy_event_status_.store(cudaSuccess);
  fake_cuda_state.get_device_call_count_.store(0);
  fake_cuda_state.set_device_call_count_.store(0);
  fake_cuda_state.create_event_call_count_.store(0);
  fake_cuda_state.destroy_event_call_count_.store(0);
  fake_cuda_state.next_event_index_.store(0);
  fake_cuda_state.last_create_device_.store(-1);
  fake_cuda_state.last_create_flags_.store(0);
  fake_cuda_state.return_null_event_.store(false);
  fake_cuda_state.block_create_.store(false);
  fake_cuda_state.create_entered_.store(false);
  fake_cuda_state.allow_create_.store(false);
}

auto FakeGetDevice(int *device) -> cudaError_t {
  fake_cuda_state.get_device_call_count_.fetch_add(1);
  const auto status = fake_cuda_state.get_device_status_.load();
  if (status == cudaSuccess) {
    *device = fake_current_device;
  }
  return status;
}

auto FakeSetDevice(int device) -> cudaError_t {
  fake_cuda_state.set_device_call_count_.fetch_add(1);
  const auto status = fake_cuda_state.set_device_status_.load();
  if (status == cudaSuccess) {
    fake_current_device = device;
  }
  return status;
}

auto FakeCreateEventWithFlags(cudaEvent_t *event, unsigned int flags) -> cudaError_t {
  fake_cuda_state.create_event_call_count_.fetch_add(1);
  fake_cuda_state.last_create_device_.store(fake_current_device);
  fake_cuda_state.last_create_flags_.store(flags);
  if (fake_cuda_state.block_create_.load()) {
    fake_cuda_state.create_entered_.store(true);
    fake_cuda_state.create_entered_.notify_all();
    fake_cuda_state.allow_create_.wait(false);
  }

  const auto status = fake_cuda_state.create_event_status_.load();
  if (status != cudaSuccess) {
    return status;
  }
  if (fake_cuda_state.return_null_event_.load()) {
    *event = nullptr;
    return cudaSuccess;
  }

  const auto index = fake_cuda_state.next_event_index_.fetch_add(1);
  if (index >= fake_event_storage.size()) {
    return cudaErrorMemoryAllocation;
  }
  *event = reinterpret_cast<cudaEvent_t>(&fake_event_storage[index]);
  return cudaSuccess;
}

auto FakeDestroyEvent(cudaEvent_t /* event */) -> cudaError_t {
  fake_cuda_state.destroy_event_call_count_.fetch_add(1);
  return fake_cuda_state.destroy_event_status_.load();
}

const CudaApi FAKE_CUDA_API{
    .get_device_count_ = cudaGetDeviceCount,
    .get_device_properties_ = cudaGetDeviceProperties,
    .get_device_ = FakeGetDevice,
    .set_device_ = FakeSetDevice,
    .get_last_error_ = cudaGetLastError,
    .get_stream_priority_range_ = cudaDeviceGetStreamPriorityRange,
    .create_stream_with_priority_ = cudaStreamCreateWithPriority,
    .destroy_stream_ = cudaStreamDestroy,
    .create_event_with_flags_ = FakeCreateEventWithFlags,
    .record_event_ = cudaEventRecord,
    .query_event_ = cudaEventQuery,
    .synchronize_event_ = cudaEventSynchronize,
    .destroy_event_ = FakeDestroyEvent,
    .stream_wait_event_ = cudaStreamWaitEvent,
};

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override {
    std::scoped_lock lock{latch_};
    record_.emplace(std::move(error));
    report_count_++;
  }

  [[nodiscard]] auto GetRecord() const -> std::optional<ErrorRecord> {
    std::scoped_lock lock{latch_};
    return record_;
  }

  [[nodiscard]] auto GetReportCount() const -> size_t {
    std::scoped_lock lock{latch_};
    return report_count_;
  }

 private:
  mutable std::mutex latch_;
  std::optional<ErrorRecord> record_;
  size_t report_count_{0};
};

class EventPoolTest : public testing::Test {
 protected:
  void SetUp() override { ResetFakeCudaState(); }

 private:
  ScopedCudaApiOverride cuda_api_override_{FAKE_CUDA_API};
};

TEST_F(EventPoolTest, AcquiresAndReusesTimingDisabledEventsOnTheOwningDevice) {
  fake_current_device = 1;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{2}, error_sink, 2};
  cudaEvent_t first_event = nullptr;

  {
    auto event = pool.Acquire();
    first_event = event.GetNative();
    EXPECT_TRUE(event);
    EXPECT_NE(first_event, nullptr);
    EXPECT_EQ(fake_cuda_state.last_create_flags_.load(), cudaEventDisableTiming);
    EXPECT_EQ(fake_cuda_state.last_create_device_.load(), 2);
    EXPECT_EQ(fake_current_device, 1);

    const auto stats = pool.GetStats();
    EXPECT_EQ(stats.cached_event_count_, 0);
    EXPECT_EQ(stats.outstanding_event_count_, 1);
  }

  EXPECT_EQ(pool.GetStats().cached_event_count_, 1);
  {
    auto event = pool.Acquire();
    EXPECT_EQ(event.GetNative(), first_event);
  }
  EXPECT_EQ(fake_cuda_state.create_event_call_count_.load(), 1);

  pool.Close();
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_.load(), 1);
  EXPECT_EQ(fake_current_device, 1);
  EXPECT_TRUE(pool.GetStats().is_closed_);
}

TEST_F(EventPoolTest, RejectsNullErrorSinkBeforeCallingCuda) {
  EXPECT_THROW(EventPool(Device{0}, nullptr, 1), InvalidArgumentError);
  EXPECT_EQ(fake_cuda_state.create_event_call_count_.load(), 0);
}

TEST_F(EventPoolTest, ReserveWarmsTheCacheAndValidatesItsCapacity) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, 3};

  pool.Reserve(3);
  EXPECT_EQ(fake_cuda_state.create_event_call_count_.load(), 3);
  EXPECT_EQ(pool.GetStats().cached_event_count_, 3);

  pool.Reserve(2);
  EXPECT_EQ(fake_cuda_state.create_event_call_count_.load(), 3);
  EXPECT_EQ(pool.GetStats().cached_event_count_, 3);
  EXPECT_THROW(pool.Reserve(4), InvalidArgumentError);
}

TEST_F(EventPoolTest, ReserveRollsBackAcquiredLeasesWhenCreationFails) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, 2};
  fake_cuda_state.next_event_index_.store(MAX_FAKE_EVENTS - 1);

  EXPECT_THROW(pool.Reserve(2), CudaError);

  const auto stats = pool.GetStats();
  EXPECT_EQ(stats.cached_event_count_, 1);
  EXPECT_EQ(stats.outstanding_event_count_, 0);
  EXPECT_EQ(fake_cuda_state.create_event_call_count_.load(), 2);
  {
    auto event = pool.Acquire();
    EXPECT_TRUE(event);
  }
  EXPECT_EQ(fake_cuda_state.create_event_call_count_.load(), 2);
}

TEST_F(EventPoolTest, MoveAssignmentReturnsTheReplacedLeaseAndTransfersOwnership) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, 2};

  {
    auto first = pool.Acquire();
    auto second = pool.Acquire();
    const auto second_native = second.GetNative();

    first = std::move(second);

    EXPECT_FALSE(second);
    EXPECT_EQ(first.GetNative(), second_native);
    EXPECT_EQ(pool.GetStats().cached_event_count_, 1);
    EXPECT_EQ(pool.GetStats().outstanding_event_count_, 1);
  }

  EXPECT_EQ(pool.GetStats().cached_event_count_, 2);
  EXPECT_EQ(pool.GetStats().outstanding_event_count_, 0);
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_.load(), 0);
}

TEST_F(EventPoolTest, EnforcesTheCacheLimitWithoutLeakingOutstandingAccounting) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, 1};

  {
    auto first = pool.Acquire();
    {
      auto second = pool.Acquire();
      EXPECT_NE(first.GetNative(), second.GetNative());
    }
    EXPECT_EQ(pool.GetStats().cached_event_count_, 1);
    EXPECT_EQ(pool.GetStats().outstanding_event_count_, 1);
  }

  const auto stats = pool.GetStats();
  EXPECT_EQ(stats.cached_event_count_, 1);
  EXPECT_EQ(stats.outstanding_event_count_, 0);
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_.load(), 1);
}

TEST_F(EventPoolTest, DiscardDestroysAnEventInsteadOfCachingIt) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, 1};

  auto event = pool.Acquire();
  event.Discard();

  EXPECT_FALSE(event);
  EXPECT_EQ(pool.GetStats().cached_event_count_, 0);
  EXPECT_EQ(pool.GetStats().outstanding_event_count_, 0);
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_.load(), 1);
}

TEST_F(EventPoolTest, CloseDestroysCachedEventsAndDefersOutstandingLeaseDestruction) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto pool = std::make_unique<EventPool>(Device{0}, error_sink, 2);
  auto outstanding_event = pool->Acquire();

  {
    auto cached_event = pool->Acquire();
  }
  ASSERT_EQ(pool->GetStats().cached_event_count_, 1);

  pool->Close();
  EXPECT_TRUE(pool->GetStats().is_closed_);
  EXPECT_EQ(pool->GetStats().outstanding_event_count_, 1);
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_.load(), 1);
  EXPECT_THROW(static_cast<void>(pool->Acquire()), InvalidArgumentError);

  pool.reset();
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_.load(), 1);
  outstanding_event.Discard();
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_.load(), 2);
}

TEST_F(EventPoolTest, CloseRacingWithColdAcquireRejectsAndDestroysTheCreatedEvent) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, 1};
  fake_cuda_state.block_create_.store(true);
  std::exception_ptr acquire_error;
  bool acquire_succeeded = false;

  std::jthread acquire_thread{[&] {
    try {
      const auto event = pool.Acquire();
      acquire_succeeded = static_cast<bool>(event);
    } catch (...) {
      acquire_error = std::current_exception();
    }
  }};

  fake_cuda_state.create_entered_.wait(false);
  pool.Close();
  fake_cuda_state.allow_create_.store(true);
  fake_cuda_state.allow_create_.notify_all();
  acquire_thread.join();

  ASSERT_NE(acquire_error, nullptr);
  EXPECT_FALSE(acquire_succeeded);
  EXPECT_THROW(std::rethrow_exception(acquire_error), InvalidArgumentError);
  EXPECT_EQ(fake_cuda_state.destroy_event_call_count_.load(), 1);
  EXPECT_EQ(pool.GetStats().cached_event_count_, 0);
  EXPECT_EQ(pool.GetStats().outstanding_event_count_, 0);
}

TEST_F(EventPoolTest, RecoversAfterCreationFailuresAndRejectsNullSuccessfulResults) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, 1};

  fake_cuda_state.create_event_status_.store(cudaErrorInitializationError);
  EXPECT_THROW(static_cast<void>(pool.Acquire()), CudaError);

  fake_cuda_state.create_event_status_.store(cudaSuccess);
  fake_cuda_state.return_null_event_.store(true);
  EXPECT_THROW(static_cast<void>(pool.Acquire()), InternalError);

  fake_cuda_state.return_null_event_.store(false);
  auto event = pool.Acquire();
  EXPECT_TRUE(event);
  EXPECT_EQ(pool.GetStats().outstanding_event_count_, 1);
}

TEST_F(EventPoolTest, ReportsDestroyFailuresWithoutThrowingFromLeaseDestruction) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, 0};
  fake_cuda_state.destroy_event_status_.store(cudaErrorInvalidResourceHandle);

  {
    auto event = pool.Acquire();
  }

  EXPECT_EQ(error_sink->GetReportCount(), 1);
  const auto record = error_sink->GetRecord();
  ASSERT_TRUE(record.has_value());
  EXPECT_EQ(record->code_, ErrorCode::CUDA);
  EXPECT_NE(record->message_.find("cudaEventDestroy (event pool)"), std::string::npos);
  EXPECT_EQ(record->device_, Device{0});
}

TEST_F(EventPoolTest, SupportsConcurrentAcquireAndReleaseFromTheWarmCache) {
  constexpr size_t THREAD_COUNT = 8;
  constexpr size_t ITERATION_COUNT = 500;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, THREAD_COUNT};
  pool.Reserve(THREAD_COUNT);
  std::barrier start_barrier{static_cast<ptrdiff_t>(THREAD_COUNT)};
  std::atomic<bool> saw_invalid_event{false};

  std::vector<std::jthread> threads;
  threads.reserve(THREAD_COUNT);
  for (size_t thread_index = 0; thread_index < THREAD_COUNT; thread_index++) {
    threads.emplace_back([&] {
      start_barrier.arrive_and_wait();
      for (size_t iteration = 0; iteration < ITERATION_COUNT; iteration++) {
        auto event = pool.Acquire();
        if (!event) {
          saw_invalid_event.store(true);
        }
      }
    });
  }
  threads.clear();

  const auto stats = pool.GetStats();
  EXPECT_FALSE(saw_invalid_event.load());
  EXPECT_EQ(stats.cached_event_count_, THREAD_COUNT);
  EXPECT_EQ(stats.outstanding_event_count_, 0);
  EXPECT_EQ(fake_cuda_state.create_event_call_count_.load(), THREAD_COUNT);
  EXPECT_EQ(error_sink->GetReportCount(), 0);
}

TEST_F(EventPoolTest, SupportsConcurrentColdAcquireAndReturnsEveryEventToTheCache) {
  constexpr size_t THREAD_COUNT = 8;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  EventPool pool{Device{0}, error_sink, THREAD_COUNT};
  std::barrier start_barrier{static_cast<ptrdiff_t>(THREAD_COUNT)};
  std::barrier acquired_barrier{static_cast<ptrdiff_t>(THREAD_COUNT)};
  std::atomic<bool> saw_invalid_event{false};

  std::vector<std::jthread> threads;
  threads.reserve(THREAD_COUNT);
  for (size_t thread_index = 0; thread_index < THREAD_COUNT; thread_index++) {
    threads.emplace_back([&] {
      start_barrier.arrive_and_wait();
      auto event = pool.Acquire();
      if (!event) {
        saw_invalid_event.store(true);
      }
      acquired_barrier.arrive_and_wait();
    });
  }
  threads.clear();

  const auto stats = pool.GetStats();
  EXPECT_FALSE(saw_invalid_event.load());
  EXPECT_EQ(stats.cached_event_count_, THREAD_COUNT);
  EXPECT_EQ(stats.outstanding_event_count_, 0);
  EXPECT_EQ(fake_cuda_state.create_event_call_count_.load(), THREAD_COUNT);
  EXPECT_EQ(error_sink->GetReportCount(), 0);
}

TEST(EventPoolIntegrationTest, ReusesACompletedCudaEvent) {
  int device_count = 0;
  const auto count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    GTEST_SKIP() << "A CUDA device is required";
  }

  const auto error_sink = std::make_shared<RecordingErrorSink>();
  DeviceGuard device_guard{Device{0}, *error_sink};
  EventPool pool{Device{0}, error_sink, 1};
  cudaEvent_t first_native = nullptr;

  {
    auto event = pool.Acquire();
    first_native = event.GetNative();
    ASSERT_EQ(cudaEventRecord(first_native, nullptr), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(first_native), cudaSuccess);
  }

  {
    auto event = pool.Acquire();
    EXPECT_EQ(event.GetNative(), first_native);
    ASSERT_EQ(cudaEventRecord(event.GetNative(), nullptr), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(event.GetNative()), cudaSuccess);
  }

  pool.Close();
  EXPECT_EQ(error_sink->GetReportCount(), 0);
}

}  // namespace
}  // namespace ttl::internal
