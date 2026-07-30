#include "ttl/internal/device_allocator.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/allocation.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/event.hpp"
#include "ttl/internal/event_pool.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {
namespace {

constexpr size_t FAKE_RESOURCE_COUNT = 256;
std::array<int, FAKE_RESOURCE_COUNT> fake_stream_storage;
std::array<int, FAKE_RESOURCE_COUNT> fake_event_storage;
std::array<std::max_align_t, FAKE_RESOURCE_COUNT> fake_allocation_storage;
int fake_pool_storage;
thread_local int fake_current_device = 0;

struct FakeCudaState final {
  std::atomic<size_t> create_stream_count_{0};
  std::atomic<size_t> destroy_stream_count_{0};
  std::atomic<size_t> create_event_count_{0};
  std::atomic<size_t> record_event_count_{0};
  std::atomic<size_t> query_event_count_{0};
  std::atomic<size_t> synchronize_event_count_{0};
  std::atomic<size_t> destroy_event_count_{0};
  std::atomic<size_t> wait_event_count_{0};
  std::atomic<size_t> synchronize_stream_count_{0};
  std::atomic<size_t> create_pool_count_{0};
  std::atomic<size_t> destroy_pool_count_{0};
  std::atomic<size_t> set_pool_attribute_count_{0};
  std::atomic<size_t> set_pool_access_count_{0};
  std::atomic<size_t> trim_pool_count_{0};
  std::atomic<size_t> malloc_count_{0};
  std::atomic<size_t> free_count_{0};
  std::atomic<size_t> malloc_failures_before_success_{0};
  std::atomic<size_t> next_allocation_index_{0};
  std::atomic<bool> block_malloc_{false};
  std::atomic<bool> malloc_entered_{false};
  std::atomic<cudaError_t> create_event_status_{cudaSuccess};
  std::atomic<cudaError_t> query_event_status_{cudaSuccess};
  std::atomic<cudaError_t> free_status_{cudaSuccess};
  std::atomic<cudaError_t> last_error_{cudaSuccess};
  std::atomic<cudaMemoryType> pointer_type_{cudaMemoryTypeDevice};
  std::atomic<int> pointer_device_{0};
  std::atomic<int> can_access_peer_{1};
  std::atomic<uint64_t> pool_used_bytes_{0};
  std::atomic<uint64_t> pool_reserved_bytes_{0};
  std::atomic<cudaStream_t> last_free_stream_{nullptr};
  std::atomic<int> last_pool_location_{-1};
  std::atomic<int> last_access_device_{-1};
  std::atomic<cudaMemAccessFlags> last_access_flags_{cudaMemAccessFlagsProtNone};
};

FakeCudaState fake_cuda_state;

void ResetFakeCudaState() {
  fake_current_device = 0;
  fake_cuda_state.create_stream_count_.store(0);
  fake_cuda_state.destroy_stream_count_.store(0);
  fake_cuda_state.create_event_count_.store(0);
  fake_cuda_state.record_event_count_.store(0);
  fake_cuda_state.query_event_count_.store(0);
  fake_cuda_state.synchronize_event_count_.store(0);
  fake_cuda_state.destroy_event_count_.store(0);
  fake_cuda_state.wait_event_count_.store(0);
  fake_cuda_state.synchronize_stream_count_.store(0);
  fake_cuda_state.create_pool_count_.store(0);
  fake_cuda_state.destroy_pool_count_.store(0);
  fake_cuda_state.set_pool_attribute_count_.store(0);
  fake_cuda_state.set_pool_access_count_.store(0);
  fake_cuda_state.trim_pool_count_.store(0);
  fake_cuda_state.malloc_count_.store(0);
  fake_cuda_state.free_count_.store(0);
  fake_cuda_state.malloc_failures_before_success_.store(0);
  fake_cuda_state.next_allocation_index_.store(0);
  fake_cuda_state.block_malloc_.store(false);
  fake_cuda_state.malloc_entered_.store(false);
  fake_cuda_state.create_event_status_.store(cudaSuccess);
  fake_cuda_state.query_event_status_.store(cudaSuccess);
  fake_cuda_state.free_status_.store(cudaSuccess);
  fake_cuda_state.last_error_.store(cudaSuccess);
  fake_cuda_state.pointer_type_.store(cudaMemoryTypeDevice);
  fake_cuda_state.pointer_device_.store(0);
  fake_cuda_state.can_access_peer_.store(1);
  fake_cuda_state.pool_used_bytes_.store(0);
  fake_cuda_state.pool_reserved_bytes_.store(0);
  fake_cuda_state.last_free_stream_.store(nullptr);
  fake_cuda_state.last_pool_location_.store(-1);
  fake_cuda_state.last_access_device_.store(-1);
  fake_cuda_state.last_access_flags_.store(cudaMemAccessFlagsProtNone);
}

auto FakeGetDevice(int *device) -> cudaError_t {
  *device = fake_current_device;
  return cudaSuccess;
}

auto FakeSetDevice(int device) -> cudaError_t {
  fake_current_device = device;
  return cudaSuccess;
}

auto FakeGetLastError() -> cudaError_t { return fake_cuda_state.last_error_.exchange(cudaSuccess); }

auto FakeGetStreamPriorityRange(int *least_priority, int *greatest_priority) -> cudaError_t {
  *least_priority = 0;
  *greatest_priority = 0;
  return cudaSuccess;
}

auto FakeCreateStream(cudaStream_t *stream, unsigned int flags, int priority) -> cudaError_t {
  if (flags != cudaStreamNonBlocking || priority != 0) {
    return cudaErrorInvalidValue;
  }
  const auto index = fake_cuda_state.create_stream_count_.fetch_add(1);
  *stream = reinterpret_cast<cudaStream_t>(&fake_stream_storage.at(index));
  return cudaSuccess;
}

auto FakeDestroyStream(cudaStream_t /* stream */) -> cudaError_t {
  fake_cuda_state.destroy_stream_count_.fetch_add(1);
  return cudaSuccess;
}

auto FakeCreateEvent(cudaEvent_t *event, unsigned int flags) -> cudaError_t {
  const auto status = fake_cuda_state.create_event_status_.load();
  if (status != cudaSuccess) {
    return status;
  }
  if (flags != cudaEventDisableTiming) {
    return cudaErrorInvalidValue;
  }
  const auto index = fake_cuda_state.create_event_count_.fetch_add(1);
  *event = reinterpret_cast<cudaEvent_t>(&fake_event_storage.at(index));
  return cudaSuccess;
}

auto FakeRecordEvent(cudaEvent_t /* event */, cudaStream_t /* stream */) -> cudaError_t {
  fake_cuda_state.record_event_count_.fetch_add(1);
  return cudaSuccess;
}

auto FakeQueryEvent(cudaEvent_t /* event */) -> cudaError_t {
  fake_cuda_state.query_event_count_.fetch_add(1);
  return fake_cuda_state.query_event_status_.load();
}

auto FakeSynchronizeEvent(cudaEvent_t /* event */) -> cudaError_t {
  fake_cuda_state.synchronize_event_count_.fetch_add(1);
  return cudaSuccess;
}

auto FakeDestroyEvent(cudaEvent_t /* event */) -> cudaError_t {
  fake_cuda_state.destroy_event_count_.fetch_add(1);
  return cudaSuccess;
}

auto FakeWaitEvent(cudaStream_t /* stream */, cudaEvent_t /* event */, unsigned int flags) -> cudaError_t {
  if (flags != cudaEventWaitDefault) {
    return cudaErrorInvalidValue;
  }
  fake_cuda_state.wait_event_count_.fetch_add(1);
  return cudaSuccess;
}

auto FakeSynchronizeStream(cudaStream_t /* stream */) -> cudaError_t {
  fake_cuda_state.synchronize_stream_count_.fetch_add(1);
  return cudaSuccess;
}

auto FakeCreatePool(cudaMemPool_t *pool, const cudaMemPoolProps *properties) -> cudaError_t {
  fake_cuda_state.create_pool_count_.fetch_add(1);
  fake_cuda_state.last_pool_location_.store(properties->location.id);
  *pool = reinterpret_cast<cudaMemPool_t>(&fake_pool_storage);
  return cudaSuccess;
}

auto FakeDestroyPool(cudaMemPool_t /* pool */) -> cudaError_t {
  fake_cuda_state.destroy_pool_count_.fetch_add(1);
  return cudaSuccess;
}

auto FakeSetPoolAttribute(cudaMemPool_t /* pool */, cudaMemPoolAttr /* attribute */, void * /* value */)
    -> cudaError_t {
  fake_cuda_state.set_pool_attribute_count_.fetch_add(1);
  return cudaSuccess;
}

auto FakeGetPoolAttribute(cudaMemPool_t /* pool */, cudaMemPoolAttr attribute, void *value) -> cudaError_t {
  if (attribute == cudaMemPoolAttrUsedMemCurrent) {
    *static_cast<uint64_t *>(value) = fake_cuda_state.pool_used_bytes_.load();
  } else if (attribute == cudaMemPoolAttrReservedMemCurrent) {
    *static_cast<uint64_t *>(value) = fake_cuda_state.pool_reserved_bytes_.load();
  }
  return cudaSuccess;
}

auto FakeSetPoolAccess(cudaMemPool_t /* pool */, const cudaMemAccessDesc *descriptor, size_t count) -> cudaError_t {
  if (count != 1) {
    return cudaErrorInvalidValue;
  }
  fake_cuda_state.set_pool_access_count_.fetch_add(1);
  fake_cuda_state.last_access_device_.store(descriptor->location.id);
  fake_cuda_state.last_access_flags_.store(descriptor->flags);
  return cudaSuccess;
}

auto FakeTrimPool(cudaMemPool_t /* pool */, size_t /* minimum_bytes */) -> cudaError_t {
  fake_cuda_state.trim_pool_count_.fetch_add(1);
  return cudaSuccess;
}

auto FakeMallocFromPool(void **pointer, size_t /* bytes */, cudaMemPool_t /* pool */, cudaStream_t /* stream */)
    -> cudaError_t {
  fake_cuda_state.malloc_entered_.store(true, std::memory_order_release);
  fake_cuda_state.malloc_entered_.notify_all();
  while (fake_cuda_state.block_malloc_.load(std::memory_order_acquire)) {
    fake_cuda_state.block_malloc_.wait(true, std::memory_order_relaxed);
  }

  const auto call = fake_cuda_state.malloc_count_.fetch_add(1);
  if (call < fake_cuda_state.malloc_failures_before_success_.load()) {
    fake_cuda_state.last_error_.store(cudaErrorMemoryAllocation);
    return cudaErrorMemoryAllocation;
  }
  const auto index = fake_cuda_state.next_allocation_index_.fetch_add(1);
  *pointer = &fake_allocation_storage.at(index);
  return cudaSuccess;
}

auto FakeFreeAsync(void * /* pointer */, cudaStream_t stream) -> cudaError_t {
  fake_cuda_state.free_count_.fetch_add(1);
  fake_cuda_state.last_free_stream_.store(stream);
  return fake_cuda_state.free_status_.load();
}

auto FakeGetMemoryInfo(size_t *free_bytes, size_t *total_bytes) -> cudaError_t {
  *free_bytes = 1024;
  *total_bytes = 2048;
  return cudaSuccess;
}

auto FakeGetPointerAttributes(cudaPointerAttributes *attributes, const void * /* pointer */) -> cudaError_t {
  attributes->type = fake_cuda_state.pointer_type_.load();
  attributes->device = fake_cuda_state.pointer_device_.load();
  return cudaSuccess;
}

auto FakeCanAccessPeer(int *can_access, int /* device */, int /* peer */) -> cudaError_t {
  *can_access = fake_cuda_state.can_access_peer_.load();
  return cudaSuccess;
}

[[nodiscard]] auto MakeFakeCudaApi() -> CudaApi {
  auto cuda_api = GetCudaApi();
  cuda_api.get_device_ = FakeGetDevice;
  cuda_api.set_device_ = FakeSetDevice;
  cuda_api.get_last_error_ = FakeGetLastError;
  cuda_api.get_stream_priority_range_ = FakeGetStreamPriorityRange;
  cuda_api.create_stream_with_priority_ = FakeCreateStream;
  cuda_api.destroy_stream_ = FakeDestroyStream;
  cuda_api.create_event_with_flags_ = FakeCreateEvent;
  cuda_api.record_event_ = FakeRecordEvent;
  cuda_api.query_event_ = FakeQueryEvent;
  cuda_api.synchronize_event_ = FakeSynchronizeEvent;
  cuda_api.destroy_event_ = FakeDestroyEvent;
  cuda_api.stream_wait_event_ = FakeWaitEvent;
  cuda_api.synchronize_stream_ = FakeSynchronizeStream;
  cuda_api.create_memory_pool_ = FakeCreatePool;
  cuda_api.destroy_memory_pool_ = FakeDestroyPool;
  cuda_api.set_memory_pool_attribute_ = FakeSetPoolAttribute;
  cuda_api.get_memory_pool_attribute_ = FakeGetPoolAttribute;
  cuda_api.set_memory_pool_access_ = FakeSetPoolAccess;
  cuda_api.trim_memory_pool_ = FakeTrimPool;
  cuda_api.malloc_from_pool_async_ = FakeMallocFromPool;
  cuda_api.free_async_ = FakeFreeAsync;
  cuda_api.get_memory_info_ = FakeGetMemoryInfo;
  cuda_api.get_pointer_attributes_ = FakeGetPointerAttributes;
  cuda_api.can_access_peer_ = FakeCanAccessPeer;
  return cuda_api;
}

const CudaApi FAKE_CUDA_API = MakeFakeCudaApi();

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override {
    std::scoped_lock lock{latch_};
    last_record_.emplace(std::move(error));
    report_count_++;
  }

  [[nodiscard]] auto GetReportCount() const -> size_t {
    std::scoped_lock lock{latch_};
    return report_count_;
  }

 private:
  mutable std::mutex latch_;
  std::optional<ErrorRecord> last_record_;
  size_t report_count_{0};
};

[[nodiscard]] auto MakeExternalStream(size_t index, Device device, const std::shared_ptr<ErrorSink> &error_sink)
    -> Stream {
  return StreamAccess::WrapExternal(device, reinterpret_cast<cudaStream_t>(&fake_stream_storage.at(index + 32)),
                                    nullptr, error_sink);
}

[[nodiscard]] auto MakeAllocator(const std::shared_ptr<ErrorSink> &error_sink, DeviceAllocatorOptions options = {})
    -> std::shared_ptr<DeviceAllocator> {
  auto event_pool = std::make_shared<EventPool>(Device{0}, error_sink, 64);
  return DeviceAllocator::Create(Device{0}, error_sink, event_pool, options);
}

[[nodiscard]] auto MakeContext(std::string_view operation = "test allocation") -> AllocationContext {
  return AllocationContext{
      .operation_ = operation,
      .output_shape_ = std::nullopt,
      .dtype_ = std::nullopt,
      .location_ = std::source_location::current(),
  };
}

class DeviceAllocatorTest : public testing::Test {
 protected:
  void SetUp() override { ResetFakeCudaState(); }

 private:
  ScopedCudaApiOverride cuda_api_override_{FAKE_CUDA_API};
};

TEST_F(DeviceAllocatorTest, CreatesPrivatePoolAndRepresentsZeroByteStorageWithoutCudaAllocation) {
  fake_current_device = 2;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);

  EXPECT_EQ(fake_current_device, 2);
  EXPECT_EQ(fake_cuda_state.create_pool_count_.load(), 1);
  EXPECT_EQ(fake_cuda_state.set_pool_attribute_count_.load(), 4);
  EXPECT_EQ(fake_cuda_state.last_pool_location_.load(), 0);

  auto storage = allocator->Allocate(stream, 0, 1, MakeContext());
  EXPECT_EQ(storage->GetBasePointer(), nullptr);
  EXPECT_EQ(storage->GetCapacityBytes(), 0);
  EXPECT_EQ(storage->GetDevice(), Device{0});
  EXPECT_EQ(fake_cuda_state.malloc_count_.load(), 0);
  EXPECT_EQ(allocator->GetStats().outstanding_storage_count_, 1);

  storage.reset();
  EXPECT_EQ(allocator->GetStats().outstanding_storage_count_, 0);
  EXPECT_EQ(allocator->GetStats().pending_retirement_count_, 0);
  allocator->Shutdown();
  EXPECT_EQ(fake_cuda_state.destroy_pool_count_.load(), 1);
  EXPECT_EQ(fake_current_device, 2);
}

TEST_F(DeviceAllocatorTest, ValidatesStreamAlignmentAndBudgetBeforeCallingCuda) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink, DeviceAllocatorOptions{.max_live_bytes_ = 64});
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  const auto other_device_stream = MakeExternalStream(1, Device{1}, error_sink);

  EXPECT_THROW(
      { [[maybe_unused]] const auto storage = allocator->Allocate(stream, 16, 3, MakeContext()); },
      InvalidArgumentError);
  EXPECT_THROW(
      { [[maybe_unused]] const auto storage = allocator->Allocate(stream, 16, 512, MakeContext()); },
      InvalidArgumentError);
  EXPECT_THROW(
      { [[maybe_unused]] const auto storage = allocator->Allocate(other_device_stream, 16, 16, MakeContext()); },
      InvalidArgumentError);
  EXPECT_THROW(
      { [[maybe_unused]] const auto storage = allocator->Allocate(stream, 65, 16, MakeContext()); }, OutOfMemoryError);
  EXPECT_EQ(fake_cuda_state.malloc_count_.load(), 0);
  EXPECT_EQ(allocator->GetStats().oom_count_, 1);

  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, RetiresSingleStreamAllocationAfterItsCompletionFence) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  auto storage = allocator->Allocate(stream, 32, 16, MakeContext());

  EXPECT_EQ(allocator->GetStats().logical_live_bytes_, 32);
  fake_cuda_state.query_event_status_.store(cudaErrorNotReady);
  storage.reset();

  auto stats = allocator->GetStats();
  EXPECT_EQ(stats.logical_live_bytes_, 0);
  EXPECT_EQ(stats.retiring_bytes_, 32);
  EXPECT_EQ(stats.pending_retirement_count_, 1);
  EXPECT_EQ(fake_cuda_state.free_count_.load(), 1);
  EXPECT_EQ(fake_cuda_state.record_event_count_.load(), 1);

  allocator->Poll();
  EXPECT_EQ(allocator->GetStats().pending_retirement_count_, 1);
  fake_cuda_state.query_event_status_.store(cudaSuccess);
  allocator->Poll();
  stats = allocator->GetStats();
  EXPECT_EQ(stats.retiring_bytes_, 0);
  EXPECT_EQ(stats.pending_retirement_count_, 0);

  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, UnifiesAllocationAndCrossDeviceSideStreamsBeforeFree) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto allocation_stream = MakeExternalStream(0, Device{0}, error_sink);
  const auto side_stream = MakeExternalStream(1, Device{1}, error_sink);
  auto storage = allocator->Allocate(allocation_stream, 32, 16, MakeContext());

  storage->RecordUsage(side_stream);
  storage.reset();

  EXPECT_EQ(fake_cuda_state.wait_event_count_.load(), 2);
  EXPECT_EQ(fake_cuda_state.record_event_count_.load(), 3);
  EXPECT_EQ(fake_cuda_state.destroy_event_count_.load(), 2);
  EXPECT_EQ(fake_cuda_state.last_free_stream_.load(), reinterpret_cast<cudaStream_t>(&fake_stream_storage.front()));

  allocator->Poll();
  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, RetriesCudaOutOfMemoryExactlyOnceAfterPollingAndTrimming) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink, DeviceAllocatorOptions{.max_reserved_bytes_ = 64});
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  fake_cuda_state.malloc_failures_before_success_.store(1);

  auto storage = allocator->Allocate(stream, 32, 16, MakeContext("retry"));
  EXPECT_EQ(fake_cuda_state.malloc_count_.load(), 2);
  EXPECT_EQ(fake_cuda_state.trim_pool_count_.load(), 1);
  EXPECT_EQ(allocator->GetStats().retry_count_, 1);

  storage.reset();
  allocator->Poll();
  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, ReportsOutOfMemoryAfterOneFailedRetryAndRollsBackReservation) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  fake_cuda_state.malloc_failures_before_success_.store(2);

  EXPECT_THROW(
      { [[maybe_unused]] const auto storage = allocator->Allocate(stream, 32, 16, MakeContext("failure")); },
      OutOfMemoryError);
  EXPECT_EQ(fake_cuda_state.malloc_count_.load(), 2);
  EXPECT_EQ(fake_cuda_state.trim_pool_count_.load(), 1);
  const auto stats = allocator->GetStats();
  EXPECT_EQ(stats.logical_live_bytes_, 0);
  EXPECT_EQ(stats.retiring_bytes_, 0);
  EXPECT_EQ(stats.outstanding_storage_count_, 0);
  EXPECT_EQ(stats.oom_count_, 1);

  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, DelaysExternalOwnerReleaseButNeverFreesBorrowedMemory) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  auto owner = std::make_shared<int>(7);
  const std::weak_ptr<int> weak_owner = owner;

  auto owned =
      allocator->WrapExternal(stream, &fake_allocation_storage.front(), 32, ExternalOwnership::SHARED_OWNER, owner);
  owner.reset();
  owned.reset();
  EXPECT_FALSE(weak_owner.expired());
  EXPECT_EQ(fake_cuda_state.free_count_.load(), 0);

  allocator->Poll();
  EXPECT_TRUE(weak_owner.expired());

  auto borrowed =
      allocator->WrapExternal(stream, &fake_allocation_storage.at(1), 32, ExternalOwnership::BORROWED, nullptr);
  borrowed.reset();
  EXPECT_EQ(allocator->GetStats().pending_retirement_count_, 0);
  EXPECT_EQ(fake_cuda_state.free_count_.load(), 0);

  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, ValidatesExternalPointerAndOwnershipContracts) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  auto owner = std::make_shared<int>(1);

  EXPECT_THROW(
      {
        [[maybe_unused]] const auto storage =
            allocator->WrapExternal(stream, nullptr, 1, ExternalOwnership::BORROWED, nullptr);
      },
      InvalidArgumentError);
  EXPECT_THROW(
      {
        [[maybe_unused]] const auto storage =
            allocator->WrapExternal(stream, &fake_allocation_storage.front(), 1, ExternalOwnership::BORROWED, owner);
      },
      InvalidArgumentError);
  EXPECT_THROW(
      {
        [[maybe_unused]] const auto storage = allocator->WrapExternal(stream, &fake_allocation_storage.front(), 1,
                                                                      ExternalOwnership::SHARED_OWNER, nullptr);
      },
      InvalidArgumentError);

  fake_cuda_state.pointer_type_.store(cudaMemoryTypeManaged);
  EXPECT_THROW(
      {
        [[maybe_unused]] const auto storage = allocator->WrapExternal(stream, &fake_allocation_storage.front(), 1,
                                                                      ExternalOwnership::SHARED_OWNER, owner);
      },
      InvalidArgumentError);
  fake_cuda_state.pointer_type_.store(cudaMemoryTypeDevice);
  fake_cuda_state.pointer_device_.store(1);
  EXPECT_THROW(
      {
        [[maybe_unused]] const auto storage = allocator->WrapExternal(stream, &fake_allocation_storage.front(), 1,
                                                                      ExternalOwnership::SHARED_OWNER, owner);
      },
      InvalidArgumentError);

  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, RecoversPoisonedRetirementOnlyAtExplicitShutdown) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  auto storage = allocator->Allocate(stream, 32, 16, MakeContext());
  fake_cuda_state.create_event_status_.store(cudaErrorMemoryAllocation);

  storage.reset();
  EXPECT_EQ(allocator->GetStats().pending_retirement_count_, 1);
  EXPECT_GT(error_sink->GetReportCount(), 0);
  EXPECT_THROW(
      { [[maybe_unused]] const auto rejected = allocator->Allocate(stream, 1, 1, MakeContext()); },
      InvalidArgumentError);

  allocator->Shutdown();
  EXPECT_EQ(fake_cuda_state.free_count_.load(), 1);
  EXPECT_EQ(fake_cuda_state.synchronize_stream_count_.load(), 3);
}

TEST_F(DeviceAllocatorTest, RetainsExternalOwnerWhenCompletionEventAcquisitionFails) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  auto owner = std::make_shared<int>(7);
  const std::weak_ptr<int> weak_owner = owner;
  auto storage = allocator->WrapExternal(stream, &fake_allocation_storage.front(), 32, ExternalOwnership::SHARED_OWNER,
                                         std::move(owner));
  fake_cuda_state.create_event_status_.store(cudaErrorMemoryAllocation);

  storage.reset();
  EXPECT_FALSE(weak_owner.expired());
  EXPECT_EQ(allocator->GetStats().pending_retirement_count_, 1);

  allocator->Shutdown();
  EXPECT_TRUE(weak_owner.expired());
  EXPECT_EQ(fake_cuda_state.free_count_.load(), 0);
}

TEST_F(DeviceAllocatorTest, ConvertsCompletionQueryFailureIntoPoisonedRetirement) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  auto storage = allocator->Allocate(stream, 32, 16, MakeContext());
  storage.reset();
  fake_cuda_state.query_event_status_.store(cudaErrorInvalidValue);

  allocator->Poll();
  const auto poisoned_stats = allocator->GetStats();
  EXPECT_EQ(poisoned_stats.retiring_bytes_, 32);
  EXPECT_EQ(poisoned_stats.pending_retirement_count_, 1);
  EXPECT_GT(error_sink->GetReportCount(), 0);

  allocator->Shutdown();
  EXPECT_EQ(fake_cuda_state.free_count_.load(), 1);
}

TEST_F(DeviceAllocatorTest, RejectsShutdownUntilEveryStorageOwnerIsReleased) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  auto storage = allocator->Allocate(stream, 16, 16, MakeContext());

  EXPECT_THROW(allocator->Shutdown(), InvalidArgumentError);
  EXPECT_THROW(
      { [[maybe_unused]] const auto rejected = allocator->Allocate(stream, 1, 1, MakeContext()); },
      InvalidArgumentError);
  storage.reset();
  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, AppliesValidatedPeerAccessToThePrivatePool) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);

  allocator->SetPeerAccess(Device{1}, true);
  EXPECT_EQ(fake_cuda_state.set_pool_access_count_.load(), 1);
  EXPECT_EQ(fake_cuda_state.last_access_device_.load(), 1);
  EXPECT_EQ(fake_cuda_state.last_access_flags_.load(), cudaMemAccessFlagsProtReadWrite);

  fake_cuda_state.can_access_peer_.store(0);
  EXPECT_THROW(allocator->SetPeerAccess(Device{2}, true), NotSupportedError);
  EXPECT_THROW(allocator->SetPeerAccess(Device{0}, true), InvalidArgumentError);

  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, EnforcesBudgetUnderConcurrentAllocation) {
  constexpr size_t thread_count = 8;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink, DeviceAllocatorOptions{.max_live_bytes_ = 64});
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  std::mutex result_latch;
  std::vector<std::shared_ptr<Storage>> allocations;
  size_t oom_count = 0;
  std::vector<std::jthread> threads;
  threads.reserve(thread_count);

  for (size_t index = 0; index < thread_count; index++) {
    threads.emplace_back([&] {
      try {
        auto storage = allocator->Allocate(stream, 16, 16, MakeContext("concurrent"));
        std::scoped_lock lock{result_latch};
        allocations.push_back(std::move(storage));
      } catch (const OutOfMemoryError &) {
        std::scoped_lock lock{result_latch};
        oom_count++;
      }
    });
  }
  threads.clear();

  EXPECT_EQ(allocations.size(), 4);
  EXPECT_EQ(oom_count, 4);
  EXPECT_EQ(allocator->GetStats().logical_live_bytes_, 64);
  allocations.clear();
  allocator->Poll();
  allocator->Shutdown();
}

TEST_F(DeviceAllocatorTest, SerializesShutdownWithAnInFlightAllocation) {
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto allocator = MakeAllocator(error_sink);
  const auto stream = MakeExternalStream(0, Device{0}, error_sink);
  fake_cuda_state.block_malloc_.store(true, std::memory_order_release);

  auto allocation_future = std::async(
      std::launch::async, [&] { return allocator->Allocate(stream, 16, 16, MakeContext("in-flight allocation")); });
  while (!fake_cuda_state.malloc_entered_.load(std::memory_order_acquire)) {
    fake_cuda_state.malloc_entered_.wait(false, std::memory_order_relaxed);
  }

  std::promise<void> shutdown_started;
  auto shutdown_started_future = shutdown_started.get_future();
  auto shutdown_future = std::async(std::launch::async, [&] {
    shutdown_started.set_value();
    allocator->Shutdown();
  });
  shutdown_started_future.get();
  EXPECT_EQ(shutdown_future.wait_for(std::chrono::milliseconds{50}), std::future_status::timeout);

  fake_cuda_state.block_malloc_.store(false, std::memory_order_release);
  fake_cuda_state.block_malloc_.notify_all();
  auto storage = allocation_future.get();
  EXPECT_THROW(shutdown_future.get(), InvalidArgumentError);

  storage.reset();
  allocator->Shutdown();
}

TEST(DeviceAllocatorIntegrationTest, RetiresMemoryAfterWorkOnTwoCudaStreams) {
  int device_count = 0;
  const auto count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    GTEST_SKIP() << "A CUDA device is required";
  }

  constexpr size_t allocation_bytes = 4096;
  const auto error_sink = std::make_shared<RecordingErrorSink>();
  auto event_pool = std::make_shared<EventPool>(Device{0}, error_sink, 8);
  auto allocator = DeviceAllocator::Create(Device{0}, error_sink, event_pool,
                                           DeviceAllocatorOptions{.enable_maintenance_thread_ = false});
  const auto allocation_stream = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
  const auto side_stream = StreamAccess::CreateOwned(Device{0}, 0, error_sink);
  auto storage = allocator->Allocate(allocation_stream, allocation_bytes, 256, MakeContext("integration"));

  ASSERT_EQ(cudaMemsetAsync(storage->GetBasePointer(), 0, allocation_bytes, StreamAccess::GetNative(allocation_stream)),
            cudaSuccess);
  const auto allocation_ready = EventAccess::Record(allocation_stream);
  EventAccess::Wait(side_stream, allocation_ready);
  ASSERT_EQ(cudaMemsetAsync(storage->GetBasePointer(), 1, allocation_bytes, StreamAccess::GetNative(side_stream)),
            cudaSuccess);
  storage->RecordUsage(side_stream);
  storage.reset();

  allocator->Shutdown();
  event_pool->Close();
  EXPECT_EQ(error_sink->GetReportCount(), 0);
}

}  // namespace
}  // namespace ttl::internal
