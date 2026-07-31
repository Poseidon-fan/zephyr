#include "ttl/runtime.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <string>
#include <utility>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/cublas_api.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/parallel_op_scope.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl::internal {
namespace {

constexpr size_t FAKE_RESOURCE_COUNT = 128;
constexpr size_t FAKE_ALLOCATION_BYTES = 16U * 1024U;
constexpr size_t FAKE_HOST_ALLOCATION_BYTES = 256;
constexpr int FAKE_DEVICE_COUNT = 2;

struct alignas(256) FakeAllocation final {
  std::array<std::byte, FAKE_ALLOCATION_BYTES> bytes_;
};

struct alignas(16) FakeHostAllocation final {
  std::array<std::byte, FAKE_HOST_ALLOCATION_BYTES> bytes_;
};

struct FakeCublasHandle final {
  cudaStream_t stream_{nullptr};
  void *workspace_{nullptr};
  size_t workspace_bytes_{0};
  cublasPointerMode_t pointer_mode_{CUBLAS_POINTER_MODE_HOST};
};

std::array<FakeAllocation, FAKE_RESOURCE_COUNT> fake_allocations;
std::array<FakeHostAllocation, FAKE_RESOURCE_COUNT> fake_host_allocations;
std::array<FakeCublasHandle, FAKE_RESOURCE_COUNT> fake_cublas_handles;
std::array<int, FAKE_RESOURCE_COUNT> fake_cublas_lt_handles;
std::array<int, FAKE_RESOURCE_COUNT> fake_streams;
std::array<int, FAKE_RESOURCE_COUNT> fake_events;
std::array<int, FAKE_DEVICE_COUNT> fake_pools;
size_t fake_allocation_index = 0;
size_t fake_host_allocation_index = 0;
size_t fake_free_host_count = 0;
size_t fake_stream_index = 0;
size_t fake_event_index = 0;
size_t fake_create_stream_count = 0;
size_t fake_destroyed_stream_count = 0;
size_t fake_destroyed_event_count = 0;
size_t fake_destroyed_pool_count = 0;
size_t fake_pool_access_count = 0;
size_t fake_record_event_count = 0;
size_t fake_stream_wait_count = 0;
size_t fake_stream_synchronize_count = 0;
size_t fake_peer_copy_count = 0;
size_t fake_cublas_handle_index = 0;
size_t fake_destroyed_cublas_handle_count = 0;
size_t fake_cublas_lt_handle_index = 0;
size_t fake_destroyed_cublas_lt_handle_count = 0;
size_t fake_cublas_bind_count = 0;
std::optional<size_t> fake_record_event_failure_call;
std::optional<size_t> fake_create_stream_failure_call;
std::optional<size_t> fake_stream_wait_failure_call;
std::optional<size_t> fake_cublas_create_failure_call;
std::optional<size_t> fake_cublas_workspace_failure_call;
int fake_current_device = 0;
cudaError_t fake_peek_status = cudaSuccess;
cudaError_t fake_query_event_status = cudaSuccess;
cudaError_t fake_host_alloc_status = cudaSuccess;
cudaError_t fake_free_host_status = cudaSuccess;
cudaError_t fake_memset_status = cudaSuccess;

void ResetFakeCuda() {
  fake_allocation_index = 0;
  fake_host_allocation_index = 0;
  fake_free_host_count = 0;
  fake_stream_index = 0;
  fake_event_index = 0;
  fake_create_stream_count = 0;
  fake_destroyed_stream_count = 0;
  fake_destroyed_event_count = 0;
  fake_destroyed_pool_count = 0;
  fake_pool_access_count = 0;
  fake_record_event_count = 0;
  fake_stream_wait_count = 0;
  fake_stream_synchronize_count = 0;
  fake_peer_copy_count = 0;
  fake_cublas_handle_index = 0;
  fake_destroyed_cublas_handle_count = 0;
  fake_cublas_lt_handle_index = 0;
  fake_destroyed_cublas_lt_handle_count = 0;
  fake_cublas_bind_count = 0;
  fake_record_event_failure_call.reset();
  fake_create_stream_failure_call.reset();
  fake_stream_wait_failure_call.reset();
  fake_cublas_create_failure_call.reset();
  fake_cublas_workspace_failure_call.reset();
  fake_current_device = 0;
  fake_peek_status = cudaSuccess;
  fake_query_event_status = cudaSuccess;
  fake_host_alloc_status = cudaSuccess;
  fake_free_host_status = cudaSuccess;
  fake_memset_status = cudaSuccess;
}

auto FakeGetDeviceCount(int *count) -> cudaError_t {
  *count = FAKE_DEVICE_COUNT;
  return cudaSuccess;
}

auto FakeGetDeviceProperties(cudaDeviceProp *properties, int device) -> cudaError_t {
  if (device < 0 || device >= FAKE_DEVICE_COUNT) {
    return cudaErrorInvalidDevice;
  }
  *properties = cudaDeviceProp{};
  properties->name[0] = 'T';
  properties->name[1] = static_cast<char>('0' + device);
  properties->major = 8;
  properties->minor = device;
  properties->warpSize = 32;
  properties->memoryPoolsSupported = 1;
  properties->computeMode = cudaComputeModeDefault;
  properties->multiProcessorCount = 80 + device;
  properties->sharedMemPerBlockOptin = 98304;
  properties->clusterLaunch = 0;
  return cudaSuccess;
}

auto FakeGetDevice(int *device) -> cudaError_t {
  *device = fake_current_device;
  return cudaSuccess;
}

auto FakeSetDevice(int device) -> cudaError_t {
  if (device < 0 || device >= FAKE_DEVICE_COUNT) {
    return cudaErrorInvalidDevice;
  }
  fake_current_device = device;
  return cudaSuccess;
}

auto FakeGetLastError() -> cudaError_t { return cudaSuccess; }

auto FakePeekAtLastError() -> cudaError_t { return fake_peek_status; }

auto FakeGetStreamPriorityRange(int *least_priority, int *greatest_priority) -> cudaError_t {
  *least_priority = 0;
  *greatest_priority = -1;
  return cudaSuccess;
}

auto FakeCreateStream(cudaStream_t *stream, unsigned int flags, int priority) -> cudaError_t {
  const auto call = fake_create_stream_count;
  fake_create_stream_count++;
  if (flags != cudaStreamNonBlocking || priority < -1 || priority > 0 || fake_stream_index == fake_streams.size()) {
    return cudaErrorInvalidValue;
  }
  if (fake_create_stream_failure_call == call) {
    return cudaErrorMemoryAllocation;
  }
  *stream = reinterpret_cast<cudaStream_t>(&fake_streams[fake_stream_index]);
  fake_stream_index++;
  return cudaSuccess;
}

auto FakeDestroyStream(cudaStream_t /*stream*/) -> cudaError_t {
  fake_destroyed_stream_count++;
  return cudaSuccess;
}

auto FakeCreateEvent(cudaEvent_t *event, unsigned int flags) -> cudaError_t {
  if (flags != cudaEventDisableTiming || fake_event_index == fake_events.size()) {
    return cudaErrorInvalidValue;
  }
  *event = reinterpret_cast<cudaEvent_t>(&fake_events[fake_event_index]);
  fake_event_index++;
  return cudaSuccess;
}

auto FakeRecordEvent(cudaEvent_t /*event*/, cudaStream_t /*stream*/) -> cudaError_t {
  const auto call = fake_record_event_count;
  fake_record_event_count++;
  if (fake_record_event_failure_call == call) {
    return cudaErrorInvalidResourceHandle;
  }
  return cudaSuccess;
}

auto FakeQueryEvent(cudaEvent_t /*event*/) -> cudaError_t { return fake_query_event_status; }

auto FakeSynchronizeEvent(cudaEvent_t /*event*/) -> cudaError_t { return cudaSuccess; }

auto FakeDestroyEvent(cudaEvent_t /*event*/) -> cudaError_t {
  fake_destroyed_event_count++;
  return cudaSuccess;
}

auto FakeStreamWaitEvent(cudaStream_t /*stream*/, cudaEvent_t /*event*/, unsigned int flags) -> cudaError_t {
  if (flags != cudaEventWaitDefault) {
    return cudaErrorInvalidValue;
  }
  const auto call = fake_stream_wait_count;
  fake_stream_wait_count++;
  if (fake_stream_wait_failure_call == call) {
    return cudaErrorInvalidResourceHandle;
  }
  return cudaSuccess;
}

auto FakeSynchronizeStream(cudaStream_t /*stream*/) -> cudaError_t {
  fake_stream_synchronize_count++;
  return cudaSuccess;
}

auto FakeCreatePool(cudaMemPool_t *pool, const cudaMemPoolProps *properties) -> cudaError_t {
  if (properties->location.type != cudaMemLocationTypeDevice || properties->location.id < 0 ||
      properties->location.id >= FAKE_DEVICE_COUNT) {
    return cudaErrorInvalidValue;
  }
  *pool = reinterpret_cast<cudaMemPool_t>(&fake_pools[static_cast<size_t>(properties->location.id)]);
  return cudaSuccess;
}

auto FakeDestroyPool(cudaMemPool_t /*pool*/) -> cudaError_t {
  fake_destroyed_pool_count++;
  return cudaSuccess;
}

auto FakeSetPoolAttribute(cudaMemPool_t /*pool*/, cudaMemPoolAttr /*attribute*/, void * /*value*/) -> cudaError_t {
  return cudaSuccess;
}

auto FakeGetPoolAttribute(cudaMemPool_t /*pool*/, cudaMemPoolAttr /*attribute*/, void *value) -> cudaError_t {
  *static_cast<uint64_t *>(value) = 0;
  return cudaSuccess;
}

auto FakeSetPoolAccess(cudaMemPool_t /*pool*/, const cudaMemAccessDesc *description, size_t count) -> cudaError_t {
  if (description == nullptr || count != 1 || description->location.type != cudaMemLocationTypeDevice ||
      description->flags != cudaMemAccessFlagsProtReadWrite) {
    return cudaErrorInvalidValue;
  }
  fake_pool_access_count++;
  return cudaSuccess;
}

auto FakeTrimPool(cudaMemPool_t /*pool*/, size_t /*minimum_bytes*/) -> cudaError_t { return cudaSuccess; }

auto FakeMallocFromPool(void **pointer, size_t bytes, cudaMemPool_t /*pool*/, cudaStream_t /*stream*/) -> cudaError_t {
  if (bytes > FAKE_ALLOCATION_BYTES || fake_allocation_index == fake_allocations.size()) {
    return cudaErrorMemoryAllocation;
  }
  *pointer = fake_allocations[fake_allocation_index].bytes_.data();
  fake_allocation_index++;
  return cudaSuccess;
}

auto FakeFreeAsync(void * /*pointer*/, cudaStream_t /*stream*/) -> cudaError_t { return cudaSuccess; }

auto FakeMemsetAsync(void *pointer, int value, size_t bytes, cudaStream_t /*stream*/) -> cudaError_t {
  if (fake_memset_status != cudaSuccess) {
    return fake_memset_status;
  }
  std::memset(pointer, value, bytes);
  return cudaSuccess;
}

auto FakeMemcpyAsync(void *destination, const void *source, size_t bytes, cudaMemcpyKind /*kind*/,
                     cudaStream_t /*stream*/) -> cudaError_t {
  std::memcpy(destination, source, bytes);
  return cudaSuccess;
}

auto FakeMemcpyPeerAsync(void *destination, int destination_device, const void *source, int source_device, size_t bytes,
                         cudaStream_t /*stream*/) -> cudaError_t {
  if (destination_device != 0 || source_device != 1 || fake_current_device != destination_device) {
    return cudaErrorInvalidDevice;
  }
  std::memcpy(destination, source, bytes);
  fake_peer_copy_count++;
  return cudaSuccess;
}

auto FakeHostAlloc(void **pointer, size_t bytes, unsigned int flags) -> cudaError_t {
  if (fake_host_alloc_status != cudaSuccess) {
    return fake_host_alloc_status;
  }
  if (bytes > FAKE_HOST_ALLOCATION_BYTES || flags != cudaHostAllocPortable ||
      fake_host_allocation_index == fake_host_allocations.size()) {
    return cudaErrorMemoryAllocation;
  }
  *pointer = fake_host_allocations[fake_host_allocation_index].bytes_.data();
  fake_host_allocation_index++;
  return cudaSuccess;
}

auto FakeFreeHost(void * /*pointer*/) -> cudaError_t {
  fake_free_host_count++;
  return fake_free_host_status;
}

auto FakeGetMemoryInfo(size_t *free_bytes, size_t *total_bytes) -> cudaError_t {
  *free_bytes = 1U << 20U;
  *total_bytes = 2U << 20U;
  return cudaSuccess;
}

auto FakeGetPointerAttributes(cudaPointerAttributes *attributes, const void * /*pointer*/) -> cudaError_t {
  attributes->type = cudaMemoryTypeDevice;
  attributes->device = fake_current_device;
  return cudaSuccess;
}

auto FakeCanAccessPeer(int *can_access, int source, int destination) -> cudaError_t {
  if (source < 0 || source >= FAKE_DEVICE_COUNT || destination < 0 || destination >= FAKE_DEVICE_COUNT) {
    return cudaErrorInvalidDevice;
  }
  *can_access = source == 0 && destination == 1 ? 1 : 0;
  return cudaSuccess;
}

auto FakeCublasCreate(cublasHandle_t *handle) -> cublasStatus_t {
  const auto call = fake_cublas_handle_index;
  if (fake_cublas_create_failure_call == call) {
    return CUBLAS_STATUS_ALLOC_FAILED;
  }
  if (fake_cublas_handle_index == fake_cublas_handles.size()) {
    return CUBLAS_STATUS_ALLOC_FAILED;
  }
  *handle = reinterpret_cast<cublasHandle_t>(&fake_cublas_handles[fake_cublas_handle_index]);
  fake_cublas_handle_index++;
  return CUBLAS_STATUS_SUCCESS;
}

auto FakeCublasDestroy(cublasHandle_t /*handle*/) -> cublasStatus_t {
  fake_destroyed_cublas_handle_count++;
  return CUBLAS_STATUS_SUCCESS;
}

auto FakeCublasLtCreate(cublasLtHandle_t *handle) -> cublasStatus_t {
  if (fake_cublas_lt_handle_index == fake_cublas_lt_handles.size()) {
    return CUBLAS_STATUS_ALLOC_FAILED;
  }
  *handle = reinterpret_cast<cublasLtHandle_t>(&fake_cublas_lt_handles[fake_cublas_lt_handle_index]);
  fake_cublas_lt_handle_index++;
  return CUBLAS_STATUS_SUCCESS;
}

auto FakeCublasLtDestroy(cublasLtHandle_t /*handle*/) -> cublasStatus_t {
  fake_destroyed_cublas_lt_handle_count++;
  return CUBLAS_STATUS_SUCCESS;
}

auto FakeCublasSetStream(cublasHandle_t handle, cudaStream_t stream) -> cublasStatus_t {
  auto &state = *reinterpret_cast<FakeCublasHandle *>(handle);
  state.stream_ = stream;
  state.workspace_ = nullptr;
  state.workspace_bytes_ = 0;
  fake_cublas_bind_count++;
  return CUBLAS_STATUS_SUCCESS;
}

auto FakeCublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) -> cublasStatus_t {
  reinterpret_cast<FakeCublasHandle *>(handle)->pointer_mode_ = mode;
  return mode == CUBLAS_POINTER_MODE_DEVICE ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

auto FakeCublasSetWorkspace(cublasHandle_t handle, void *workspace, size_t workspace_bytes) -> cublasStatus_t {
  const auto call = fake_cublas_bind_count - 1;
  if (fake_cublas_workspace_failure_call == call) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (workspace == nullptr || reinterpret_cast<uintptr_t>(workspace) % 256 != 0 ||
      workspace_bytes != FAKE_ALLOCATION_BYTES) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  auto &state = *reinterpret_cast<FakeCublasHandle *>(handle);
  state.workspace_ = workspace;
  state.workspace_bytes_ = workspace_bytes;
  return CUBLAS_STATUS_SUCCESS;
}

[[nodiscard]] auto MakeFakeCudaApi() -> CudaApi {
  auto cuda_api = GetCudaApi();
  cuda_api.get_device_count_ = FakeGetDeviceCount;
  cuda_api.get_device_properties_ = FakeGetDeviceProperties;
  cuda_api.get_device_ = FakeGetDevice;
  cuda_api.set_device_ = FakeSetDevice;
  cuda_api.get_last_error_ = FakeGetLastError;
  cuda_api.peek_at_last_error_ = FakePeekAtLastError;
  cuda_api.get_stream_priority_range_ = FakeGetStreamPriorityRange;
  cuda_api.create_stream_with_priority_ = FakeCreateStream;
  cuda_api.destroy_stream_ = FakeDestroyStream;
  cuda_api.create_event_with_flags_ = FakeCreateEvent;
  cuda_api.record_event_ = FakeRecordEvent;
  cuda_api.query_event_ = FakeQueryEvent;
  cuda_api.synchronize_event_ = FakeSynchronizeEvent;
  cuda_api.destroy_event_ = FakeDestroyEvent;
  cuda_api.stream_wait_event_ = FakeStreamWaitEvent;
  cuda_api.synchronize_stream_ = FakeSynchronizeStream;
  cuda_api.create_memory_pool_ = FakeCreatePool;
  cuda_api.destroy_memory_pool_ = FakeDestroyPool;
  cuda_api.set_memory_pool_attribute_ = FakeSetPoolAttribute;
  cuda_api.get_memory_pool_attribute_ = FakeGetPoolAttribute;
  cuda_api.set_memory_pool_access_ = FakeSetPoolAccess;
  cuda_api.trim_memory_pool_ = FakeTrimPool;
  cuda_api.malloc_from_pool_async_ = FakeMallocFromPool;
  cuda_api.free_async_ = FakeFreeAsync;
  cuda_api.memset_async_ = FakeMemsetAsync;
  cuda_api.memcpy_async_ = FakeMemcpyAsync;
  cuda_api.memcpy_peer_async_ = FakeMemcpyPeerAsync;
  cuda_api.host_alloc_ = FakeHostAlloc;
  cuda_api.free_host_ = FakeFreeHost;
  cuda_api.get_memory_info_ = FakeGetMemoryInfo;
  cuda_api.get_pointer_attributes_ = FakeGetPointerAttributes;
  cuda_api.can_access_peer_ = FakeCanAccessPeer;
  return cuda_api;
}

[[nodiscard]] auto MakeFakeCublasApi() -> CublasApi {
  auto cublas_api = GetCublasApi();
  cublas_api.create_ = FakeCublasCreate;
  cublas_api.destroy_ = FakeCublasDestroy;
  cublas_api.lt_create_ = FakeCublasLtCreate;
  cublas_api.lt_destroy_ = FakeCublasLtDestroy;
  cublas_api.set_stream_ = FakeCublasSetStream;
  cublas_api.set_pointer_mode_ = FakeCublasSetPointerMode;
  cublas_api.set_workspace_ = FakeCublasSetWorkspace;
  return cublas_api;
}

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override {
    try {
      std::scoped_lock lock{latch_};
      records_.push_back(std::move(error));
    } catch (...) {
      return;
    }
  }

  [[nodiscard]] auto GetRecords() const -> std::vector<ErrorRecord> {
    std::scoped_lock lock{latch_};
    return records_;
  }

 private:
  mutable std::mutex latch_;
  std::vector<ErrorRecord> records_;
};

[[nodiscard]] auto MakeRuntimeOptions(const std::shared_ptr<ErrorSink> &error_sink,
                                      std::vector<Device> devices = {Device{0}, Device{1}}) -> RuntimeOptions {
  return RuntimeOptions{
      .devices_ = std::move(devices),
      .device_memory_ =
          {
              .enable_maintenance_thread_ = false,
          },
      .blas_workspace_bytes_ = FAKE_ALLOCATION_BYTES,
      .event_pool_capacity_per_device_ = 8,
      .event_pool_reserve_per_device_ = 2,
      .error_sink_ = error_sink,
  };
}

class RuntimeTest : public testing::Test {
 protected:
  void SetUp() override { ResetFakeCuda(); }

  CudaApi cuda_api_{MakeFakeCudaApi()};
  ScopedCudaApiOverride cuda_api_override_{cuda_api_};
  CublasApi cublas_api_{MakeFakeCublasApi()};
  ScopedCublasApiOverride cublas_api_override_{cublas_api_};
};

TEST_F(RuntimeTest, RejectsInvalidOptionsBeforePublishingResources) {
  auto error_sink = std::make_shared<RecordingErrorSink>();

  EXPECT_THROW([[maybe_unused]] Runtime runtime{MakeRuntimeOptions(error_sink, {})}, InvalidArgumentError);

  auto null_sink_options = MakeRuntimeOptions(error_sink, {Device{0}});
  null_sink_options.error_sink_.reset();
  EXPECT_THROW([[maybe_unused]] Runtime runtime{std::move(null_sink_options)}, InvalidArgumentError);

  EXPECT_THROW([[maybe_unused]] Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}, Device{0}})},
               InvalidArgumentError);

  auto event_options = MakeRuntimeOptions(error_sink, {Device{0}});
  event_options.event_pool_capacity_per_device_ = 1;
  event_options.event_pool_reserve_per_device_ = 2;
  EXPECT_THROW([[maybe_unused]] Runtime runtime{std::move(event_options)}, InvalidArgumentError);

  auto memory_options = MakeRuntimeOptions(error_sink, {Device{0}});
  memory_options.device_memory_.max_live_bytes_ = 1024;
  memory_options.device_memory_.max_reserved_bytes_ = 512;
  EXPECT_THROW([[maybe_unused]] Runtime runtime{std::move(memory_options)}, InvalidArgumentError);

  auto blas_options = MakeRuntimeOptions(error_sink, {Device{0}});
  blas_options.blas_workspace_bytes_ = 4096;
  EXPECT_THROW([[maybe_unused]] Runtime runtime{std::move(blas_options)}, InvalidArgumentError);

  EXPECT_EQ(fake_stream_index, 0);
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, OwnsPerDeviceServicesAndCoordinatesStreams) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink)};

  ASSERT_EQ(runtime.GetDevices().size(), 2);
  EXPECT_EQ(runtime.GetDevices()[0], Device{0});
  EXPECT_EQ(runtime.GetDeviceProperties(Device{1}).multiprocessor_count_, 81);
  EXPECT_TRUE(runtime.CanAccessPeer(Device{0}, Device{0}));
  EXPECT_TRUE(runtime.CanAccessPeer(Device{0}, Device{1}));
  EXPECT_FALSE(runtime.CanAccessPeer(Device{1}, Device{0}));
  EXPECT_EQ(fake_pool_access_count, 1);

  {
    auto producer = runtime.CreateExecutionContext(Device{0});
    auto consumer = runtime.CreateExecutionContext(Device{1}, ExecutionContextOptions{.stream_priority_ = -1});
    EXPECT_FALSE(producer.IsExternalStream());
    EXPECT_NE(producer.GetStream().GetId(), consumer.GetStream().GetId());

    const auto event = producer.RecordEvent();
    consumer.Wait(event);
    consumer.Synchronize();

    EXPECT_EQ(fake_stream_wait_count, 1);
    EXPECT_GE(fake_stream_synchronize_count, 1);
  }

  runtime.Shutdown();
  EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::CLOSED);
  EXPECT_EQ(fake_destroyed_pool_count, 2);
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, CopiesAcrossDevicesWithAnExplicitProducerDependency) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink)};

  {
    auto destination_context = runtime.CreateExecutionContext(Device{0});
    auto source_context = runtime.CreateExecutionContext(Device{1});
    auto source = Empty(source_context, Shape{4}, DType::INT32);
    auto destination = Empty(destination_context, Shape{4}, DType::INT32);
    constexpr std::array<int32_t, 4> values{3, 1, 4, 1};
    std::memcpy(TensorAccess::GetMutableData(source), values.data(), sizeof(values));

    const auto source_ready = source_context.RecordEvent();
    CopyPeerOut(destination_context, destination, source, source_ready);

    std::array<int32_t, 4> result{};
    std::memcpy(result.data(), TensorAccess::GetData(destination), sizeof(result));
    EXPECT_EQ(result, values);
    EXPECT_EQ(fake_peer_copy_count, 1);
    EXPECT_EQ(fake_stream_wait_count, 1);

    const auto wrong_ready = destination_context.RecordEvent();
    EXPECT_THROW(CopyPeerOut(destination_context, destination, source, wrong_ready), InvalidArgumentError);

    auto unsupported_source = Empty(destination_context, Shape{4}, DType::INT32);
    auto unsupported_destination = Empty(source_context, Shape{4}, DType::INT32);
    const auto unsupported_ready = destination_context.RecordEvent();
    EXPECT_THROW(CopyPeerOut(source_context, unsupported_destination, unsupported_source, unsupported_ready),
                 NotSupportedError);
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, WrapsExternalStreamsWithoutTakingNativeOwnership) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};
  auto owner_release_count = std::make_shared<std::atomic<size_t>>(0);

  {
    auto owner = std::shared_ptr<void>{&fake_streams.back(), [owner_release_count](void * /*pointer*/) noexcept {
                                         owner_release_count->fetch_add(1, std::memory_order_relaxed);
                                       }};
    auto context =
        runtime.WrapExternalStream(Device{0}, reinterpret_cast<cudaStream_t>(&fake_streams.back()), std::move(owner));
    EXPECT_TRUE(context.IsExternalStream());
    EXPECT_EQ(context.GetDevice(), Device{0});
    EXPECT_EQ(owner_release_count->load(std::memory_order_relaxed), 0);
  }

  EXPECT_EQ(owner_release_count->load(std::memory_order_relaxed), 0);
  runtime.Poll();
  EXPECT_EQ(owner_release_count->load(std::memory_order_relaxed), 1);
  EXPECT_EQ(fake_destroyed_stream_count, 0);
  EXPECT_THROW([[maybe_unused]] auto context = runtime.WrapExternalStream(Device{0}, cudaStreamLegacy),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] auto context = runtime.CreateExecutionContext(Device{1}), InvalidArgumentError);

  runtime.Shutdown();
  EXPECT_EQ(fake_destroyed_stream_count, 1);
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, ChecksAndReportsDeviceErrorResourceFailures) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  fake_host_alloc_status = cudaErrorMemoryAllocation;
  EXPECT_THROW([[maybe_unused]] auto context = runtime.CreateExecutionContext(Device{0}), CudaError);
  EXPECT_EQ(fake_free_host_count, 0);

  fake_host_alloc_status = cudaSuccess;
  fake_memset_status = cudaErrorInvalidValue;
  EXPECT_THROW([[maybe_unused]] auto context = runtime.CreateExecutionContext(Device{0}), CudaError);
  EXPECT_EQ(fake_free_host_count, 1);

  fake_memset_status = cudaSuccess;
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    fake_free_host_status = cudaErrorInvalidValue;
  }
  fake_free_host_status = cudaSuccess;

  const auto records = error_sink->GetRecords();
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0].code_, ErrorCode::CUDA);
  EXPECT_NE(records[0].message_.find("cudaFreeHost"), std::string::npos);

  runtime.Poll();
  runtime.Shutdown();
}

TEST_F(RuntimeTest, CreatesValidatedContiguousAndDenseTensors) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(Device{0});
    {
      const auto contiguous = Empty(context, Shape{2, 3}, DType::FLOAT32);
      EXPECT_EQ(contiguous.GetShape(), Shape({2, 3}));
      EXPECT_EQ(contiguous.GetStrides(), Strides({3, 1}));
      EXPECT_TRUE(contiguous.IsContiguous());

      const auto transposed_dense = EmptyStrided(context, Shape{2, 3}, Strides{1, 2}, DType::FLOAT16);
      EXPECT_FALSE(transposed_dense.IsContiguous());
      EXPECT_TRUE(transposed_dense.IsNonOverlappingDense());

      EXPECT_THROW(
          [[maybe_unused]] const auto tensor = EmptyStrided(context, Shape{2, 2}, Strides{1, 1}, DType::FLOAT32),
          InvalidArgumentError);
      EXPECT_THROW([[maybe_unused]] const auto tensor = EmptyStrided(context, Shape{2, 2}, Strides{1}, DType::FLOAT32),
                   InvalidArgumentError);
    }
    context.Poll();
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, WrapsExternalMemoryWithExplicitContextAndOwnership) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};
  auto release_count = std::make_shared<std::atomic<size_t>>(0);

  {
    auto context = runtime.CreateExecutionContext(Device{0});
    {
      auto owner = std::shared_ptr<void>{
          fake_allocations.back().bytes_.data(),
          [release_count](void * /*pointer*/) noexcept { release_count->fetch_add(1, std::memory_order_relaxed); }};
      const auto tensor = runtime.FromBlob(context,
                                           ExternalMemory{
                                               .pointer_ = fake_allocations.back().bytes_.data(),
                                               .capacity_bytes_ = 64,
                                               .device_ = Device{0},
                                               .owner_ = std::move(owner),
                                           },
                                           Shape{4}, Strides{1}, DType::FLOAT32);
      EXPECT_EQ(tensor.GetData<float>(), reinterpret_cast<const float *>(fake_allocations.back().bytes_.data()));
      EXPECT_EQ(release_count->load(std::memory_order_relaxed), 0);
    }
    context.Poll();
    EXPECT_EQ(release_count->load(std::memory_order_relaxed), 1);

    EXPECT_THROW(
        [[maybe_unused]] const auto tensor = runtime.FromBlob(context,
                                                              ExternalMemory{
                                                                  .pointer_ = fake_allocations.back().bytes_.data(),
                                                                  .capacity_bytes_ = 64,
                                                                  .device_ = Device{1},
                                                                  .owner_ = nullptr,
                                                              },
                                                              Shape{4}, Strides{1}, DType::FLOAT32),
        InvalidArgumentError);
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, RejectsExecutionContextsFromAnotherRuntime) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime first_runtime{MakeRuntimeOptions(error_sink, {Device{0}})};
  Runtime second_runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto foreign_context = second_runtime.CreateExecutionContext(Device{0});
    EXPECT_THROW([[maybe_unused]] const auto tensor = first_runtime.FromBlob(foreign_context,
                                                                             ExternalMemory{
                                                                                 .pointer_ = nullptr,
                                                                                 .capacity_bytes_ = 0,
                                                                                 .device_ = Device{0},
                                                                                 .owner_ = nullptr,
                                                                             },
                                                                             Shape{0}, Strides{1}, DType::FLOAT32),
                 InvalidArgumentError);
  }

  first_runtime.Shutdown();
  second_runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, ShutdownRejectsLiveContextsAndAllowsExplicitCleanup) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(Device{0});
    EXPECT_THROW(runtime.Shutdown(), InvalidArgumentError);
    EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::CLOSING);
    EXPECT_THROW([[maybe_unused]] const auto event = context.RecordEvent(), InvalidArgumentError);
    EXPECT_NO_THROW(context.Synchronize());
  }

  EXPECT_NO_THROW(runtime.Shutdown());
  EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::CLOSED);
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, ShutdownRejectsLiveTensorStorageAndCompletesAfterRelease) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};
  std::optional<Tensor> tensor;

  {
    auto context = runtime.CreateExecutionContext(Device{0});
    tensor.emplace(Empty(context, Shape{4}, DType::FLOAT32));
  }

  EXPECT_THROW(runtime.Shutdown(), InvalidArgumentError);
  EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::CLOSING);
  tensor.reset();
  EXPECT_NO_THROW(runtime.Shutdown());
  EXPECT_EQ(runtime.GetStatus(), RuntimeStatus::CLOSED);
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, DetectsConcurrentContextEntryAndChecksKernelLaunches) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(Device{0});
    {
      ContextUseGuard use_guard{context, ContextUseMode::SUBMIT, std::source_location::current()};
      EXPECT_THROW([[maybe_unused]] const auto event = context.RecordEvent(), InvalidArgumentError);
    }

    const auto tensor = Empty(context, Shape{1}, DType::FLOAT32);
    {
      OpGuard op_guard{context, "test kernel"};
      op_guard.RecordTensor(tensor);
      EXPECT_NO_THROW(op_guard.CheckLaunch());
      fake_peek_status = cudaErrorInvalidConfiguration;
      EXPECT_THROW(op_guard.CheckLaunch(), CudaError);
      fake_peek_status = cudaSuccess;
    }
    EXPECT_THROW([[maybe_unused]] OpGuard op_guard(context, ""), InvalidArgumentError);
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, ForksAndJoinsContextPrivateAuxiliaryStreams) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(
        Device{0}, ExecutionContextOptions{.stream_priority_ = -1, .max_auxiliary_stream_count_ = 2});
    EXPECT_EQ(context.GetAuxiliaryStreamCount(), 2);

    {
      const auto tensor = Empty(context, Shape{4}, DType::FLOAT32);
      OpGuard op_guard{context, "parallel test"};
      op_guard.RecordTensor(tensor);
      ParallelOpScope parallel{op_guard, 2};

      EXPECT_NE(parallel.GetNativeAuxiliaryStream(0), op_guard.GetNativeStream());
      EXPECT_NE(parallel.GetNativeAuxiliaryStream(1), op_guard.GetNativeStream());
      EXPECT_NE(parallel.GetNativeAuxiliaryStream(0), parallel.GetNativeAuxiliaryStream(1));
      EXPECT_NE(op_guard.GetCublasHandle(), parallel.GetAuxiliaryCublasHandle(0));
      EXPECT_NE(parallel.GetAuxiliaryCublasHandle(0), parallel.GetAuxiliaryCublasHandle(1));
      EXPECT_NE(op_guard.GetCublasLtHandle(), nullptr);
      EXPECT_NE(reinterpret_cast<void *>(op_guard.GetCublasLtHandle()),
                reinterpret_cast<void *>(op_guard.GetCublasHandle()));
      EXPECT_EQ(op_guard.GetBlasWorkspaceBytes(), FAKE_ALLOCATION_BYTES);
      EXPECT_EQ(parallel.GetAuxiliaryBlasWorkspaceBytes(0), FAKE_ALLOCATION_BYTES);
      EXPECT_NE(op_guard.GetBlasWorkspace(), parallel.GetAuxiliaryBlasWorkspace(0));
      parallel.RecordTensor(tensor, 0);
      parallel.RecordTensor(tensor, 1);
      parallel.CheckLaunch();

      EXPECT_EQ(fake_record_event_count, 1);
      EXPECT_EQ(fake_stream_wait_count, 2);
      parallel.Finish();
      EXPECT_EQ(fake_record_event_count, 3);
      EXPECT_EQ(fake_stream_wait_count, 4);
    }

    const auto event_count = fake_event_index;
    {
      OpGuard op_guard{context, "parallel event reuse"};
      ParallelOpScope parallel{op_guard, 1};
      parallel.Finish();
    }
    EXPECT_EQ(fake_event_index, event_count);
    context.Synchronize();
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, ReusesBlasResourcesOnlyAfterStreamCompletion) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  cublasHandle_t first_handle = nullptr;
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    OpGuard op_guard{context, "first cuBLAS lease"};
    first_handle = op_guard.GetCublasHandle();
  }

  EXPECT_EQ(fake_cublas_handle_index, 1);
  runtime.Poll();
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    OpGuard op_guard{context, "reused cuBLAS lease"};
    EXPECT_EQ(op_guard.GetCublasHandle(), first_handle);
  }

  EXPECT_EQ(fake_cublas_handle_index, 1);
  EXPECT_EQ(fake_cublas_bind_count, 2);
  EXPECT_EQ(fake_destroyed_cublas_handle_count, 0);
  runtime.Shutdown();
  EXPECT_EQ(fake_destroyed_cublas_handle_count, 1);
  EXPECT_EQ(fake_destroyed_cublas_lt_handle_count, 1);
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, DoesNotReuseBlasResourcesBeforeStreamCompletion) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  fake_query_event_status = cudaErrorNotReady;
  {
    auto first_context = runtime.CreateExecutionContext(Device{0});
    OpGuard op_guard{first_context, "first incomplete cuBLAS lease"};
    EXPECT_NE(op_guard.GetCublasHandle(), nullptr);
  }
  {
    auto second_context = runtime.CreateExecutionContext(Device{0});
    OpGuard op_guard{second_context, "second incomplete cuBLAS lease"};
    EXPECT_NE(op_guard.GetCublasHandle(), nullptr);
  }
  EXPECT_EQ(fake_cublas_handle_index, 2);

  fake_query_event_status = cudaSuccess;
  runtime.Poll();
  {
    auto third_context = runtime.CreateExecutionContext(Device{0});
    OpGuard op_guard{third_context, "completed cuBLAS lease"};
    EXPECT_NE(op_guard.GetCublasHandle(), nullptr);
  }
  EXPECT_EQ(fake_cublas_handle_index, 2);

  runtime.Shutdown();
  EXPECT_EQ(fake_destroyed_cublas_handle_count, 2);
  EXPECT_EQ(fake_destroyed_cublas_lt_handle_count, 2);
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, RollsBackCublasCreationAndBindingFailures) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  fake_cublas_create_failure_call = 0;
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    OpGuard op_guard{context, "failed cublasCreate"};
    EXPECT_THROW([[maybe_unused]] const auto handle = op_guard.GetCublasHandle(), CublasError);
  }
  fake_cublas_create_failure_call.reset();

  fake_cublas_workspace_failure_call = 0;
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    OpGuard op_guard{context, "failed cublasSetWorkspace"};
    EXPECT_THROW([[maybe_unused]] const auto handle = op_guard.GetCublasHandle(), CublasError);
  }
  fake_cublas_workspace_failure_call.reset();
  runtime.Poll();

  EXPECT_EQ(fake_destroyed_cublas_handle_count, 1);
  {
    auto context = runtime.CreateExecutionContext(Device{0});
    OpGuard op_guard{context, "successful cublas binding"};
    EXPECT_NE(op_guard.GetCublasHandle(), nullptr);
  }
  runtime.Shutdown();
  EXPECT_EQ(fake_destroyed_cublas_handle_count, 2);
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, RollsBackPartiallyConstructedAuxiliaryStreams) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  fake_create_stream_failure_call = fake_create_stream_count + 2;
  EXPECT_THROW([[maybe_unused]] auto context =
                   runtime.CreateExecutionContext(Device{0}, ExecutionContextOptions{.max_auxiliary_stream_count_ = 2}),
               CudaError);
  runtime.Poll();
  EXPECT_EQ(fake_destroyed_stream_count, 2);

  {
    auto context = runtime.CreateExecutionContext(Device{0}, ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
    EXPECT_EQ(context.GetAuxiliaryStreamCount(), 1);
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, RejectsInvalidAndNestedParallelScopesWithoutPoisoningContext) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(Device{0}, ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
    {
      OpGuard op_guard{context, "invalid parallel scope"};
      EXPECT_THROW([[maybe_unused]] ParallelOpScope parallel(op_guard, 0), InvalidArgumentError);
      EXPECT_THROW([[maybe_unused]] ParallelOpScope parallel(op_guard, 2), InvalidArgumentError);
    }

    {
      OpGuard op_guard{context, "nested parallel scope"};
      ParallelOpScope parallel{op_guard, 1};
      EXPECT_THROW([[maybe_unused]] ParallelOpScope nested(op_guard, 1), InvalidArgumentError);
      parallel.Finish();
    }

    EXPECT_NO_THROW([[maybe_unused]] const auto event = context.RecordEvent());
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, FailedParallelForkPoisonsContextWithoutPublishingScope) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(Device{0}, ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
    {
      OpGuard op_guard{context, "failing parallel fork"};
      fake_record_event_failure_call = fake_record_event_count;
      EXPECT_THROW([[maybe_unused]] ParallelOpScope parallel(op_guard, 1), CudaError);
    }

    EXPECT_THROW([[maybe_unused]] const auto event = context.RecordEvent(), InvalidArgumentError);
    const auto synchronize_count = fake_stream_synchronize_count;
    EXPECT_NO_THROW(context.Synchronize());
    EXPECT_EQ(fake_stream_synchronize_count - synchronize_count, 3);
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, FailedParallelJoinPoisonsContextAndSynchronizesEveryStream) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(Device{0}, ExecutionContextOptions{.max_auxiliary_stream_count_ = 2});
    {
      OpGuard op_guard{context, "failing parallel join"};
      ParallelOpScope parallel{op_guard, 2};
      fake_stream_wait_failure_call = fake_stream_wait_count;
      EXPECT_THROW(parallel.Finish(), CudaError);
    }

    EXPECT_THROW([[maybe_unused]] const auto event = context.RecordEvent(), InvalidArgumentError);
    const auto synchronize_count = fake_stream_synchronize_count;
    EXPECT_NO_THROW(context.Synchronize());
    EXPECT_EQ(fake_stream_synchronize_count - synchronize_count, 4);
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST_F(RuntimeTest, UnfinishedParallelScopeReportsAndPoisonsContext) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(Device{0}, ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
    {
      OpGuard op_guard{context, "unfinished parallel test"};
      [[maybe_unused]] ParallelOpScope parallel{op_guard, 1};
    }

    EXPECT_THROW([[maybe_unused]] const auto event = context.RecordEvent(), InvalidArgumentError);
    EXPECT_NO_THROW(context.Synchronize());
  }

  runtime.Shutdown();
  const auto records = error_sink->GetRecords();
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0].code_, ErrorCode::INTERNAL);
  EXPECT_NE(records[0].message_.find("without Finish"), std::string::npos);
}

TEST_F(RuntimeTest, ReportsMissingExplicitShutdownWithoutInvalidatingChildren) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  std::optional<ExecutionContext> context;
  {
    Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};
    context.emplace(runtime.CreateExecutionContext(Device{0}));
  }

  ASSERT_EQ(error_sink->GetRecords().size(), 1);
  EXPECT_NE(error_sink->GetRecords()[0].message_.find("without a successful Shutdown"), std::string::npos);
  EXPECT_NO_THROW(context->Synchronize());
  context.reset();
}

TEST(RuntimeIntegrationTest, AllocatesSubmitsAndSynchronizesRealCudaWork) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(Device{0});
    {
      auto tensor = Empty(context, Shape{256}, DType::FLOAT32);
      OpGuard op_guard{context, "cudaMemsetAsync"};
      op_guard.RecordTensor(tensor);
      ASSERT_EQ(cudaMemsetAsync(TensorAccess::GetMutableData<float>(tensor), 0, 256 * sizeof(float),
                                op_guard.GetNativeStream()),
                cudaSuccess);
      op_guard.CheckLaunch();
    }
    context.Synchronize();
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

TEST(RuntimeIntegrationTest, ExecutesOneOperatorAcrossPrimaryAndAuxiliaryStreams) {
  auto error_sink = std::make_shared<RecordingErrorSink>();
  Runtime runtime{MakeRuntimeOptions(error_sink, {Device{0}})};

  {
    auto context = runtime.CreateExecutionContext(Device{0}, ExecutionContextOptions{.max_auxiliary_stream_count_ = 1});
    auto tensor = Empty(context, Shape{256}, DType::UINT8);
    auto *data = TensorAccess::GetMutableData<uint8_t>(tensor);

    {
      OpGuard op_guard{context, "parallel cudaMemsetAsync"};
      op_guard.RecordTensor(tensor);
      ParallelOpScope parallel{op_guard, 1};
      parallel.RecordTensor(tensor, 0);
      EXPECT_NE(op_guard.GetCublasHandle(), nullptr);
      EXPECT_NE(parallel.GetAuxiliaryCublasHandle(0), nullptr);
      EXPECT_NE(op_guard.GetCublasHandle(), parallel.GetAuxiliaryCublasHandle(0));

      ASSERT_EQ(cudaMemsetAsync(data, 0x11, 128, op_guard.GetNativeStream()), cudaSuccess);
      parallel.CheckLaunch();
      ASSERT_EQ(cudaMemsetAsync(data + 128, 0x22, 128, parallel.GetNativeAuxiliaryStream(0)), cudaSuccess);
      parallel.CheckLaunch();
      parallel.Finish();
    }

    context.Synchronize();
    std::array<uint8_t, 256> host{};
    ASSERT_EQ(cudaMemcpy(host.data(), data, host.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t index = 0; index < host.size(); index++) {
      EXPECT_EQ(host[index], index < 128 ? uint8_t{0x11} : uint8_t{0x22});
    }
  }

  runtime.Shutdown();
  EXPECT_TRUE(error_sink->GetRecords().empty());
}

}  // namespace
}  // namespace ttl::internal
