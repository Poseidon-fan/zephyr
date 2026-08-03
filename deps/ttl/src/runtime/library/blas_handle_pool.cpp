#include "ttl/internal/runtime/library/blas_handle_pool.hpp"

#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <string>
#include <utility>
#include <vector>

#include <cublasLt.h>   // IWYU pragma: keep
#include <cublas_v2.h>  // IWYU pragma: keep
#include <driver_types.h>

#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"
#include "ttl/internal/runtime/execution/event_pool.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/library/cublas_api.hpp"
#include "ttl/internal/runtime/memory/device_allocator.hpp"
#include "ttl/internal/tensor/storage.hpp"
#include "ttl/runtime/device.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {
namespace {

constexpr size_t MINIMUM_WORKSPACE_BYTES = 16U * 1024U;
constexpr size_t WORKSPACE_ALIGNMENT = 256;

struct BlasResource final {
  BlasResource(cublasHandle_t handle, cublasLtHandle_t lt_handle, std::shared_ptr<Storage> workspace) noexcept
      : handle_(handle), lt_handle_(lt_handle), workspace_(std::move(workspace)) {}

  BlasResource(const BlasResource &) = delete;
  auto operator=(const BlasResource &) -> BlasResource & = delete;
  BlasResource(BlasResource &&other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)),
        lt_handle_(std::exchange(other.lt_handle_, nullptr)),
        workspace_(std::move(other.workspace_)) {}

  auto operator=(BlasResource &&other) noexcept -> BlasResource & {
    if (this != &other) {
      if (handle_ != nullptr || lt_handle_ != nullptr) {
        std::terminate();
      }
      handle_ = std::exchange(other.handle_, nullptr);
      lt_handle_ = std::exchange(other.lt_handle_, nullptr);
      workspace_ = std::move(other.workspace_);
    }
    return *this;
  }

  cublasHandle_t handle_;
  cublasLtHandle_t lt_handle_;
  std::shared_ptr<Storage> workspace_;
};

struct PendingBlasResource final {
  PendingBlasResource(BlasResource resource, std::shared_ptr<StreamState> stream,
                      std::optional<PooledEvent> completion_event, bool poisoned) noexcept
      : resource_(std::move(resource)),
        stream_(std::move(stream)),
        completion_event_(std::move(completion_event)),
        poisoned_(poisoned) {}

  PendingBlasResource(const PendingBlasResource &) = delete;
  auto operator=(const PendingBlasResource &) -> PendingBlasResource & = delete;
  PendingBlasResource(PendingBlasResource &&) noexcept = default;
  auto operator=(PendingBlasResource &&) noexcept -> PendingBlasResource & = default;

  BlasResource resource_;
  std::shared_ptr<StreamState> stream_;
  std::optional<PooledEvent> completion_event_;
  bool poisoned_;
};

void ValidateDependencies(Device device, size_t workspace_bytes, const std::shared_ptr<ErrorSink> &error_sink,
                          const std::shared_ptr<EventPool> &event_pool,
                          const std::shared_ptr<DeviceAllocator> &allocator, std::source_location location) {
  if (workspace_bytes < MINIMUM_WORKSPACE_BYTES) {
    throw InvalidArgumentError("cuBLAS workspace must contain at least 16 KiB", location);
  }
  if (error_sink == nullptr) {
    throw InvalidArgumentError("cuBLAS handle pool error sink must not be null", location);
  }
  if (event_pool == nullptr || event_pool->GetDevice() != device) {
    throw InvalidArgumentError("cuBLAS handle pool requires an event pool on the same device", location);
  }
  if (allocator == nullptr || allocator->GetDevice() != device) {
    throw InvalidArgumentError("cuBLAS handle pool requires an allocator on the same device", location);
  }
}

[[nodiscard]] auto MakeErrorContext(Device device, const StreamState &stream, std::source_location location) noexcept
    -> ErrorReportContext {
  return ErrorReportContext{
      .location_ = location,
      .device_ = device,
      .stream_id_ = stream.GetId(),
  };
}

[[nodiscard]] auto FormatOutstandingLeases(size_t count) -> std::string {
  std::string message{"cannot shut down cuBLAS handle pool while "};
  message.append(std::to_string(count));
  message.append(" handle lease");
  if (count != 1) {
    message.push_back('s');
  }
  message.append(" remain alive");
  return message;
}

}  // namespace

class BlasHandlePoolState final : public std::enable_shared_from_this<BlasHandlePoolState> {
 public:
  BlasHandlePoolState(Device device, size_t workspace_bytes, std::shared_ptr<ErrorSink> error_sink,
                      std::shared_ptr<EventPool> event_pool, std::shared_ptr<DeviceAllocator> allocator,
                      std::source_location location)
      : device_(device),
        workspace_bytes_(workspace_bytes),
        error_sink_(std::move(error_sink)),
        event_pool_(std::move(event_pool)),
        allocator_(std::move(allocator)),
        location_(location) {
    ValidateDependencies(device_, workspace_bytes_, error_sink_, event_pool_, allocator_, location_);
  }

  BlasHandlePoolState(const BlasHandlePoolState &) = delete;
  auto operator=(const BlasHandlePoolState &) -> BlasHandlePoolState & = delete;
  BlasHandlePoolState(BlasHandlePoolState &&) = delete;
  auto operator=(BlasHandlePoolState &&) -> BlasHandlePoolState & = delete;

  ~BlasHandlePoolState() noexcept { CloseNoexcept(); }

  [[nodiscard]] auto Acquire(const Stream &stream, std::source_location location) -> BlasHandleLease {
    if (stream.GetDevice() != device_) {
      throw InvalidArgumentError("cuBLAS handle pool and stream must belong to the same device", location);
    }

    DeviceGuard device_guard{device_, *error_sink_, location};
    std::scoped_lock lock{latch_};
    if (is_closed_) {
      throw InvalidArgumentError("cannot acquire from a closed cuBLAS handle pool", location);
    }
    PollReadyResources(location);
    if (outstanding_lease_count_ == std::numeric_limits<size_t>::max()) {
      throw OverflowError("cuBLAS handle lease count overflow", location);
    }
    if (pending_resources_.size() > std::numeric_limits<size_t>::max() - cached_resources_.size()) {
      throw OverflowError("cuBLAS handle pool retained resource count overflow", location);
    }
    const auto retained_resource_count = cached_resources_.size() + pending_resources_.size();
    if (retained_resource_count >= std::numeric_limits<size_t>::max() - outstanding_lease_count_) {
      throw OverflowError("cuBLAS handle pool resource count overflow", location);
    }
    const auto required_capacity = retained_resource_count + outstanding_lease_count_ + 1;
    if (required_capacity > cached_resources_.max_size() || required_capacity > pending_resources_.max_size()) {
      throw OverflowError("cuBLAS handle pool exceeds the host container limit", location);
    }
    cached_resources_.reserve(required_capacity);
    pending_resources_.reserve(required_capacity);

    auto resource = cached_resources_.empty() ? CreateResource(stream, location) : TakeCachedResource();
    try {
      BindResource(resource, stream, location);
    } catch (...) {
      DestroyResourceNoexcept(resource);
      throw;
    }

    outstanding_lease_count_++;
    return BlasHandleLease{std::exchange(resource.handle_, nullptr), std::exchange(resource.lt_handle_, nullptr),
                           std::move(resource.workspace_), StreamAccess::GetState(stream), shared_from_this()};
  }

  void Release(cublasHandle_t handle, cublasLtHandle_t lt_handle, std::shared_ptr<Storage> workspace,
               std::shared_ptr<StreamState> stream) noexcept {
    if (handle == nullptr || lt_handle == nullptr) {
      return;
    }

    BlasResource resource{handle, lt_handle, std::move(workspace)};
    std::optional<PooledEvent> completion_event;
    bool poisoned = false;
    const auto error_context = MakeErrorContext(device_, *stream, location_);

    CleanupDeviceGuard device_guard{device_, *error_sink_, error_context, "retire cuBLAS handle",
                                    "restore after cuBLAS handle retirement"};
    if (!device_guard) {
      poisoned = true;
    } else {
      try {
        completion_event.emplace(event_pool_->Acquire(location_));
        const auto status = GetCudaApi().record_event_(completion_event->GetNative(), stream->GetNative());
        if (!TryCuda(status, "cudaEventRecord", "cuBLAS handle retirement", *error_sink_, error_context)) {
          completion_event->Discard();
          completion_event.reset();
          poisoned = true;
        }
      } catch (...) {
        ReportReleaseFailure(error_context);
        completion_event.reset();
        poisoned = true;
      }
    }

    {
      std::scoped_lock lock{latch_};
      if (outstanding_lease_count_ == 0) {
        std::terminate();
      }
      outstanding_lease_count_--;
      if (!is_closed_) {
        pending_resources_.emplace_back(std::move(resource), std::move(stream), std::move(completion_event), poisoned);
        return;
      }
    }

    PendingBlasResource pending{std::move(resource), std::move(stream), std::move(completion_event), poisoned};
    if (SynchronizePendingNoexcept(pending)) {
      DestroyResourceNoexcept(pending.resource_);
    } else {
      pending.resource_.handle_ = nullptr;
      pending.resource_.lt_handle_ = nullptr;
      pending.resource_.workspace_.reset();
    }
  }

  void PollNoexcept() noexcept {
    const ErrorReportContext error_context{
        .location_ = location_,
        .device_ = device_,
        .stream_id_ = std::nullopt,
    };
    CleanupDeviceGuard device_guard{device_, *error_sink_, error_context, "poll cuBLAS handle pool",
                                    "restore after cuBLAS handle pool poll"};
    if (!device_guard) {
      return;
    }

    std::scoped_lock lock{latch_};
    PollReadyResourcesNoexcept();
  }

  void Shutdown(std::source_location location) {
    DeviceGuard device_guard{device_, *error_sink_, location};
    std::scoped_lock lock{latch_};
    if (is_closed_) {
      return;
    }
    if (outstanding_lease_count_ != 0) {
      throw InvalidArgumentError(FormatOutstandingLeases(outstanding_lease_count_), location);
    }

    while (!pending_resources_.empty()) {
      auto &pending = pending_resources_.back();
      SynchronizePending(pending, location);
      DestroyResource(pending.resource_, location);
      pending_resources_.pop_back();
    }
    while (!cached_resources_.empty()) {
      DestroyResource(cached_resources_.back(), location);
      cached_resources_.pop_back();
    }
    is_closed_ = true;
  }

  [[nodiscard]] auto GetDevice() const noexcept -> Device { return device_; }

  [[nodiscard]] auto GetWorkspaceBytes() const noexcept -> size_t { return workspace_bytes_; }

 private:
  [[nodiscard]] auto CreateResource(const Stream &stream, std::source_location location) -> BlasResource {
    auto workspace = allocator_->Allocate(stream, workspace_bytes_, WORKSPACE_ALIGNMENT,
                                          AllocationContext{
                                              .operation_ = "cuBLAS workspace",
                                              .output_shape_ = std::nullopt,
                                              .dtype_ = std::nullopt,
                                              .location_ = location,
                                          });

    cublasHandle_t handle = nullptr;
    const auto status = GetCublasApi().create_(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      if (handle != nullptr) {
        const ErrorReportContext error_context{
            .location_ = location,
            .device_ = device_,
            .stream_id_ = stream.GetId(),
        };
        TryCublas(GetCublasApi().destroy_(handle), "cublasDestroy after failed creation", *error_sink_, error_context);
      }
      CheckCublas(status, "cublasCreate", location);
    }
    if (handle == nullptr) {
      throw InternalError("cublasCreate returned a null handle", location);
    }
    cublasLtHandle_t lt_handle = nullptr;
    const auto lt_status = GetCublasApi().lt_create_(&lt_handle);
    if (lt_status != CUBLAS_STATUS_SUCCESS) {
      TryCublas(GetCublasApi().destroy_(handle), "cublasDestroy after failed cublasLtCreate", *error_sink_,
                MakeErrorContext(device_, *StreamAccess::GetState(stream), location));
      CheckCublas(lt_status, "cublasLtCreate", location);
    }
    if (lt_handle == nullptr) {
      TryCublas(GetCublasApi().destroy_(handle), "cublasDestroy after null cublasLt handle", *error_sink_,
                MakeErrorContext(device_, *StreamAccess::GetState(stream), location));
      throw InternalError("cublasLtCreate returned a null handle", location);
    }
    return BlasResource{handle, lt_handle, std::move(workspace)};
  }

  [[nodiscard]] auto TakeCachedResource() -> BlasResource {
    auto resource = std::move(cached_resources_.back());
    cached_resources_.pop_back();
    return resource;
  }

  void BindResource(BlasResource &resource, const Stream &stream, std::source_location location) {
    const auto &cublas_api = GetCublasApi();
    CheckCublas(cublas_api.set_stream_(resource.handle_, StreamAccess::GetNative(stream)), "cublasSetStream", location);
    CheckCublas(cublas_api.set_pointer_mode_(resource.handle_, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode",
                location);
    CheckCublas(cublas_api.set_workspace_(resource.handle_, resource.workspace_->GetBasePointer(),
                                          resource.workspace_->GetCapacityBytes()),
                "cublasSetWorkspace", location);
    resource.workspace_->RecordUsage(stream);
  }

  void PollReadyResources(std::source_location location) {
    for (size_t index = 0; index < pending_resources_.size();) {
      auto &pending = pending_resources_[index];
      if (pending.poisoned_ || !pending.completion_event_.has_value()) {
        index++;
        continue;
      }

      const auto status = GetCudaApi().query_event_(pending.completion_event_->GetNative());
      if (status == cudaErrorNotReady) {
        index++;
        continue;
      }
      if (status != cudaSuccess) {
        pending.completion_event_->Discard();
        pending.completion_event_.reset();
        pending.poisoned_ = true;
        CheckCuda(status, "cudaEventQuery (cuBLAS handle retirement)", location);
      }

      cached_resources_.push_back(std::move(pending.resource_));
      pending_resources_.erase(pending_resources_.begin() + static_cast<ptrdiff_t>(index));
    }
  }

  void PollReadyResourcesNoexcept() noexcept {
    for (size_t index = 0; index < pending_resources_.size();) {
      auto &pending = pending_resources_[index];
      if (pending.poisoned_ || !pending.completion_event_.has_value()) {
        index++;
        continue;
      }

      const auto status = GetCudaApi().query_event_(pending.completion_event_->GetNative());
      if (status == cudaErrorNotReady) {
        index++;
        continue;
      }
      if (status != cudaSuccess) {
        const auto error_context = MakeErrorContext(device_, *pending.stream_, location_);
        TryCuda(status, "cudaEventQuery", "cuBLAS handle retirement", *error_sink_, error_context);
        pending.completion_event_->Discard();
        pending.completion_event_.reset();
        pending.poisoned_ = true;
        index++;
        continue;
      }

      cached_resources_.push_back(std::move(pending.resource_));
      pending_resources_.erase(pending_resources_.begin() + static_cast<ptrdiff_t>(index));
    }
  }

  void SynchronizePending(PendingBlasResource &pending, std::source_location location) {
    if (pending.completion_event_.has_value()) {
      const auto event_status = GetCudaApi().synchronize_event_(pending.completion_event_->GetNative());
      if (event_status == cudaSuccess) {
        return;
      }
      pending.completion_event_->Discard();
      pending.completion_event_.reset();
      CheckCuda(GetCudaApi().synchronize_stream_(pending.stream_->GetNative()),
                "cudaStreamSynchronize (cuBLAS handle retirement fallback)", location);
      CheckCuda(event_status, "cudaEventSynchronize (cuBLAS handle retirement)", location);
      return;
    }
    CheckCuda(GetCudaApi().synchronize_stream_(pending.stream_->GetNative()),
              "cudaStreamSynchronize (cuBLAS handle retirement fallback)", location);
  }

  [[nodiscard]] auto SynchronizePendingNoexcept(PendingBlasResource &pending) noexcept -> bool {
    const auto error_context = MakeErrorContext(device_, *pending.stream_, location_);
    if (pending.completion_event_.has_value()) {
      if (TryCuda(GetCudaApi().synchronize_event_(pending.completion_event_->GetNative()), "cudaEventSynchronize",
                  "cuBLAS handle retirement", *error_sink_, error_context)) {
        return true;
      }
    }
    return TryCuda(GetCudaApi().synchronize_stream_(pending.stream_->GetNative()), "cudaStreamSynchronize",
                   "cuBLAS handle retirement fallback", *error_sink_, error_context);
  }

  void DestroyResource(BlasResource &resource, std::source_location location) {
    if (resource.lt_handle_ != nullptr) {
      CheckCublas(GetCublasApi().lt_destroy_(resource.lt_handle_), "cublasLtDestroy", location);
      resource.lt_handle_ = nullptr;
    }
    if (resource.handle_ != nullptr) {
      CheckCublas(GetCublasApi().destroy_(resource.handle_), "cublasDestroy", location);
      resource.handle_ = nullptr;
    }
    resource.workspace_.reset();
  }

  void DestroyResourceNoexcept(BlasResource &resource) noexcept {
    const ErrorReportContext error_context{
        .location_ = location_,
        .device_ = device_,
        .stream_id_ = std::nullopt,
    };
    if (resource.lt_handle_ != nullptr) {
      TryCublas(GetCublasApi().lt_destroy_(resource.lt_handle_), "cublasLtDestroy", *error_sink_, error_context);
      resource.lt_handle_ = nullptr;
    }
    if (resource.handle_ != nullptr) {
      TryCublas(GetCublasApi().destroy_(resource.handle_), "cublasDestroy", *error_sink_, error_context);
      resource.handle_ = nullptr;
    }
    resource.workspace_.reset();
  }

  void ReportReleaseFailure(const ErrorReportContext &context) noexcept {
    try {
      error_sink_->Report(ErrorRecord{
          .code_ = ErrorCode::INTERNAL,
          .message_ = "failed to acquire a completion event while retiring a cuBLAS handle",
          .device_ = context.device_,
          .stream_id_ = context.stream_id_,
          .location_ = context.location_,
      });
    } catch (...) {
      return;
    }
  }

  void CloseNoexcept() noexcept {
    std::vector<BlasResource> cached_resources;
    std::vector<PendingBlasResource> pending_resources;
    {
      std::scoped_lock lock{latch_};
      if (is_closed_) {
        return;
      }
      is_closed_ = true;
      cached_resources.swap(cached_resources_);
      pending_resources.swap(pending_resources_);
    }

    const ErrorReportContext error_context{
        .location_ = location_,
        .device_ = device_,
        .stream_id_ = std::nullopt,
    };
    CleanupDeviceGuard device_guard{device_, *error_sink_, error_context, "close cuBLAS handle pool",
                                    "restore after cuBLAS handle pool close"};
    if (!device_guard) {
      for (auto &pending : pending_resources) {
        pending.resource_.handle_ = nullptr;
        pending.resource_.lt_handle_ = nullptr;
      }
      for (auto &resource : cached_resources) {
        resource.handle_ = nullptr;
        resource.lt_handle_ = nullptr;
      }
      return;
    }
    for (auto &pending : pending_resources) {
      if (SynchronizePendingNoexcept(pending)) {
        DestroyResourceNoexcept(pending.resource_);
      } else {
        pending.resource_.handle_ = nullptr;
        pending.resource_.lt_handle_ = nullptr;
        pending.resource_.workspace_.reset();
      }
    }
    for (auto &resource : cached_resources) {
      DestroyResourceNoexcept(resource);
    }
  }

  Device device_;
  size_t workspace_bytes_;
  std::shared_ptr<ErrorSink> error_sink_;
  std::shared_ptr<EventPool> event_pool_;
  std::shared_ptr<DeviceAllocator> allocator_;
  std::source_location location_;
  std::mutex latch_;
  std::vector<BlasResource> cached_resources_;
  std::vector<PendingBlasResource> pending_resources_;
  size_t outstanding_lease_count_{0};
  bool is_closed_{false};
};

BlasHandleLease::BlasHandleLease(cublasHandle_t handle, cublasLtHandle_t lt_handle, std::shared_ptr<Storage> workspace,
                                 std::shared_ptr<StreamState> stream,
                                 std::shared_ptr<BlasHandlePoolState> pool) noexcept
    : handle_(handle),
      lt_handle_(lt_handle),
      workspace_(std::move(workspace)),
      stream_(std::move(stream)),
      pool_(std::move(pool)) {}

BlasHandleLease::BlasHandleLease(BlasHandleLease &&other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      lt_handle_(std::exchange(other.lt_handle_, nullptr)),
      workspace_(std::move(other.workspace_)),
      stream_(std::move(other.stream_)),
      pool_(std::move(other.pool_)) {}

auto BlasHandleLease::operator=(BlasHandleLease &&other) noexcept -> BlasHandleLease & {
  if (this != &other) {
    Reset();
    handle_ = std::exchange(other.handle_, nullptr);
    lt_handle_ = std::exchange(other.lt_handle_, nullptr);
    workspace_ = std::move(other.workspace_);
    stream_ = std::move(other.stream_);
    pool_ = std::move(other.pool_);
  }
  return *this;
}

BlasHandleLease::~BlasHandleLease() noexcept { Reset(); }

auto BlasHandleLease::GetCublasHandle() const noexcept -> cublasHandle_t { return handle_; }

auto BlasHandleLease::GetCublasLtHandle() const noexcept -> cublasLtHandle_t { return lt_handle_; }

auto BlasHandleLease::GetWorkspace() const noexcept -> Storage & { return *workspace_; }

auto BlasHandleLease::GetWorkspaceStorage() const noexcept -> const std::shared_ptr<Storage> & { return workspace_; }

void BlasHandleLease::Reset() noexcept {
  if (handle_ == nullptr) {
    return;
  }
  const auto handle = std::exchange(handle_, nullptr);
  const auto lt_handle = std::exchange(lt_handle_, nullptr);
  auto workspace = std::move(workspace_);
  auto stream = std::move(stream_);
  auto pool = std::move(pool_);
  pool->Release(handle, lt_handle, std::move(workspace), std::move(stream));
}

BlasHandlePool::BlasHandlePool(Device device, size_t workspace_bytes, std::shared_ptr<ErrorSink> error_sink,
                               std::shared_ptr<EventPool> event_pool, std::shared_ptr<DeviceAllocator> allocator,
                               std::source_location location)
    : state_(std::make_shared<BlasHandlePoolState>(device, workspace_bytes, std::move(error_sink),
                                                   std::move(event_pool), std::move(allocator), location)) {}

BlasHandlePool::~BlasHandlePool() noexcept = default;

auto BlasHandlePool::Acquire(const Stream &stream, std::source_location location) -> BlasHandleLease {
  return state_->Acquire(stream, location);
}

void BlasHandlePool::Poll() noexcept { state_->PollNoexcept(); }

void BlasHandlePool::Shutdown(std::source_location location) { state_->Shutdown(location); }

auto BlasHandlePool::GetDevice() const noexcept -> Device { return state_->GetDevice(); }

auto BlasHandlePool::GetWorkspaceBytes() const noexcept -> size_t { return state_->GetWorkspaceBytes(); }

}  // namespace ttl::internal
