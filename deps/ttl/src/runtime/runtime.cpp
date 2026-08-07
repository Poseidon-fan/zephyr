#include "ttl/runtime/runtime.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <driver_types.h>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/distributed/communicator.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/distributed/communicator.hpp"
#include "ttl/internal/ops/matmul_plan.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_properties.hpp"
#include "ttl/internal/runtime/execution/event_pool.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/graph/graph.hpp"
#include "ttl/internal/runtime/library/blas_handle_pool.hpp"
#include "ttl/internal/runtime/memory/device/allocation.hpp"
#include "ttl/internal/runtime/memory/device/allocator.hpp"
#include "ttl/internal/runtime/memory/device/storage.hpp"
#include "ttl/internal/runtime/memory/pinned/allocator.hpp"
#include "ttl/internal/runtime/runtime.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/device_properties.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::internal {
namespace {

constexpr size_t MINIMUM_BLAS_WORKSPACE_BYTES = 16U * 1024U;
constexpr size_t DEFAULT_BLAS_WORKSPACE_BYTES = 4U * 1024U * 1024U;
constexpr size_t HOPPER_BLAS_WORKSPACE_BYTES = 32U * 1024U * 1024U;

void ValidateRuntimeOptions(const RuntimeOptions &options, std::source_location location) {
  if (options.error_sink_ == nullptr) {
    throw InvalidArgumentError("runtime error sink must not be null", location);
  }
  if (options.devices_.empty()) {
    throw InvalidArgumentError("runtime must register at least one CUDA device", location);
  }
  if (options.event_pool_reserve_per_device_ > options.event_pool_capacity_per_device_) {
    throw InvalidArgumentError("runtime event pool reserve cannot exceed its cache capacity", location);
  }
  if (options.blas_workspace_bytes_ != 0 && options.blas_workspace_bytes_ < MINIMUM_BLAS_WORKSPACE_BYTES) {
    throw InvalidArgumentError("runtime cuBLAS workspace must be zero or contain at least 16 KiB", location);
  }
  for (size_t index = 0; index < options.devices_.size(); index++) {
    if (std::ranges::find(options.devices_.begin() + static_cast<ptrdiff_t>(index + 1), options.devices_.end(),
                          options.devices_[index]) != options.devices_.end()) {
      throw InvalidArgumentError("runtime device ordinals must be unique", location);
    }
  }
}

[[nodiscard]] auto ResolveBlasWorkspaceBytes(const RuntimeOptions &options, const DeviceProperties &properties) noexcept
    -> size_t {
  if (options.blas_workspace_bytes_ != 0) {
    return options.blas_workspace_bytes_;
  }
  return properties.compute_capability_.major_ >= 9 ? HOPPER_BLAS_WORKSPACE_BYTES : DEFAULT_BLAS_WORKSPACE_BYTES;
}

[[nodiscard]] auto ToAllocatorOptions(const DeviceMemoryOptions &options) noexcept -> DeviceAllocatorOptions {
  return DeviceAllocatorOptions{
      .release_threshold_bytes_ = options.release_threshold_bytes_,
      .max_live_bytes_ = options.max_live_bytes_,
  };
}

[[nodiscard]] auto FormatUnknownDevice(Device device) -> std::string {
  std::string message{"runtime does not contain "};
  message.append(device.ToString());
  return message;
}

[[nodiscard]] auto FormatOutstandingContexts(size_t count) -> std::string {
  std::string message{"cannot shut down runtime while "};
  message.append(std::to_string(count));
  message.append(" ExecutionContext registration");
  if (count != 1) {
    message.push_back('s');
  }
  message.append(" remain active");
  return message;
}

[[nodiscard]] auto FormatOutstandingGraphs(size_t count) -> std::string {
  std::string message{"cannot shut down runtime while "};
  message.append(std::to_string(count));
  message.append(" captured CUDA graph");
  if (count != 1) {
    message.push_back('s');
  }
  message.append(" remain alive");
  return message;
}

void ReportAbandonedRuntime(ErrorSink &error_sink, std::source_location location) noexcept {
  try {
    error_sink.Report(ErrorRecord{
        .code_ = ErrorCode::INTERNAL,
        .message_ = "runtime was destroyed without a successful Shutdown",
        .device_ = std::nullopt,
        .stream_id_ = std::nullopt,
        .location_ = location,
    });
  } catch (...) {
    return;
  }
}

}  // namespace

DeviceContext::DeviceContext(DeviceProperties properties, std::shared_ptr<EventPool> event_pool,
                             std::shared_ptr<DeviceAllocator> allocator,
                             std::shared_ptr<BlasHandlePool> blas_handle_pool) noexcept
    : properties_(std::move(properties)),
      event_pool_(std::move(event_pool)),
      allocator_(std::move(allocator)),
      blas_handle_pool_(std::move(blas_handle_pool)) {}

auto DeviceContext::GetDevice() const noexcept -> Device { return properties_.device_; }

auto DeviceContext::GetProperties() const noexcept -> const DeviceProperties & { return properties_; }

auto DeviceContext::GetEventPool() const noexcept -> const std::shared_ptr<EventPool> & { return event_pool_; }

auto DeviceContext::GetAllocator() const noexcept -> const std::shared_ptr<DeviceAllocator> & { return allocator_; }

auto DeviceContext::GetBlasHandlePool() const noexcept -> const std::shared_ptr<BlasHandlePool> & {
  return blas_handle_pool_;
}

auto DeviceContext::GetMatmulAlgorithmCache() noexcept -> MatmulAlgorithmCache & { return matmul_algorithm_cache_; }

RuntimeState::RuntimeState(RuntimeOptions options, std::source_location location) : location_(location) {
  ValidateRuntimeOptions(options, location);

  devices_ = std::move(options.devices_);
  error_sink_ = std::move(options.error_sink_);
  device_contexts_.reserve(devices_.size());
  for (const auto device : devices_) {
    auto properties = QueryDeviceProperties(device, location);
    auto event_pool =
        std::make_shared<EventPool>(device, error_sink_, options.event_pool_capacity_per_device_, location);
    event_pool->Reserve(options.event_pool_reserve_per_device_, location);
    auto allocator =
        DeviceAllocator::Create(device, error_sink_, event_pool, ToAllocatorOptions(options.device_memory_), location);
    auto blas_handle_pool = std::make_shared<BlasHandlePool>(device, ResolveBlasWorkspaceBytes(options, properties),
                                                             error_sink_, event_pool, allocator, location);
    device_contexts_.push_back(std::make_shared<DeviceContext>(std::move(properties), std::move(event_pool),
                                                               std::move(allocator), std::move(blas_handle_pool)));
  }

  std::vector<std::shared_ptr<EventPool>> event_pools;
  event_pools.reserve(device_contexts_.size());
  for (const auto &device_context : device_contexts_) {
    event_pools.push_back(device_context->GetEventPool());
  }
  blas_shutdown_.assign(device_contexts_.size(), uint8_t{0});
  allocator_shutdown_.assign(device_contexts_.size(), uint8_t{0});
  event_pool_shutdown_.assign(device_contexts_.size(), uint8_t{0});
  pinned_allocator_ = PinnedAllocator::Create(error_sink_, event_pools, options.pinned_memory_, location);

  // Matrix entry [source][destination] answers whether source may dereference an allocation on destination. Enabling
  // access belongs to the destination allocator because it owns the memory pool whose access descriptor is updated.
  const auto peer_entry_count =
      CheckedMultiply(devices_.size(), devices_.size(), "runtime peer capability matrix size", location);
  peer_access_.assign(peer_entry_count, uint8_t{0});
  const auto &cuda_api = GetCudaApi();
  for (size_t source_index = 0; source_index < devices_.size(); source_index++) {
    for (size_t destination_index = 0; destination_index < devices_.size(); destination_index++) {
      const auto matrix_index =
          CheckedAdd(CheckedMultiply(source_index, devices_.size(), "runtime peer matrix row offset", location),
                     destination_index, "runtime peer matrix index", location);
      if (source_index == destination_index) {
        peer_access_[matrix_index] = uint8_t{1};
        continue;
      }

      int can_access = 0;
      CheckCuda(cuda_api.can_access_peer_(&can_access, devices_[source_index].GetOrdinal(),
                                          devices_[destination_index].GetOrdinal()),
                "cudaDeviceCanAccessPeer", location);
      if (can_access != 0 && can_access != 1) {
        throw InternalError("cudaDeviceCanAccessPeer returned a value other than zero or one", location);
      }
      peer_access_[matrix_index] = static_cast<uint8_t>(can_access);
      if (can_access != 0) {
        device_contexts_[destination_index]->GetAllocator()->SetPeerAccess(devices_[source_index], true, location);
      }
    }
  }
}

auto RuntimeState::GetDevices() const noexcept -> std::span<const Device> { return devices_; }

auto RuntimeState::FindDeviceIndex(Device device, std::source_location location) const -> size_t {
  const auto iterator = std::ranges::find(devices_, device);
  if (iterator == devices_.end()) {
    throw InvalidArgumentError(FormatUnknownDevice(device), location);
  }
  return static_cast<size_t>(iterator - devices_.begin());
}

auto RuntimeState::GetDeviceContext(Device device, std::source_location location) const
    -> const std::shared_ptr<DeviceContext> & {
  return device_contexts_[FindDeviceIndex(device, location)];
}

auto RuntimeState::CanAccessPeer(Device device, Device peer_device, std::source_location location) const -> bool {
  const auto device_index = FindDeviceIndex(device, location);
  const auto peer_device_index = FindDeviceIndex(peer_device, location);
  const auto matrix_index =
      CheckedAdd(CheckedMultiply(device_index, devices_.size(), "runtime peer matrix row offset", location),
                 peer_device_index, "runtime peer matrix index", location);
  return peer_access_[matrix_index] != 0;
}

auto RuntimeState::GetErrorSink() const noexcept -> const std::shared_ptr<ErrorSink> & { return error_sink_; }

auto RuntimeState::GetPinnedAllocator() const noexcept -> const std::shared_ptr<PinnedAllocator> & {
  return pinned_allocator_;
}

auto RuntimeState::GetStatus() const noexcept -> RuntimeStatus { return status_.load(std::memory_order_acquire); }

auto RuntimeState::GetStatistics(std::source_location location) const -> RuntimeStatistics {
  const std::scoped_lock lock{lifecycle_latch_};
  RuntimeStatistics result{
      .status_ = GetStatus(),
      .execution_context_count_ = execution_context_count_.load(std::memory_order_acquire),
      .captured_graph_count_ = graph_count_.load(std::memory_order_acquire),
      .active_capture_count_ = active_capture_count_.load(std::memory_order_acquire),
      .devices_ = {},
      .pinned_memory_ = {},
  };
  result.devices_.reserve(device_contexts_.size());
  for (const auto &device_context : device_contexts_) {
    const auto allocator = device_context->GetAllocator()->GetStats(location);
    const auto events = device_context->GetEventPool()->GetStats();
    result.devices_.push_back(DeviceMemoryStatistics{
        .device_ = device_context->GetDevice(),
        .logical_live_bytes_ = allocator.logical_live_bytes_,
        .retiring_bytes_ = allocator.retiring_bytes_,
        .peak_physical_in_use_bytes_ = allocator.peak_physical_in_use_bytes_,
        .allocation_count_ = allocator.allocation_count_,
        .retirement_count_ = allocator.retirement_count_,
        .retry_count_ = allocator.retry_count_,
        .oom_count_ = allocator.oom_count_,
        .trim_count_ = allocator.trim_count_,
        .pending_retirement_count_ = allocator.pending_retirement_count_,
        .quarantined_retirement_count_ = allocator.quarantined_retirement_count_,
        .quarantined_bytes_ = allocator.quarantined_bytes_,
        .pool_used_bytes_ = allocator.pool_used_bytes_,
        .pool_reserved_bytes_ = allocator.pool_reserved_bytes_,
        .outstanding_storage_count_ = allocator.outstanding_storage_count_,
        .cached_event_count_ = events.cached_event_count_,
        .outstanding_event_count_ = events.outstanding_event_count_,
        .event_cache_capacity_ = events.max_cached_event_count_,
        .blas_workspace_bytes_ = device_context->GetBlasHandlePool()->GetWorkspaceBytes(),
    });
  }
  const auto pinned = pinned_allocator_->GetStats();
  result.pinned_memory_ = PinnedMemoryStatistics{
      .logical_live_bytes_ = pinned.logical_live_bytes_,
      .live_capacity_bytes_ = pinned.live_capacity_bytes_,
      .retiring_capacity_bytes_ = pinned.retiring_capacity_bytes_,
      .cached_capacity_bytes_ = pinned.cached_capacity_bytes_,
      .budgeted_capacity_bytes_ = pinned.budgeted_capacity_bytes_,
      .peak_budgeted_capacity_bytes_ = pinned.peak_budgeted_capacity_bytes_,
      .physical_bytes_ = pinned.physical_bytes_,
      .peak_physical_bytes_ = pinned.peak_physical_bytes_,
      .host_allocation_count_ = pinned.host_allocation_count_,
      .host_free_count_ = pinned.host_free_count_,
      .cache_hit_count_ = pinned.cache_hit_count_,
      .retirement_count_ = pinned.retirement_count_,
      .pending_retirement_count_ = pinned.pending_retirement_count_,
      .quarantined_retirement_count_ = pinned.quarantined_retirement_count_,
      .quarantined_bytes_ = pinned.quarantined_bytes_,
      .outstanding_buffer_count_ = pinned.outstanding_buffer_count_,
  };
  return result;
}

void RuntimeState::EnsureRunning(std::source_location location) const {
  if (GetStatus() != RuntimeStatus::RUNNING) {
    throw InvalidArgumentError("runtime is not accepting new work", location);
  }
}

auto RuntimeState::BeginExecutionContextCreation(std::source_location location) -> ExecutionContextRegistration {
  const std::scoped_lock lock{lifecycle_latch_};
  EnsureRunning(location);
  if (HasActiveCapture()) {
    throw CaptureError("cannot create an execution context during CUDA graph capture", location);
  }
  if (execution_context_count_.load(std::memory_order_relaxed) == std::numeric_limits<size_t>::max()) {
    throw OverflowError("execution context count overflow", location);
  }
  execution_context_count_.fetch_add(1, std::memory_order_relaxed);
  return ExecutionContextRegistration{shared_from_this()};
}

void RuntimeState::CommitExecutionContextCreation(std::source_location location) {
  const std::scoped_lock lock{lifecycle_latch_};
  EnsureRunning(location);
}

void RuntimeState::UnregisterExecutionContext() noexcept {
  if (execution_context_count_.fetch_sub(1, std::memory_order_release) == 0) {
    std::terminate();
  }
}

void RuntimeState::RegisterGraph(std::source_location location) {
  const std::scoped_lock lock{lifecycle_latch_};
  EnsureRunning(location);
  if (graph_count_.load(std::memory_order_relaxed) == std::numeric_limits<size_t>::max()) {
    throw OverflowError("captured CUDA graph count overflow", location);
  }
  graph_count_.fetch_add(1, std::memory_order_relaxed);
}

void RuntimeState::UnregisterGraph() noexcept {
  if (graph_count_.fetch_sub(1, std::memory_order_release) == 0) {
    std::terminate();
  }
}

void RuntimeState::BeginCapture(std::source_location location) {
  const std::scoped_lock lock{lifecycle_latch_};
  EnsureRunning(location);
  if (active_capture_count_.load(std::memory_order_relaxed) == std::numeric_limits<size_t>::max()) {
    throw OverflowError("active CUDA capture count overflow", location);
  }
  active_capture_count_.fetch_add(1, std::memory_order_relaxed);
}

void RuntimeState::EndCapture() noexcept {
  if (active_capture_count_.fetch_sub(1, std::memory_order_release) == 0) {
    std::terminate();
  }
}

void RuntimeState::TrackCaptureSession(const std::shared_ptr<CaptureSessionState> &state,
                                       std::source_location location) {
  if (state == nullptr) {
    throw InvalidArgumentError("cannot track a null CUDA graph capture session", location);
  }
  const std::scoped_lock lock{lifecycle_latch_};
  EnsureRunning(location);
  capture_sessions_.push_back(state);
}

void RuntimeState::EnqueueGraphCleanup(GraphCleanupState *state) noexcept {
  if (state == nullptr) {
    std::terminate();
  }
  const std::scoped_lock lock{graph_cleanup_latch_};
  state->next_ = pending_graph_cleanup_head_;
  pending_graph_cleanup_head_ = state;
}

void RuntimeState::PollGraphCleanupsNoexcept() noexcept {
  const std::scoped_lock lock{graph_cleanup_latch_};
  auto **link = &pending_graph_cleanup_head_;
  while (*link != nullptr) {
    auto *state = *link;
    if (!state->RetryNoexcept(*this)) {
      link = &state->next_;
      continue;
    }
    *link = state->next_;
    delete state;
  }
}

void RuntimeState::DrainGraphCleanups(std::source_location location) {
  const std::scoped_lock lock{graph_cleanup_latch_};
  auto **link = &pending_graph_cleanup_head_;
  while (*link != nullptr) {
    auto *state = *link;
    if (!state->SynchronizeAndRetry(*this, location)) {
      link = &state->next_;
      continue;
    }
    *link = state->next_;
    delete state;
  }
}

auto RuntimeState::HasActiveCapture() const noexcept -> bool {
  return active_capture_count_.load(std::memory_order_acquire) != 0;
}

auto RuntimeState::CreateCommunicatorGroup(std::span<const Device> rank_order, const NcclOptions &options,
                                           std::source_location location) -> std::shared_ptr<CommunicatorGroupState> {
  const std::scoped_lock lock{lifecycle_latch_};
  EnsureRunning(location);
  if (HasActiveCapture()) {
    throw CaptureError("cannot create an NCCL communicator group during CUDA graph capture", location);
  }
  auto state = CommunicatorGroupState::Create(shared_from_this(), rank_order, options, location);
  communicator_groups_.push_back(state);
  return state;
}

auto RuntimeState::AllocatePinned(size_t bytes, std::source_location location) -> PinnedBuffer {
  const std::scoped_lock lock{lifecycle_latch_};
  EnsureRunning(location);
  if (HasActiveCapture()) {
    throw CaptureError("cannot allocate pinned memory during CUDA graph capture", location);
  }
  return pinned_allocator_->Allocate(bytes, location);
}

void RuntimeState::TrimMemory(Device device, size_t target_reserved_bytes, std::source_location location) {
  const std::scoped_lock lock{lifecycle_latch_};
  EnsureRunning(location);
  if (HasActiveCapture()) {
    throw CaptureError("cannot trim device memory during CUDA graph capture", location);
  }
  GetDeviceContext(device, location)->GetAllocator()->TrimTo(target_reserved_bytes, location);
}

void RuntimeState::TrimPinnedMemory(std::source_location location) {
  const std::scoped_lock lock{lifecycle_latch_};
  EnsureRunning(location);
  if (HasActiveCapture()) {
    throw CaptureError("cannot trim pinned memory during CUDA graph capture", location);
  }
  pinned_allocator_->Trim(location);
}

void RuntimeState::Poll() noexcept {
  const std::scoped_lock lock{lifecycle_latch_};
  // Capture cleanup must progress before ordinary allocators: graph-owned resources can retain streams, events, and
  // allocations. Allocator polling is skipped during active capture to avoid capture-unsafe CUDA calls.
  std::erase_if(capture_sessions_, [](const auto &weak_session) {
    const auto session = weak_session.lock();
    if (session == nullptr) {
      return true;
    }
    if (session->HasPendingCleanup()) {
      session->RetryPendingCleanupNoexcept();
    }
    return !session->IsActive() && !session->HasPendingCleanup();
  });
  PollGraphCleanupsNoexcept();
  if (HasActiveCapture()) {
    return;
  }
  std::erase_if(communicator_groups_, [](const auto &group) { return group.expired(); });
  for (const auto &weak_group : communicator_groups_) {
    if (const auto group = weak_group.lock(); group != nullptr) {
      group->PollNoexcept();
    }
  }
  pinned_allocator_->Poll();
  for (const auto &device_context : device_contexts_) {
    device_context->GetBlasHandlePool()->Poll();
    device_context->GetAllocator()->Poll();
  }
}

auto RuntimeState::HasOpenCommunicatorGroups() noexcept -> bool {
  std::erase_if(communicator_groups_, [](const auto &group) { return group.expired(); });
  return std::ranges::any_of(communicator_groups_, [](const auto &weak_group) {
    const auto group = weak_group.lock();
    return group != nullptr && group->HasNativeResources();
  });
}

void RuntimeState::Shutdown(std::source_location location) {
  const std::scoped_lock lock{lifecycle_latch_};
  if (GetStatus() == RuntimeStatus::CLOSED) {
    return;
  }
  status_.store(RuntimeStatus::CLOSING, std::memory_order_release);

  std::erase_if(capture_sessions_, [](const auto &weak_session) {
    const auto session = weak_session.lock();
    if (session == nullptr) {
      return true;
    }
    if (session->HasPendingCleanup()) {
      session->RetryPendingCleanupNoexcept();
    }
    return !session->IsActive() && !session->HasPendingCleanup();
  });
  PollGraphCleanupsNoexcept();

  const auto context_count = execution_context_count_.load(std::memory_order_acquire);
  if (context_count != 0) {
    throw InvalidArgumentError(FormatOutstandingContexts(context_count), location);
  }
  if (HasActiveCapture()) {
    throw InvalidArgumentError("cannot shut down runtime while CUDA graph capture is active", location);
  }
  DrainGraphCleanups(location);
  const auto graph_count = graph_count_.load(std::memory_order_acquire);
  if (graph_count != 0) {
    throw InvalidArgumentError(FormatOutstandingGraphs(graph_count), location);
  }
  if (HasOpenCommunicatorGroups()) {
    throw InvalidArgumentError("cannot shut down runtime while an NCCL communicator group remains open", location);
  }

  // Close in dependency order: BLAS workspaces use device allocations, both device and pinned retirement use events,
  // and event pools must therefore be last. Per-component flags make a throwing shutdown call safely retryable.
  for (size_t index = 0; index < device_contexts_.size(); ++index) {
    if (blas_shutdown_[index] == 0) {
      device_contexts_[index]->GetBlasHandlePool()->Shutdown(location);
      blas_shutdown_[index] = uint8_t{1};
    }
  }
  for (size_t index = 0; index < device_contexts_.size(); ++index) {
    if (allocator_shutdown_[index] == 0) {
      device_contexts_[index]->GetAllocator()->Shutdown(location);
      allocator_shutdown_[index] = uint8_t{1};
    }
  }
  if (!pinned_allocator_shutdown_) {
    pinned_allocator_->Shutdown(location);
    pinned_allocator_shutdown_ = true;
  }
  for (size_t index = 0; index < device_contexts_.size(); ++index) {
    if (event_pool_shutdown_[index] == 0) {
      device_contexts_[index]->GetEventPool()->Close();
      event_pool_shutdown_[index] = uint8_t{1};
    }
  }
  status_.store(RuntimeStatus::CLOSED, std::memory_order_release);
}

void RuntimeState::Abandon() noexcept {
  auto expected = RuntimeStatus::RUNNING;
  if (status_.compare_exchange_strong(expected, RuntimeStatus::CLOSING, std::memory_order_acq_rel,
                                      std::memory_order_acquire) ||
      expected == RuntimeStatus::CLOSING) {
    ReportAbandonedRuntime(*error_sink_, location_);
  }
}

}  // namespace ttl::internal

namespace ttl {

Runtime::Runtime(RuntimeOptions options, std::source_location location)
    : state_(std::make_shared<internal::RuntimeState>(std::move(options), location)) {}

Runtime::~Runtime() noexcept {
  if (state_ != nullptr) {
    state_->Abandon();
  }
}

auto Runtime::GetDevices() const noexcept -> std::span<const Device> { return state_->GetDevices(); }

auto Runtime::GetDeviceProperties(Device device, std::source_location location) const -> const DeviceProperties & {
  return state_->GetDeviceContext(device, location)->GetProperties();
}

auto Runtime::CanAccessPeer(Device device, Device peer_device, std::source_location location) const -> bool {
  return state_->CanAccessPeer(device, peer_device, location);
}

auto Runtime::GetStatus() const noexcept -> RuntimeStatus { return state_->GetStatus(); }

auto Runtime::GetStatistics(std::source_location location) const -> RuntimeStatistics {
  return state_->GetStatistics(location);
}

auto Runtime::CreateExecutionContext(Device device, const ExecutionContextOptions &options,
                                     std::source_location location) -> ExecutionContext {
  auto registration = state_->BeginExecutionContextCreation(location);
  const auto device_context = state_->GetDeviceContext(device, location);
  auto stream = internal::StreamAccess::CreateOwned(device, options.stream_priority_, state_->GetErrorSink(), location);
  return internal::ContextAccess::Create(state_, std::move(registration), device_context, std::move(stream), options,
                                         location);
}

auto Runtime::WrapExternalStream(Device device, cudaStream_t stream, std::shared_ptr<void> owner,
                                 const ExecutionContextOptions &options, std::source_location location)
    -> ExecutionContext {
  auto registration = state_->BeginExecutionContextCreation(location);
  const auto device_context = state_->GetDeviceContext(device, location);
  auto wrapped_stream =
      internal::StreamAccess::WrapExternal(device, stream, std::move(owner), state_->GetErrorSink(), location);
  return internal::ContextAccess::Create(state_, std::move(registration), device_context, std::move(wrapped_stream),
                                         options, location);
}

auto Runtime::FromBlob(ExecutionContext &context, ExternalDeviceMemory memory, const Shape &shape,
                       const Strides &strides, DType dtype, int64_t storage_offset, std::source_location location)
    -> Tensor {
  internal::ContextUseGuard use_guard{context, internal::ContextUseMode::SUBMIT, location};
  if (state_->HasActiveCapture()) {
    throw CaptureError("cannot wrap external memory during CUDA graph capture", location);
  }
  if (internal::ContextAccess::GetRuntimeState(context, location).get() != state_.get()) {
    throw InvalidArgumentError("execution context belongs to a different runtime", location);
  }
  if (memory.device_ != context.GetDevice()) {
    throw InvalidArgumentError("external memory and execution context must belong to the same device", location);
  }

  const auto ownership =
      memory.owner_ == nullptr ? internal::ExternalOwnership::BORROWED : internal::ExternalOwnership::SHARED_OWNER;
  auto storage = internal::ContextAccess::GetAllocator(context, location)
                     ->WrapExternal(context.GetStream(), memory.pointer_, memory.capacity_bytes_, ownership,
                                    std::move(memory.owner_), location);
  return internal::TensorFactory::Create(std::move(storage), dtype, shape, strides, storage_offset, location);
}

auto Runtime::AllocatePinned(size_t bytes, std::source_location location) -> PinnedBuffer {
  return state_->AllocatePinned(bytes, location);
}

void Runtime::TrimMemory(Device device, size_t target_reserved_bytes, std::source_location location) {
  state_->TrimMemory(device, target_reserved_bytes, location);
}

void Runtime::TrimPinnedMemory(std::source_location location) { state_->TrimPinnedMemory(location); }

void Runtime::Poll() noexcept { state_->Poll(); }

void Runtime::Shutdown(std::source_location location) { state_->Shutdown(location); }

}  // namespace ttl

namespace ttl::internal {

auto RuntimeAccess::GetState(Runtime &runtime, std::source_location location) -> const std::shared_ptr<RuntimeState> & {
  if (runtime.state_ == nullptr) {
    throw InvalidArgumentError("runtime has no state", location);
  }
  return runtime.state_;
}

}  // namespace ttl::internal
