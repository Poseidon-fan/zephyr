#include "ttl/runtime/execution_context.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <source_location>
#include <utility>
#include <vector>

#include <driver_types.h>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/ops/matmul_plan.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"
#include "ttl/internal/runtime/execution/device_error.hpp"
#include "ttl/internal/runtime/execution/event.hpp"
#include "ttl/internal/runtime/execution/event_pool.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/execution/execution_lane.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/graph/graph.hpp"
#include "ttl/internal/runtime/library/blas_handle_pool.hpp"
#include "ttl/internal/runtime/memory/pinned/allocator.hpp"
#include "ttl/internal/runtime/runtime.hpp"
#include "ttl/runtime/event.hpp"
#include "ttl/runtime/graph.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {

ExecutionContextRegistration::ExecutionContextRegistration(std::shared_ptr<RuntimeState> runtime_state) noexcept
    : runtime_state_(std::move(runtime_state)) {}

ExecutionContextRegistration::ExecutionContextRegistration(ExecutionContextRegistration &&other) noexcept
    : runtime_state_(std::move(other.runtime_state_)) {}

ExecutionContextRegistration::~ExecutionContextRegistration() noexcept {
  if (runtime_state_ != nullptr) {
    runtime_state_->UnregisterExecutionContext();
  }
}

ExecutionContextState::ExecutionContextState(std::shared_ptr<RuntimeState> runtime_state,
                                             ExecutionContextRegistration registration,
                                             std::shared_ptr<DeviceContext> device_context, ExecutionLane primary_lane,
                                             std::vector<ExecutionLane> auxiliary_lanes,
                                             std::optional<PooledEvent> fork_event,
                                             std::vector<PooledEvent> join_events,
                                             std::unique_ptr<DeviceErrorState> device_error_state) noexcept
    : runtime_state_(std::move(runtime_state)),
      registration_(std::move(registration)),
      device_context_(std::move(device_context)),
      primary_lane_(std::move(primary_lane)),
      auxiliary_lanes_(std::move(auxiliary_lanes)),
      fork_event_(std::move(fork_event)),
      join_events_(std::move(join_events)),
      device_error_state_(std::move(device_error_state)) {}

ExecutionContextState::~ExecutionContextState() noexcept {
  // Context-owned dependency events may still be referenced by an asynchronous stream operation. Do not return
  // these handles to the shared cache during teardown; Discard destroys the handle without making it available for
  // another recording sequence.
  if (fork_event_.has_value()) {
    fork_event_->Discard();
  }
  for (auto &event : join_events_) {
    event.Discard();
  }
}

ContextUseGuard::ContextUseGuard(ExecutionContext &context, ContextUseMode mode, std::source_location location)
    : ctx_state_(ContextAccess::GetState(context, location)) {
  const auto status = ctx_state_.runtime_state_->GetStatus();
  const auto is_cleanup = mode == ContextUseMode::CLEANUP;
  const auto is_available = status == RuntimeStatus::RUNNING || (is_cleanup && status == RuntimeStatus::CLOSING);
  if (!is_available) {
    throw InvalidArgumentError("execution context runtime is not available", location);
  }
  if (ctx_state_.in_use_.test_and_set(std::memory_order_acquire)) {
    throw InvalidArgumentError("execution context is already in use by another host thread", location);
  }
  if (ctx_state_.status_.load(std::memory_order_acquire) == ExecutionContextStatus::FAILED && !is_cleanup) {
    ctx_state_.in_use_.clear(std::memory_order_release);
    throw InvalidArgumentError("execution context is in a failed state", location);
  }
  if (!ctx_state_.capture_state_.expired() && is_cleanup) {
    ctx_state_.in_use_.clear(std::memory_order_release);
    throw CaptureError("synchronization and polling are forbidden during CUDA graph capture", location);
  }
}

ContextUseGuard::~ContextUseGuard() noexcept { ctx_state_.in_use_.clear(std::memory_order_release); }

auto ContextAccess::Create(const std::shared_ptr<RuntimeState> &runtime_state,
                           ExecutionContextRegistration registration, std::shared_ptr<DeviceContext> device_context,
                           Stream stream, const ExecutionContextOptions &options, std::source_location location)
    -> ExecutionContext {
  ExecutionLane primary_lane{std::move(stream), device_context->GetBlasHandlePool(), device_context->GetAllocator()};

  std::vector<ExecutionLane> auxiliary_lanes;
  if (options.max_auxiliary_stream_count_ > auxiliary_lanes.max_size()) {
    throw OverflowError("auxiliary stream count exceeds the host container limit", location);
  }
  auxiliary_lanes.reserve(options.max_auxiliary_stream_count_);
  for (size_t index = 0; index < options.max_auxiliary_stream_count_; index++) {
    auto auxiliary_stream = StreamAccess::CreateOwned(primary_lane.GetStream().GetDevice(), options.stream_priority_,
                                                      runtime_state->GetErrorSink(), location);
    auxiliary_lanes.emplace_back(std::move(auxiliary_stream), device_context->GetBlasHandlePool(),
                                 device_context->GetAllocator());
  }

  std::optional<PooledEvent> fork_event;
  std::vector<PooledEvent> join_events;
  if (!auxiliary_lanes.empty()) {
    const auto &event_pool = device_context->GetEventPool();
    fork_event.emplace(event_pool->Acquire(location));
    if (options.max_auxiliary_stream_count_ > join_events.max_size()) {
      throw OverflowError("auxiliary join event count exceeds the host container limit", location);
    }
    join_events.reserve(options.max_auxiliary_stream_count_);
    for (size_t index = 0; index < options.max_auxiliary_stream_count_; index++) {
      join_events.push_back(event_pool->Acquire(location));
    }
  }

  auto device_error_state =
      DeviceErrorState::Create(device_context->GetAllocator(), primary_lane.GetStream(),
                               runtime_state->GetPinnedAllocator(), runtime_state->GetErrorSink(), location);
  return ExecutionContext{std::make_shared<ExecutionContextState>(
      runtime_state, std::move(registration), std::move(device_context), std::move(primary_lane),
      std::move(auxiliary_lanes), std::move(fork_event), std::move(join_events), std::move(device_error_state))};
}

auto ContextAccess::GetState(ExecutionContext &context, std::source_location location) -> ExecutionContextState & {
  if (context.state_ == nullptr) {
    throw InvalidArgumentError("execution context is in a moved-from state", location);
  }
  return *context.state_;
}

auto ContextAccess::GetStateOwner(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<ExecutionContextState> & {
  static_cast<void>(GetState(context, location));
  return context.state_;
}

auto ContextAccess::GetRuntimeState(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<RuntimeState> & {
  return GetState(context, location).runtime_state_;
}

auto ContextAccess::GetDeviceContext(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<DeviceContext> & {
  return GetState(context, location).device_context_;
}

auto ContextAccess::GetAllocator(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<DeviceAllocator> & {
  return GetDeviceContext(context, location)->GetAllocator();
}

auto ContextAccess::GetDeviceProperties(ExecutionContext &context, std::source_location location)
    -> const DeviceProperties & {
  return GetDeviceContext(context, location)->GetProperties();
}

auto ContextAccess::CanAccessPeer(ExecutionContext &context, Device peer_device, std::source_location location)
    -> bool {
  return GetRuntimeState(context, location)->CanAccessPeer(context.GetDevice(), peer_device, location);
}

auto ContextAccess::GetErrorSink(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<ErrorSink> & {
  return GetRuntimeState(context, location)->GetErrorSink();
}

auto ContextAccess::GetPinnedAllocator(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<PinnedAllocator> & {
  return GetRuntimeState(context, location)->GetPinnedAllocator();
}

auto ContextAccess::AllocatePinned(ExecutionContext &context, size_t bytes, std::source_location location)
    -> PinnedBuffer {
  return GetRuntimeState(context, location)->AllocatePinned(bytes, location);
}

auto ContextAccess::GetStream(ExecutionContext &context, std::source_location location) -> const Stream & {
  return GetState(context, location).primary_lane_.GetStream();
}

auto ContextAccess::GetNativeStream(ExecutionContext &context, std::source_location location) -> cudaStream_t {
  return StreamAccess::GetNative(GetStream(context, location));
}

auto ContextAccess::GetPrimaryLane(ExecutionContext &context, std::source_location location) -> ExecutionLane & {
  return GetState(context, location).primary_lane_;
}

auto ContextAccess::GetMatmulAlgorithmCache(ExecutionContext &context, std::source_location location)
    -> MatmulAlgorithmCache & {
  return GetDeviceContext(context, location)->GetMatmulAlgorithmCache();
}

auto ContextAccess::GetDeviceErrorState(ExecutionContext &context, std::source_location location)
    -> DeviceErrorState & {
  return *GetState(context, location).device_error_state_;
}

}  // namespace ttl::internal

namespace ttl {
namespace {

void SynchronizeAndCheckDeviceErrors(internal::ExecutionContextState &state, std::source_location location) {
  const auto &cuda_api = internal::GetCudaApi();
  const auto primary_stream = internal::StreamAccess::GetNative(state.primary_lane_.GetStream());
  const auto failed = state.status_.load(std::memory_order_acquire) == internal::ExecutionContextStatus::FAILED;

  auto first_status = cudaSuccess;
  if (failed) {
    // A failed structured multi-stream submission may have bypassed its normal join. Drain every lane before reading
    // or resetting context-owned error state so no auxiliary kernel can still write it.
    first_status = cuda_api.synchronize_stream_(primary_stream);
    for (const auto &lane : state.auxiliary_lanes_) {
      const auto status = cuda_api.synchronize_stream_(internal::StreamAccess::GetNative(lane.GetStream()));
      if (first_status == cudaSuccess && status != cudaSuccess) {
        first_status = status;
      }
    }
    internal::CheckCuda(first_status, "cudaStreamSynchronize", location);
  }

  state.device_error_state_->EnqueueRead(state.primary_lane_.GetStream(), location);
  // The device-to-host copy is ordered after all successful submissions on the primary stream. Synchronizing once
  // therefore observes both native launch failures and the sticky semantic error record.
  first_status = cuda_api.synchronize_stream_(primary_stream);
  if (first_status != cudaSuccess) {
    state.status_.store(internal::ExecutionContextStatus::FAILED, std::memory_order_release);
    for (const auto &lane : state.auxiliary_lanes_) {
      static_cast<void>(cuda_api.synchronize_stream_(internal::StreamAccess::GetNative(lane.GetStream())));
    }
  }
  internal::CheckCuda(first_status, "cudaStreamSynchronize", location);
  state.device_error_state_->ConsumeAndReset(primary_stream, location);
}

}  // namespace

ExecutionContext::ExecutionContext(std::shared_ptr<internal::ExecutionContextState> state) noexcept
    : state_(std::move(state)) {}

auto ExecutionContext::GetDevice(std::source_location location) const -> Device {
  if (state_ == nullptr) {
    throw InvalidArgumentError("execution context is in a moved-from state", location);
  }
  return state_->primary_lane_.GetStream().GetDevice(location);
}

auto ExecutionContext::GetStream(std::source_location location) const -> const Stream & {
  if (state_ == nullptr) {
    throw InvalidArgumentError("execution context is in a moved-from state", location);
  }
  return state_->primary_lane_.GetStream();
}

auto ExecutionContext::GetAuxiliaryStreamCount(std::source_location location) const -> size_t {
  if (state_ == nullptr) {
    throw InvalidArgumentError("execution context is in a moved-from state", location);
  }
  return state_->auxiliary_lanes_.size();
}

auto ExecutionContext::IsExternalStream(std::source_location location) const -> bool {
  return GetStream(location).IsExternal(location);
}

auto ExecutionContext::RecordEvent(std::source_location location) -> Event {
  internal::ContextUseGuard use_guard{*this, internal::ContextUseMode::SUBMIT, location};
  if (!state_->capture_state_.expired()) {
    throw CaptureError("public event recording is forbidden during CUDA graph capture", location);
  }
  return internal::EventAccess::Record(state_->primary_lane_.GetStream(), location);
}

void ExecutionContext::Wait(const Event &event, std::source_location location) {
  internal::ContextUseGuard use_guard{*this, internal::ContextUseMode::SUBMIT, location};
  if (!state_->capture_state_.expired()) {
    throw CaptureError("public event waits are forbidden during CUDA graph capture", location);
  }
  internal::EventAccess::Wait(state_->primary_lane_.GetStream(), event, location);
}

auto ExecutionContext::BeginCapture(std::source_location location) -> CaptureSession {
  return BeginCapture(GraphCaptureOptions{}, location);
}

auto ExecutionContext::BeginCapture(const GraphCaptureOptions &options, std::source_location location)
    -> CaptureSession {
  return CaptureSession{internal::CaptureSessionState::Begin(*this, options, location)};
}

void ExecutionContext::CheckAsyncErrors(std::source_location location) {
  internal::ContextUseGuard use_guard{*this, internal::ContextUseMode::CLEANUP, location};
  internal::DeviceGuard device_guard{GetDevice(), *state_->runtime_state_->GetErrorSink(), location};
  SynchronizeAndCheckDeviceErrors(*state_, location);
}

void ExecutionContext::Synchronize(std::source_location location) {
  internal::ContextUseGuard use_guard{*this, internal::ContextUseMode::CLEANUP, location};
  internal::DeviceGuard device_guard{GetDevice(), *state_->runtime_state_->GetErrorSink(), location};
  SynchronizeAndCheckDeviceErrors(*state_, location);
  state_->device_context_->GetBlasHandlePool()->Poll();
  state_->device_context_->GetAllocator()->Poll();
  state_->runtime_state_->GetPinnedAllocator()->Poll();
}

void ExecutionContext::Poll(std::source_location location) {
  internal::ContextUseGuard use_guard{*this, internal::ContextUseMode::CLEANUP, location};
  state_->device_context_->GetBlasHandlePool()->Poll();
  state_->device_context_->GetAllocator()->Poll();
  state_->runtime_state_->GetPinnedAllocator()->Poll();
}

}  // namespace ttl
