#include "ttl/execution_context.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <source_location>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/event.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/device_guard.hpp"
#include "ttl/internal/event.hpp"
#include "ttl/internal/event_pool.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/internal/runtime.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {

ExecutionContextImpl::ExecutionContextImpl(std::shared_ptr<RuntimeState> runtime_state,
                                           std::shared_ptr<DeviceContext> device_context, Stream stream,
                                           std::vector<Stream> auxiliary_streams, std::optional<PooledEvent> fork_event,
                                           std::vector<PooledEvent> join_events) noexcept
    : runtime_state_(std::move(runtime_state)),
      device_context_(std::move(device_context)),
      stream_(std::move(stream)),
      auxiliary_streams_(std::move(auxiliary_streams)),
      fork_event_(std::move(fork_event)),
      join_events_(std::move(join_events)) {}

ExecutionContextImpl::~ExecutionContextImpl() noexcept { runtime_state_->UnregisterExecutionContext(); }

ContextUseGuard::ContextUseGuard(ExecutionContext &context, ContextUseMode mode, std::source_location location)
    : impl_(ContextAccess::GetImpl(context, location)) {
  const auto status = impl_.runtime_state_->GetStatus();
  const auto is_cleanup = mode == ContextUseMode::CLEANUP;
  const auto is_available = status == RuntimeStatus::RUNNING || (is_cleanup && status == RuntimeStatus::CLOSING);
  if (!is_available) {
    throw InvalidArgumentError("execution context runtime is not available", location);
  }
  if (impl_.in_use_.test_and_set(std::memory_order_acquire)) {
    throw InvalidArgumentError("execution context is already in use by another host thread", location);
  }
  if (impl_.status_.load(std::memory_order_acquire) == ExecutionContextStatus::FAILED && !is_cleanup) {
    impl_.in_use_.clear(std::memory_order_release);
    throw InvalidArgumentError("execution context is in a failed state", location);
  }
}

ContextUseGuard::~ContextUseGuard() noexcept { impl_.in_use_.clear(std::memory_order_release); }

auto ContextAccess::Create(const std::shared_ptr<RuntimeState> &runtime_state,
                           std::shared_ptr<DeviceContext> device_context, Stream stream,
                           const ExecutionContextOptions &options, std::source_location location) -> ExecutionContext {
  std::vector<Stream> auxiliary_streams;
  if (options.max_auxiliary_stream_count_ > auxiliary_streams.max_size()) {
    throw OverflowError("auxiliary stream count exceeds the host container limit", location);
  }
  auxiliary_streams.reserve(options.max_auxiliary_stream_count_);
  for (size_t index = 0; index < options.max_auxiliary_stream_count_; index++) {
    auxiliary_streams.push_back(StreamAccess::CreateOwned(stream.GetDevice(), options.stream_priority_,
                                                          runtime_state->GetErrorSink(), location));
  }

  std::optional<PooledEvent> fork_event;
  std::vector<PooledEvent> join_events;
  if (!auxiliary_streams.empty()) {
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

  runtime_state->RegisterExecutionContext(location);
  try {
    return ExecutionContext{std::make_unique<ExecutionContextImpl>(runtime_state, std::move(device_context),
                                                                   std::move(stream), std::move(auxiliary_streams),
                                                                   std::move(fork_event), std::move(join_events))};
  } catch (...) {
    runtime_state->UnregisterExecutionContext();
    throw;
  }
}

auto ContextAccess::GetImpl(ExecutionContext &context, std::source_location location) -> ExecutionContextImpl & {
  if (context.impl_ == nullptr) {
    throw InvalidArgumentError("execution context is in a moved-from state", location);
  }
  return *context.impl_;
}

auto ContextAccess::GetRuntimeState(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<RuntimeState> & {
  return GetImpl(context, location).runtime_state_;
}

auto ContextAccess::GetDeviceContext(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<DeviceContext> & {
  return GetImpl(context, location).device_context_;
}

auto ContextAccess::GetAllocator(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<DeviceAllocator> & {
  return GetDeviceContext(context, location)->GetAllocator();
}

auto ContextAccess::GetErrorSink(ExecutionContext &context, std::source_location location)
    -> const std::shared_ptr<ErrorSink> & {
  return GetRuntimeState(context, location)->GetErrorSink();
}

auto ContextAccess::GetStream(ExecutionContext &context, std::source_location location) -> const Stream & {
  return GetImpl(context, location).stream_;
}

auto ContextAccess::GetNativeStream(ExecutionContext &context, std::source_location location) -> cudaStream_t {
  return StreamAccess::GetNative(GetStream(context, location));
}

}  // namespace ttl::internal

namespace ttl {

ExecutionContext::ExecutionContext(std::unique_ptr<internal::ExecutionContextImpl> impl) noexcept
    : impl_(std::move(impl)) {}

ExecutionContext::ExecutionContext(ExecutionContext &&) noexcept = default;

auto ExecutionContext::operator=(ExecutionContext &&) noexcept -> ExecutionContext & = default;

ExecutionContext::~ExecutionContext() noexcept = default;

auto ExecutionContext::GetDevice() const noexcept -> Device { return impl_->stream_.GetDevice(); }

auto ExecutionContext::GetStream() const noexcept -> const Stream & { return impl_->stream_; }

auto ExecutionContext::GetAuxiliaryStreamCount() const noexcept -> size_t { return impl_->auxiliary_streams_.size(); }

auto ExecutionContext::IsExternalStream() const noexcept -> bool { return impl_->stream_.IsExternal(); }

auto ExecutionContext::RecordEvent(std::source_location location) -> Event {
  internal::ContextUseGuard use_guard{*this, internal::ContextUseMode::SUBMIT, location};
  return internal::EventAccess::Record(impl_->stream_, location);
}

void ExecutionContext::Wait(const Event &event, std::source_location location) {
  internal::ContextUseGuard use_guard{*this, internal::ContextUseMode::SUBMIT, location};
  internal::EventAccess::Wait(impl_->stream_, event, location);
}

void ExecutionContext::Synchronize(std::source_location location) {
  internal::ContextUseGuard use_guard{*this, internal::ContextUseMode::CLEANUP, location};
  internal::DeviceGuard device_guard{GetDevice(), *impl_->runtime_state_->GetErrorSink(), location};
  const auto &cuda_api = internal::GetCudaApi();
  auto first_status = cuda_api.synchronize_stream_(internal::StreamAccess::GetNative(impl_->stream_));
  if (impl_->status_.load(std::memory_order_acquire) == internal::ExecutionContextStatus::FAILED) {
    for (const auto &stream : impl_->auxiliary_streams_) {
      const auto status = cuda_api.synchronize_stream_(internal::StreamAccess::GetNative(stream));
      if (first_status == cudaSuccess && status != cudaSuccess) {
        first_status = status;
      }
    }
  }
  internal::CheckCuda(first_status, "cudaStreamSynchronize", location);
  impl_->device_context_->GetAllocator()->Poll();
}

void ExecutionContext::Poll(std::source_location location) {
  internal::ContextUseGuard use_guard{*this, internal::ContextUseMode::CLEANUP, location};
  impl_->device_context_->GetAllocator()->Poll();
}

}  // namespace ttl
