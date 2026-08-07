#include "ttl/internal/runtime/execution/event.hpp"

#include <cstdint>
#include <memory>
#include <source_location>
#include <utility>

#include <cuda_runtime_api.h>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/runtime/event.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {
namespace {

void ValidateErrorSink(const std::shared_ptr<ErrorSink> &error_sink, std::source_location location) {
  if (error_sink == nullptr) {
    throw InvalidArgumentError("event error sink must not be null", location);
  }
}

}  // namespace

EventState::EventState(Device device, uint64_t recording_stream_id, std::shared_ptr<ErrorSink> error_sink,
                       std::source_location location)
    : device_(device),
      recording_stream_id_(recording_stream_id),
      error_sink_(std::move(error_sink)),
      location_(location) {
  ValidateErrorSink(error_sink_, location_);
}

EventState::~EventState() noexcept {
  if (event_ == nullptr) {
    return;
  }

  const ErrorReportContext context{
      .location_ = location_,
      .device_ = device_,
      .stream_id_ = recording_stream_id_,
  };

  CleanupDeviceGuard device_guard{device_, *error_sink_, context, "destroy event", "restore after event destruction"};
  if (!device_guard) {
    return;
  }

  TryCuda(GetCudaApi().destroy_event_(event_), "cudaEventDestroy", *error_sink_, context);
}

void EventState::Initialize(cudaStream_t stream, std::source_location location) {
  DeviceGuard device_guard{device_, *error_sink_, location};
  const auto &cuda_api = GetCudaApi();

  cudaEvent_t event = nullptr;
  CheckCuda(cuda_api.create_event_with_flags_(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags", location);
  if (event == nullptr) {
    throw InternalError("cudaEventCreateWithFlags returned a null event", location);
  }
  event_ = event;

  CheckCuda(cuda_api.record_event_(event_, stream), "cudaEventRecord", location);
}

auto EventState::GetDevice() const noexcept -> Device { return device_; }

auto EventState::GetRecordingStreamId() const noexcept -> uint64_t { return recording_stream_id_; }

auto EventState::GetNative() const noexcept -> cudaEvent_t { return event_; }

auto EventState::Query(std::source_location location) const -> bool {
  return QueryCudaEvent(event_, "cudaEventQuery", location);
}

void EventState::Synchronize(std::source_location location) const {
  CheckCuda(GetCudaApi().synchronize_event_(event_), "cudaEventSynchronize", location);
}

auto EventAccess::Record(const Stream &stream, std::source_location location) -> Event {
  auto state =
      std::make_shared<EventState>(stream.GetDevice(), stream.GetId(), StreamAccess::GetErrorSink(stream), location);
  state->Initialize(StreamAccess::GetNative(stream), location);
  return Event{std::move(state)};
}

void EventAccess::Wait(const Stream &stream, const Event &event, std::source_location location) {
  if (event.state_ == nullptr) {
    throw InvalidArgumentError("event is in a moved-from state", location);
  }
  if (stream.GetDevice() == event.state_->GetDevice() && stream.GetId() == event.state_->GetRecordingStreamId()) {
    return;
  }

  const auto error_sink = StreamAccess::GetErrorSink(stream);
  DeviceGuard device_guard{stream.GetDevice(), *error_sink, location};
  const auto &cuda_api = GetCudaApi();
  CheckCuda(
      cuda_api.stream_wait_event_(StreamAccess::GetNative(stream), event.state_->GetNative(), cudaEventWaitDefault),
      "cudaStreamWaitEvent", location);
}

auto EventAccess::GetNative(const Event &event) noexcept -> cudaEvent_t { return event.state_->GetNative(); }

}  // namespace ttl::internal

namespace ttl {

Event::Event(std::shared_ptr<internal::EventState> state) noexcept : state_(std::move(state)) {}

auto Event::GetDevice(std::source_location location) const -> Device {
  if (state_ == nullptr) {
    throw InvalidArgumentError("event is in a moved-from state", location);
  }
  return state_->GetDevice();
}

auto Event::Query(std::source_location location) const -> bool {
  if (state_ == nullptr) {
    throw InvalidArgumentError("event is in a moved-from state", location);
  }
  return state_->Query(location);
}

void Event::Synchronize(std::source_location location) const {
  if (state_ == nullptr) {
    throw InvalidArgumentError("event is in a moved-from state", location);
  }
  state_->Synchronize(location);
}

}  // namespace ttl
