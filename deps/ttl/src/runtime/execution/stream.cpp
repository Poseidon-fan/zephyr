#include "ttl/internal/runtime/execution/stream.hpp"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <source_location>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"
#include "ttl/runtime/device.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {
namespace {

constinit std::atomic<uint64_t> next_stream_id{1};

[[nodiscard]] auto NextStreamId(std::source_location location) -> uint64_t {
  auto id = next_stream_id.load(std::memory_order_relaxed);
  while (true) {
    if (id == std::numeric_limits<uint64_t>::max()) {
      throw OverflowError("stream ID space exhausted", location);
    }
    if (next_stream_id.compare_exchange_weak(id, id + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return id;
    }
  }
}

[[nodiscard]] auto FormatPriorityError(int32_t priority, int least_priority, int greatest_priority) -> std::string {
  std::string message{"stream priority "};
  message.append(std::to_string(priority));
  message.append(" is outside the CUDA device range [");
  message.append(std::to_string(greatest_priority));
  message.append(", ");
  message.append(std::to_string(least_priority));
  message.push_back(']');
  return message;
}

void ValidateErrorSink(const std::shared_ptr<ErrorSink> &error_sink, std::source_location location) {
  if (error_sink == nullptr) {
    throw InvalidArgumentError("stream error sink must not be null", location);
  }
}

void ValidateExternalStream(cudaStream_t stream, std::source_location location) {
  if (stream == nullptr || stream == cudaStreamLegacy) {
    throw InvalidArgumentError("TTL does not support the legacy default CUDA stream", location);
  }
  if (stream == cudaStreamPerThread) {
    throw InvalidArgumentError("TTL does not support the per-thread default CUDA stream", location);
  }
}

}  // namespace

StreamState::StreamState(Device device, int32_t priority, std::shared_ptr<ErrorSink> error_sink,
                         std::source_location location)
    : id_(NextStreamId(location)),
      device_(device),
      stream_(nullptr),
      error_sink_(std::move(error_sink)),
      location_(location),
      is_external_(false) {
  ValidateErrorSink(error_sink_, location_);

  DeviceGuard device_guard{device_, *error_sink_, location_};
  const auto &cuda_api = GetCudaApi();
  int least_priority = 0;
  int greatest_priority = 0;
  CheckCuda(cuda_api.get_stream_priority_range_(&least_priority, &greatest_priority),
            "cudaDeviceGetStreamPriorityRange", location_);
  if (greatest_priority > least_priority) {
    throw InternalError("CUDA returned an invalid stream priority range", location_);
  }
  if (priority < greatest_priority || priority > least_priority) {
    throw InvalidArgumentError(FormatPriorityError(priority, least_priority, greatest_priority), location_);
  }

  CheckCuda(cuda_api.create_stream_with_priority_(&stream_, cudaStreamNonBlocking, priority),
            "cudaStreamCreateWithPriority", location_);
  if (stream_ == nullptr) {
    throw InternalError("cudaStreamCreateWithPriority returned a null stream", location_);
  }
}

StreamState::StreamState(Device device, cudaStream_t stream, std::shared_ptr<void> external_owner,
                         std::shared_ptr<ErrorSink> error_sink, std::source_location location)
    : id_(NextStreamId(location)),
      device_(device),
      stream_(stream),
      external_owner_(std::move(external_owner)),
      error_sink_(std::move(error_sink)),
      location_(location),
      is_external_(true) {
  ValidateErrorSink(error_sink_, location_);
  ValidateExternalStream(stream_, location_);
}

StreamState::~StreamState() noexcept {
  if (is_external_) {
    return;
  }

  const ErrorReportContext context{
      .location_ = location_,
      .device_ = device_,
      .stream_id_ = id_,
  };

  CleanupDeviceGuard device_guard{device_, *error_sink_, context, "destroy stream", "restore after stream destruction"};
  if (!device_guard) {
    return;
  }

  TryCuda(GetCudaApi().destroy_stream_(stream_), "cudaStreamDestroy", *error_sink_, context);
}

auto StreamState::GetId() const noexcept -> uint64_t { return id_; }

auto StreamState::GetDevice() const noexcept -> Device { return device_; }

auto StreamState::GetNative() const noexcept -> cudaStream_t { return stream_; }

auto StreamState::IsExternal() const noexcept -> bool { return is_external_; }

auto StreamState::HasExternalOwner() const noexcept -> bool { return external_owner_ != nullptr; }

auto StreamAccess::CreateOwned(Device device, int32_t priority, std::shared_ptr<ErrorSink> error_sink,
                               std::source_location location) -> Stream {
  return Stream{std::make_shared<StreamState>(device, priority, std::move(error_sink), location)};
}

auto StreamAccess::WrapExternal(Device device, cudaStream_t stream, std::shared_ptr<void> external_owner,
                                std::shared_ptr<ErrorSink> error_sink, std::source_location location) -> Stream {
  return Stream{
      std::make_shared<StreamState>(device, stream, std::move(external_owner), std::move(error_sink), location)};
}

auto StreamAccess::GetState(const Stream &stream) noexcept -> std::shared_ptr<StreamState> { return stream.state_; }

auto StreamAccess::GetErrorSink(const Stream &stream) noexcept -> std::shared_ptr<ErrorSink> {
  return stream.state_->error_sink_;
}

auto StreamAccess::GetNative(const Stream &stream) noexcept -> cudaStream_t { return stream.state_->GetNative(); }

}  // namespace ttl::internal

namespace ttl {

Stream::Stream(std::shared_ptr<internal::StreamState> state) noexcept : state_(std::move(state)) {}

auto Stream::GetDevice(std::source_location location) const -> Device {
  if (state_ == nullptr) {
    throw InvalidArgumentError("stream is in a moved-from state", location);
  }
  return state_->GetDevice();
}

auto Stream::GetId(std::source_location location) const -> uint64_t {
  if (state_ == nullptr) {
    throw InvalidArgumentError("stream is in a moved-from state", location);
  }
  return state_->GetId();
}

auto Stream::IsExternal(std::source_location location) const -> bool {
  if (state_ == nullptr) {
    throw InvalidArgumentError("stream is in a moved-from state", location);
  }
  return state_->IsExternal();
}

}  // namespace ttl
