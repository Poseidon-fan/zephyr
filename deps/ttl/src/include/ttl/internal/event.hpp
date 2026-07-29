#pragma once

#include <cstdint>
#include <memory>
#include <source_location>

#include <cuda_runtime_api.h>

#include "ttl/device.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/event.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {

/** Shared owner of one immutable, timing-disabled CUDA event. */
class EventState final {
 public:
  EventState(Device device, uint64_t recording_stream_id, std::shared_ptr<ErrorSink> error_sink,
             std::source_location location);

  EventState(const EventState &) = delete;
  auto operator=(const EventState &) -> EventState & = delete;
  EventState(EventState &&) = delete;
  auto operator=(EventState &&) -> EventState & = delete;

  ~EventState() noexcept;

  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetRecordingStreamId() const noexcept -> uint64_t;
  [[nodiscard]] auto GetNative() const noexcept -> cudaEvent_t;
  [[nodiscard]] auto Query(std::source_location location) const -> bool;
  void Synchronize(std::source_location location) const;

 private:
  friend class EventAccess;

  void Initialize(cudaStream_t stream, std::source_location location);

  Device device_;
  uint64_t recording_stream_id_;
  cudaEvent_t event_;
  std::shared_ptr<ErrorSink> error_sink_;
  std::source_location location_;
};

/** Private factory and native-operation gateway for ExecutionContext and tests. */
class EventAccess final {
 public:
  [[nodiscard]] static auto Record(const Stream &stream,
                                   std::source_location location = std::source_location::current()) -> Event;

  static void Wait(const Stream &stream, const Event &event,
                   std::source_location location = std::source_location::current());

  [[nodiscard]] static auto GetNative(const Event &event) noexcept -> cudaEvent_t;
};

}  // namespace ttl::internal
