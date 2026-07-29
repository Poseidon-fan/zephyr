#pragma once

#include <cstdint>
#include <source_location>

#include "ttl/device.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/cuda_api.hpp"

namespace ttl::internal {

/**
 * Sets the CUDA device for the calling host thread and restores the previous device on scope exit.
 *
 * The referenced ErrorSink and CudaApi must outlive the guard. Construction may throw; destruction never throws and
 * reports a failed restoration through ErrorSink. This class must not be used from a CUDA host callback.
 */
class DeviceGuard final {
 public:
  DeviceGuard(Device device, ErrorSink &error_sink, std::source_location location = std::source_location::current());
  DeviceGuard(Device device, ErrorSink &error_sink, const CudaApi &cuda_api,
              std::source_location location = std::source_location::current());

  DeviceGuard(const DeviceGuard &) = delete;
  auto operator=(const DeviceGuard &) -> DeviceGuard & = delete;
  DeviceGuard(DeviceGuard &&) = delete;
  auto operator=(DeviceGuard &&) -> DeviceGuard & = delete;

  ~DeviceGuard() noexcept;

 private:
  Device device_;
  ErrorSink &error_sink_;
  const CudaApi &cuda_api_;
  std::source_location location_;
  int32_t previous_ordinal_{-1};
  bool changed_{false};
};

}  // namespace ttl::internal
