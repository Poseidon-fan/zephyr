#pragma once

#include <cstdint>
#include <source_location>
#include <string_view>

#include "ttl/device.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/cuda_check.hpp"

namespace ttl::internal {

/**
 * Sets the CUDA device for the calling host thread and restores the previous device on scope exit.
 *
 * The referenced ErrorSink must outlive the guard. Construction may throw; destruction never throws and reports a
 * failed restoration through ErrorSink. This class must not be used from a CUDA host callback.
 */
class DeviceGuard final {
 public:
  DeviceGuard(Device device, ErrorSink &error_sink, std::source_location location = std::source_location::current());

  DeviceGuard(const DeviceGuard &) = delete;
  auto operator=(const DeviceGuard &) -> DeviceGuard & = delete;
  DeviceGuard(DeviceGuard &&) = delete;
  auto operator=(DeviceGuard &&) -> DeviceGuard & = delete;

  ~DeviceGuard() noexcept;

 private:
  Device device_;
  ErrorSink &error_sink_;
  std::source_location location_;
  int32_t previous_ordinal_{-1};
  bool changed_{false};
};

/**
 * Non-throwing device guard for destructors and other cleanup boundaries.
 *
 * CUDA failures are reported through ErrorSink. A false guard means the target device could not be established and
 * device-specific cleanup must be skipped. The sink and restore-operation string must outlive the guard.
 */
class CleanupDeviceGuard final {
 public:
  CleanupDeviceGuard(Device device, ErrorSink &error_sink, const ErrorReportContext &context,
                     std::string_view operation, std::string_view restore_operation) noexcept;

  CleanupDeviceGuard(const CleanupDeviceGuard &) = delete;
  auto operator=(const CleanupDeviceGuard &) -> CleanupDeviceGuard & = delete;
  CleanupDeviceGuard(CleanupDeviceGuard &&) = delete;
  auto operator=(CleanupDeviceGuard &&) -> CleanupDeviceGuard & = delete;

  ~CleanupDeviceGuard() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;

 private:
  ErrorSink &error_sink_;
  ErrorReportContext context_;
  std::string_view restore_operation_;
  int32_t previous_ordinal_{-1};
  bool changed_{false};
  bool ready_{false};
};

}  // namespace ttl::internal
