#include "ttl/internal/device_guard.hpp"

#include <optional>
#include <source_location>
#include <string_view>

#include "ttl/device.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"

namespace ttl::internal {

DeviceGuard::DeviceGuard(Device device, ErrorSink &error_sink, std::source_location location)
    : device_(device), error_sink_(error_sink), location_(location) {
  const auto &cuda_api = GetCudaApi();
  CheckCuda(cuda_api.get_device_(&previous_ordinal_), "cudaGetDevice", location_);
  if (previous_ordinal_ == device_.GetOrdinal()) {
    return;
  }

  CheckCuda(cuda_api.set_device_(device_.GetOrdinal()), "cudaSetDevice", location_);
  changed_ = true;
}

DeviceGuard::~DeviceGuard() noexcept {
  if (!changed_) {
    return;
  }

  TryCuda(GetCudaApi().set_device_(previous_ordinal_), "cudaSetDevice (restore previous device)", error_sink_,
          ErrorReportContext{
              .location_ = location_,
              .device_ = device_,
              .stream_id_ = std::nullopt,
          });
}

CleanupDeviceGuard::CleanupDeviceGuard(Device device, ErrorSink &error_sink, const ErrorReportContext &context,
                                       std::string_view operation, std::string_view restore_operation) noexcept
    : error_sink_(error_sink), context_(context), restore_operation_(restore_operation) {
  const auto &cuda_api = GetCudaApi();
  if (!TryCuda(cuda_api.get_device_(&previous_ordinal_), "cudaGetDevice", operation, error_sink_, context_)) {
    return;
  }
  if (previous_ordinal_ == device.GetOrdinal()) {
    ready_ = true;
    return;
  }
  if (!TryCuda(cuda_api.set_device_(device.GetOrdinal()), "cudaSetDevice", operation, error_sink_, context_)) {
    return;
  }
  changed_ = true;
  ready_ = true;
}

CleanupDeviceGuard::~CleanupDeviceGuard() noexcept {
  if (!changed_) {
    return;
  }

  TryCuda(GetCudaApi().set_device_(previous_ordinal_), "cudaSetDevice", restore_operation_, error_sink_, context_);
}

CleanupDeviceGuard::operator bool() const noexcept { return ready_; }

}  // namespace ttl::internal
