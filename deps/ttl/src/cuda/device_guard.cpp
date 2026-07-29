#include "ttl/internal/device_guard.hpp"

#include <optional>
#include <source_location>

#include "ttl/device.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"

namespace ttl::internal {

DeviceGuard::DeviceGuard(Device device, ErrorSink &error_sink, std::source_location location)
    : DeviceGuard(device, error_sink, GetCudaApi(), location) {}

DeviceGuard::DeviceGuard(Device device, ErrorSink &error_sink, const CudaApi &cuda_api, std::source_location location)
    : device_(device), error_sink_(error_sink), cuda_api_(cuda_api), location_(location) {
  CheckCuda(cuda_api_.get_device_(&previous_ordinal_), "cudaGetDevice", location_);
  if (previous_ordinal_ == device_.GetOrdinal()) {
    return;
  }

  CheckCuda(cuda_api_.set_device_(device_.GetOrdinal()), "cudaSetDevice", location_);
  changed_ = true;
}

DeviceGuard::~DeviceGuard() noexcept {
  if (!changed_) {
    return;
  }

  TryCuda(cuda_api_.set_device_(previous_ordinal_), "cudaSetDevice (restore previous device)", error_sink_,
          ErrorReportContext{
              .location_ = location_,
              .device_ = device_,
              .stream_id_ = std::nullopt,
          });
}

}  // namespace ttl::internal
