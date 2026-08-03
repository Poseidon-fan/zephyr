#pragma once

#include <cstdint>

#include "ttl/internal/runtime/execution/device_error.hpp"
#include "ttl/runtime/device_error.cuh"

namespace ttl::internal {

__device__ inline void ReportDeviceError(const DeviceErrorLaunchContext &context, DeviceErrorCode code,
                                         int64_t linear_index, uint64_t offending_value_bits, int64_t bound = 0) {
  ReportCudaDeviceError(context, code, linear_index, offending_value_bits, bound);
}

}  // namespace ttl::internal
