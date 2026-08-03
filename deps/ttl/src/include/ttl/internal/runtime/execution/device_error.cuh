#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "ttl/internal/runtime/execution/device_error.hpp"

namespace ttl::internal {

__device__ inline void ReportDeviceError(const DeviceErrorLaunchContext &context, DeviceErrorCode code,
                                         int64_t linear_index, uint64_t offending_value_bits, int64_t bound = 0) {
  const auto raw_code = static_cast<uint32_t>(code);
  if (atomicCAS(&context.record_->code_, uint32_t{0}, raw_code) != 0) {
    return;
  }
  context.record_->source_dtype_ = static_cast<uint8_t>(context.source_dtype_);
  context.record_->target_dtype_ = static_cast<uint8_t>(context.target_dtype_);
  context.record_->operation_sequence_ = context.operation_sequence_;
  context.record_->linear_index_ = linear_index;
  context.record_->offending_value_bits_ = offending_value_bits;
  context.record_->bound_ = bound;
}

}  // namespace ttl::internal
