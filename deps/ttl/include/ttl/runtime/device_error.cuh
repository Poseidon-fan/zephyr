#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "ttl/runtime/device_error.hpp"

namespace ttl {

/** Atomically publish the first semantic error observed by a checked external CUDA submission. */
__device__ inline void ReportCudaDeviceError(const CudaDeviceErrorContext &context, uint32_t code, int64_t linear_index,
                                             uint64_t offending_value_bits, int64_t bound = 0) {
  if (context.record_ == nullptr || code == static_cast<uint32_t>(CudaDeviceErrorCode::NONE)) {
    return;
  }
  if (atomicCAS(&context.record_->code_, uint32_t{0}, code) != 0) {
    return;
  }
  context.record_->source_dtype_ = static_cast<uint8_t>(context.source_dtype_);
  context.record_->target_dtype_ = static_cast<uint8_t>(context.target_dtype_);
  context.record_->operation_sequence_ = context.operation_sequence_;
  context.record_->linear_index_ = linear_index;
  context.record_->offending_value_bits_ = offending_value_bits;
  context.record_->bound_ = bound;
}

__device__ inline void ReportCudaDeviceError(const CudaDeviceErrorContext &context, CudaDeviceErrorCode code,
                                             int64_t linear_index, uint64_t offending_value_bits, int64_t bound = 0) {
  ReportCudaDeviceError(context, static_cast<uint32_t>(code), linear_index, offending_value_bits, bound);
}

}  // namespace ttl
