#pragma once

#include <cstdint>
#include <type_traits>

#include "ttl/tensor/dtype.hpp"

namespace ttl {

/** Stable device-side semantic error categories shared by TTL and checked external CUDA kernels. */
// NOLINTNEXTLINE(performance-enum-size): external kernels own the complete 32-bit user error-code space.
enum class CudaDeviceErrorCode : uint32_t {
  NONE = 0,
  INDEX_OUT_OF_BOUNDS = 1,
  INTEGER_DIVIDE_BY_ZERO = 2,
  CAST_OUT_OF_RANGE = 3,
  RNG_COUNTER_OVERFLOW = 4,
  INVALID_VALUE = 5,
  /** External error codes use USER_DEFINED or values greater than USER_DEFINED. */
  USER_DEFINED = 1024,
};

/** Sticky first-error record owned by one ExecutionContext. */
struct CudaDeviceErrorRecord final {
  uint32_t code_{0};
  uint8_t source_dtype_{0};
  uint8_t target_dtype_{0};
  uint16_t reserved_{0};
  uint64_t operation_sequence_{0};
  int64_t linear_index_{0};
  uint64_t offending_value_bits_{0};
  int64_t bound_{0};
};

/** Per-submission capability used only by kernels enqueued through the launch that returned it. */
struct CudaDeviceErrorContext final {
  CudaDeviceErrorRecord *record_{nullptr};
  uint64_t operation_sequence_{0};
  DType source_dtype_{DType::BOOL};
  DType target_dtype_{DType::BOOL};
};

static_assert(std::is_trivially_copyable_v<CudaDeviceErrorRecord>);
static_assert(std::is_standard_layout_v<CudaDeviceErrorRecord>);
static_assert(std::is_trivially_copyable_v<CudaDeviceErrorContext>);
static_assert(std::is_standard_layout_v<CudaDeviceErrorContext>);

}  // namespace ttl
