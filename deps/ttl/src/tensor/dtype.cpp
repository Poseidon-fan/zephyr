#include "ttl/tensor/dtype.hpp"

#include <bit>
#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ttl {

static_assert(sizeof(Float16) == sizeof(__half));
static_assert(alignof(Float16) == alignof(__half));
static_assert(std::is_trivially_copyable_v<__half>);

static_assert(sizeof(BFloat16) == sizeof(__nv_bfloat16));
static_assert(alignof(BFloat16) == alignof(__nv_bfloat16));
static_assert(std::is_trivially_copyable_v<__nv_bfloat16>);

auto FloatToFloat16(float value) noexcept -> Float16 {
  return Float16{.bits_ = std::bit_cast<uint16_t>(__float2half_rn(value))};
}

auto Float16ToFloat(Float16 value) noexcept -> float { return __half2float(std::bit_cast<__half>(value.bits_)); }

auto FloatToBFloat16(float value) noexcept -> BFloat16 {
  return BFloat16{.bits_ = std::bit_cast<uint16_t>(__float2bfloat16_rn(value))};
}

auto BFloat16ToFloat(BFloat16 value) noexcept -> float {
  return __bfloat162float(std::bit_cast<__nv_bfloat16>(value.bits_));
}

}  // namespace ttl
