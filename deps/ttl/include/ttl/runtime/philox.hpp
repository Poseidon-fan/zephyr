#pragma once

#include <cstdint>
#include <type_traits>

namespace ttl {

/** Device-resident generator state. Its layout is public so checked extension kernels can consume reservations. */
struct CudaPhiloxGeneratorState final {
  uint64_t seed_;
  uint64_t counter_;
};

/** One stream-ordered reservation of 128-bit Philox blocks. Pointers are valid for the enclosing launch callback. */
struct CudaPhiloxReservation final {
  const CudaPhiloxGeneratorState *generator_state_;
  const uint64_t *base_counter_;
};

struct CudaPhiloxResult final {
  uint32_t values_[4];
};

static_assert(std::is_trivially_copyable_v<CudaPhiloxGeneratorState>);
static_assert(std::is_standard_layout_v<CudaPhiloxGeneratorState>);
static_assert(std::is_trivially_copyable_v<CudaPhiloxReservation>);
static_assert(std::is_standard_layout_v<CudaPhiloxReservation>);
static_assert(std::is_trivially_copyable_v<CudaPhiloxResult>);
static_assert(std::is_standard_layout_v<CudaPhiloxResult>);

}  // namespace ttl
