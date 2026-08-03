#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "ttl/runtime/philox.hpp"

namespace ttl {

/** Generate one Philox4x32-10 result for a previously reserved block counter. */
__device__ inline auto GenerateCudaPhilox(uint64_t counter, uint64_t seed) noexcept -> CudaPhiloxResult {
  constexpr uint32_t multiplier_0 = 0xD2511F53U;
  constexpr uint32_t multiplier_1 = 0xCD9E8D57U;
  constexpr uint32_t key_increment_0 = 0x9E3779B9U;
  constexpr uint32_t key_increment_1 = 0xBB67AE85U;

  auto counter_0 = static_cast<uint32_t>(counter);
  auto counter_1 = static_cast<uint32_t>(counter >> 32U);
  auto counter_2 = uint32_t{0};
  auto counter_3 = uint32_t{0};
  auto key_0 = static_cast<uint32_t>(seed);
  auto key_1 = static_cast<uint32_t>(seed >> 32U);

#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const auto high_0 = __umulhi(multiplier_0, counter_0);
    const auto high_1 = __umulhi(multiplier_1, counter_2);
    const auto low_0 = multiplier_0 * counter_0;
    const auto low_1 = multiplier_1 * counter_2;
    const auto next_0 = high_1 ^ counter_1 ^ key_0;
    const auto next_1 = low_1;
    const auto next_2 = high_0 ^ counter_3 ^ key_1;
    const auto next_3 = low_0;
    counter_0 = next_0;
    counter_1 = next_1;
    counter_2 = next_2;
    counter_3 = next_3;
    key_0 += key_increment_0;
    key_1 += key_increment_1;
  }
  return {{counter_0, counter_1, counter_2, counter_3}};
}

}  // namespace ttl
