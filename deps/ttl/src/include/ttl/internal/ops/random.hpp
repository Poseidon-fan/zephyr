#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <driver_types.h>

#include "ttl/internal/runtime/execution/device_error.hpp"
#include "ttl/runtime/philox.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {

using GeneratorState = CudaPhiloxGeneratorState;

struct RandomParameters final {
  std::byte *output_;
  uint64_t shape_[TTL_MAX_RANK]{};
  uint64_t output_strides_bytes_[TTL_MAX_RANK]{};
  uint64_t num_elements_;
  uint8_t rank_;
  float first_parameter_;
  float second_parameter_;
};

enum class RandomDistribution : uint8_t {
  UNIFORM,
  NORMAL,
};

static_assert(std::is_trivially_copyable_v<RandomParameters>);
static_assert(std::is_standard_layout_v<RandomParameters>);

void LaunchInitializeGenerator(cudaStream_t stream, GeneratorState *state, uint64_t seed,
                               std::source_location location = std::source_location::current());
void LaunchReservePhilox(cudaStream_t stream, GeneratorState *state, uint64_t block_count, uint64_t *base_counter,
                         const DeviceErrorLaunchContext &error_context,
                         std::source_location location = std::source_location::current());
void LaunchRandom(cudaStream_t stream, DType dtype, RandomDistribution distribution, const RandomParameters &parameters,
                  GeneratorState *state, uint64_t *base_counter, const DeviceErrorLaunchContext &error_context,
                  std::source_location location = std::source_location::current());

}  // namespace ttl::internal
