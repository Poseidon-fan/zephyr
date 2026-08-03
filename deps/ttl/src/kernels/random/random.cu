#include "ttl/internal/ops/random.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_math.cuh"
#include "ttl/internal/runtime/execution/device_error.cuh"
#include "ttl/internal/runtime/execution/device_error.hpp"
#include "ttl/internal/runtime/library/cuda_dtype.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t RANDOM_THREADS_PER_BLOCK = 256;
constexpr uint32_t MAXIMUM_RANDOM_BLOCKS = 65535;
constexpr uint32_t PHILOX_MULTIPLIER_0 = 0xD2511F53U;
constexpr uint32_t PHILOX_MULTIPLIER_1 = 0xCD9E8D57U;
constexpr uint32_t PHILOX_KEY_INCREMENT_0 = 0x9E3779B9U;
constexpr uint32_t PHILOX_KEY_INCREMENT_1 = 0xBB67AE85U;
constexpr float UINT24_SCALE = 1.0F / 16777216.0F;
constexpr float TWO_PI = 6.2831853071795864769F;

struct PhiloxResult final {
  uint32_t values_[4];
};

__device__ auto MultiplyHigh(uint32_t lhs, uint32_t rhs) noexcept -> uint32_t { return __umulhi(lhs, rhs); }

__device__ auto GeneratePhilox(uint64_t counter, uint64_t seed) noexcept -> PhiloxResult {
  auto counter_0 = static_cast<uint32_t>(counter);
  auto counter_1 = static_cast<uint32_t>(counter >> 32U);
  auto counter_2 = uint32_t{0};
  auto counter_3 = uint32_t{0};
  auto key_0 = static_cast<uint32_t>(seed);
  auto key_1 = static_cast<uint32_t>(seed >> 32U);

#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const auto high_0 = MultiplyHigh(PHILOX_MULTIPLIER_0, counter_0);
    const auto high_1 = MultiplyHigh(PHILOX_MULTIPLIER_1, counter_2);
    const auto low_0 = PHILOX_MULTIPLIER_0 * counter_0;
    const auto low_1 = PHILOX_MULTIPLIER_1 * counter_2;
    const auto next_0 = high_1 ^ counter_1 ^ key_0;
    const auto next_1 = low_1;
    const auto next_2 = high_0 ^ counter_3 ^ key_1;
    const auto next_3 = low_0;
    counter_0 = next_0;
    counter_1 = next_1;
    counter_2 = next_2;
    counter_3 = next_3;
    key_0 += PHILOX_KEY_INCREMENT_0;
    key_1 += PHILOX_KEY_INCREMENT_1;
  }
  return {{counter_0, counter_1, counter_2, counter_3}};
}

__device__ auto UniformZeroToOne(uint32_t bits) noexcept -> float {
  return static_cast<float>(bits >> 8U) * UINT24_SCALE;
}

__device__ auto UniformOpenClosed(uint32_t bits) noexcept -> float {
  return static_cast<float>((bits >> 8U) + 1U) * UINT24_SCALE;
}

__device__ auto GetOutputOffset(const RandomParameters &parameters, uint64_t linear_index) noexcept -> uint64_t {
  auto offset = uint64_t{0};
  for (size_t remaining_rank = parameters.rank_; remaining_rank > 0; --remaining_rank) {
    const auto axis = remaining_rank - 1;
    const auto coordinate = linear_index % parameters.shape_[axis];
    linear_index /= parameters.shape_[axis];
    offset += coordinate * parameters.output_strides_bytes_[axis];
  }
  return offset;
}

template <CudaStorageType T>
__device__ void StoreValue(const RandomParameters &parameters, uint64_t index, float value) {
  *reinterpret_cast<T *>(parameters.output_ + GetOutputOffset(parameters, index)) = FromElementwiseFloat<T>(value);
}

__global__ void InitializeGeneratorKernel(GeneratorState *state, uint64_t seed) {
  state->seed_ = seed;
  state->counter_ = 0;
}

__global__ void ReserveCounterKernel(GeneratorState *state, uint64_t blocks, uint64_t *base_counter,
                                     DeviceErrorLaunchContext error_context) {
  auto *counter = reinterpret_cast<unsigned long long *>(&state->counter_);
  auto old_counter = atomicCAS(counter, 0, 0);
  while (true) {
    if (blocks > (~uint64_t{0}) - old_counter) {
      *base_counter = old_counter;
      ReportDeviceError(error_context, DeviceErrorCode::RNG_COUNTER_OVERFLOW, 0, old_counter);
      return;
    }
    const auto next_counter = old_counter + blocks;
    const auto observed = atomicCAS(counter, old_counter, next_counter);
    if (observed == old_counter) {
      *base_counter = old_counter;
      return;
    }
    old_counter = observed;
  }
}

template <CudaStorageType T, RandomDistribution distribution>
__global__ void RandomKernel(RandomParameters parameters, const GeneratorState *state, const uint64_t *base_counter) {
  auto block_index = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const auto block_count = (parameters.num_elements_ + 3U) / 4U;
  const auto step = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (block_index < block_count) {
    const auto random = GeneratePhilox(*base_counter + block_index, state->seed_);
    auto generated = float4{};
    if constexpr (distribution == RandomDistribution::UNIFORM) {
      generated.x = parameters.first_parameter_ + ((parameters.second_parameter_ - parameters.first_parameter_) *
                                                   UniformZeroToOne(random.values_[0]));
      generated.y = parameters.first_parameter_ + ((parameters.second_parameter_ - parameters.first_parameter_) *
                                                   UniformZeroToOne(random.values_[1]));
      generated.z = parameters.first_parameter_ + ((parameters.second_parameter_ - parameters.first_parameter_) *
                                                   UniformZeroToOne(random.values_[2]));
      generated.w = parameters.first_parameter_ + ((parameters.second_parameter_ - parameters.first_parameter_) *
                                                   UniformZeroToOne(random.values_[3]));
    } else {
      const auto radius_0 = sqrtf(-2.0F * logf(UniformOpenClosed(random.values_[0])));
      const auto angle_0 = TWO_PI * UniformZeroToOne(random.values_[1]);
      const auto radius_1 = sqrtf(-2.0F * logf(UniformOpenClosed(random.values_[2])));
      const auto angle_1 = TWO_PI * UniformZeroToOne(random.values_[3]);
      generated.x = parameters.first_parameter_ + (parameters.second_parameter_ * radius_0 * cosf(angle_0));
      generated.y = parameters.first_parameter_ + (parameters.second_parameter_ * radius_0 * sinf(angle_0));
      generated.z = parameters.first_parameter_ + (parameters.second_parameter_ * radius_1 * cosf(angle_1));
      generated.w = parameters.first_parameter_ + (parameters.second_parameter_ * radius_1 * sinf(angle_1));
    }
    const auto first_index = block_index * 4U;
    if (first_index < parameters.num_elements_) {
      StoreValue<T>(parameters, first_index, generated.x);
    }
    if (first_index + 1U < parameters.num_elements_) {
      StoreValue<T>(parameters, first_index + 1U, generated.y);
    }
    if (first_index + 2U < parameters.num_elements_) {
      StoreValue<T>(parameters, first_index + 2U, generated.z);
    }
    if (first_index + 3U < parameters.num_elements_) {
      StoreValue<T>(parameters, first_index + 3U, generated.w);
    }
    block_index += step;
  }
}

[[nodiscard]] auto GetBlockCount(uint64_t work_items) noexcept -> uint32_t {
  const auto blocks = (work_items + RANDOM_THREADS_PER_BLOCK - 1) / RANDOM_THREADS_PER_BLOCK;
  return static_cast<uint32_t>(blocks < MAXIMUM_RANDOM_BLOCKS ? blocks : MAXIMUM_RANDOM_BLOCKS);
}

template <CudaStorageType T>
void LaunchTyped(cudaStream_t stream, RandomDistribution distribution, const RandomParameters &parameters,
                 const GeneratorState *state, const uint64_t *base_counter, std::source_location location) {
  const auto block_count = (parameters.num_elements_ + 3U) / 4U;
  switch (distribution) {
    case RandomDistribution::UNIFORM:
      RandomKernel<T, RandomDistribution::UNIFORM>
          <<<GetBlockCount(block_count), RANDOM_THREADS_PER_BLOCK, 0, stream>>>(parameters, state, base_counter);
      return;
    case RandomDistribution::NORMAL:
      RandomKernel<T, RandomDistribution::NORMAL>
          <<<GetBlockCount(block_count), RANDOM_THREADS_PER_BLOCK, 0, stream>>>(parameters, state, base_counter);
      return;
  }
  throw InternalError("invalid random distribution reached CUDA launcher", location);
}

}  // namespace

void LaunchInitializeGenerator(cudaStream_t stream, GeneratorState *state, uint64_t seed,
                               std::source_location location) {
  if (stream == nullptr || state == nullptr) {
    throw InternalError("invalid generator initialization parameters", location);
  }
  InitializeGeneratorKernel<<<1, 1, 0, stream>>>(state, seed);
}

void LaunchRandom(cudaStream_t stream, DType dtype, RandomDistribution distribution, const RandomParameters &parameters,
                  GeneratorState *state, uint64_t *base_counter, const DeviceErrorLaunchContext &error_context,
                  std::source_location location) {
  if (stream == nullptr || parameters.output_ == nullptr || parameters.num_elements_ == 0 || state == nullptr ||
      base_counter == nullptr || error_context.record_ == nullptr) {
    throw InternalError("invalid random launch parameters", location);
  }
  const auto blocks = (parameters.num_elements_ + 3U) / 4U;
  ReserveCounterKernel<<<1, 1, 0, stream>>>(state, blocks, base_counter, error_context);
  DispatchCudaFloatingDType(
      dtype, "RandomOut",
      [&]<CudaStorageType T>(std::type_identity<T>) {
        LaunchTyped<T>(stream, distribution, parameters, state, base_counter, location);
      },
      location);
}

}  // namespace ttl::internal
