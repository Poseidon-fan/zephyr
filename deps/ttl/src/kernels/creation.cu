#include "ttl/internal/creation.hpp"

#include <cstdint>
#include <source_location>

#include <cuda_runtime.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t ARANGE_THREADS_PER_BLOCK = 256;
constexpr uint32_t MAXIMUM_ARANGE_BLOCKS = 65535;
constexpr int64_t MINIMUM_INT64 = -9223372036854775807LL - 1LL;

template <typename T>
__global__ void IntegerArangeKernel(T *output, uint64_t num_elements, int64_t start, int64_t step) {
  auto index = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const auto grid_stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const auto unsigned_start = static_cast<uint64_t>(start);
  const auto unsigned_step = static_cast<uint64_t>(step);
  while (index < num_elements) {
    constexpr auto sign_bit = uint64_t{1} << 63U;
    const auto bits = unsigned_start + (index * unsigned_step);
    const auto value =
        bits < sign_bit ? static_cast<int64_t>(bits) : MINIMUM_INT64 + static_cast<int64_t>(bits - sign_bit);
    output[index] = static_cast<T>(value);
    index += grid_stride;
  }
}

__global__ void FloatingArangeKernel(float *output, uint64_t num_elements, double start, double step) {
  auto index = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const auto grid_stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (index < num_elements) {
    output[index] = static_cast<float>(start + (static_cast<double>(index) * step));
    index += grid_stride;
  }
}

[[nodiscard]] auto GetBlockCount(uint64_t num_elements) noexcept -> uint32_t {
  const auto blocks = (num_elements + ARANGE_THREADS_PER_BLOCK - 1U) / ARANGE_THREADS_PER_BLOCK;
  return static_cast<uint32_t>(blocks < MAXIMUM_ARANGE_BLOCKS ? blocks : MAXIMUM_ARANGE_BLOCKS);
}

}  // namespace

void LaunchArange(cudaStream_t stream, DType dtype, const ArangeParameters &parameters, std::source_location location) {
  if (stream == nullptr || parameters.output_ == nullptr || parameters.num_elements_ <= 0) {
    throw InternalError("invalid arange launch parameters", location);
  }
  const auto num_elements = static_cast<uint64_t>(parameters.num_elements_);
  const auto block_count = GetBlockCount(num_elements);
  switch (dtype) {
    case DType::INT32:
      IntegerArangeKernel<<<block_count, ARANGE_THREADS_PER_BLOCK, 0, stream>>>(
          static_cast<int32_t *>(parameters.output_), num_elements, parameters.integer_start_,
          parameters.integer_step_);
      return;
    case DType::INT64:
      IntegerArangeKernel<<<block_count, ARANGE_THREADS_PER_BLOCK, 0, stream>>>(
          static_cast<int64_t *>(parameters.output_), num_elements, parameters.integer_start_,
          parameters.integer_step_);
      return;
    case DType::FLOAT32:
      FloatingArangeKernel<<<block_count, ARANGE_THREADS_PER_BLOCK, 0, stream>>>(
          static_cast<float *>(parameters.output_), num_elements, parameters.floating_start_,
          parameters.floating_step_);
      return;
    case DType::BOOL:
    case DType::UINT8:
    case DType::FLOAT16:
    case DType::BFLOAT16:
      break;
  }
  throw InternalError("unsupported dtype reached arange CUDA launcher", location);
}

}  // namespace ttl::internal
