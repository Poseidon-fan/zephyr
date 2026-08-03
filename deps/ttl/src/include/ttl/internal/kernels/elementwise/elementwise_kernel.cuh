#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

namespace ttl::internal {

inline constexpr uint32_t ELEMENTWISE_THREADS_PER_BLOCK = 256;
inline constexpr uint32_t ELEMENTWISE_MAX_BLOCK_COUNT = 65535;

template <typename Parameters>
using ElementwiseIndexType = std::remove_cvref_t<decltype(Parameters::num_elements_)>;

template <typename Parameters>
__device__ auto GetElementwisePointer(const Parameters &parameters, uint8_t operand,
                                      ElementwiseIndexType<Parameters> linear_index) -> std::byte * {
  using Index = ElementwiseIndexType<Parameters>;
  auto byte_offset = Index{0};
  for (auto axis = static_cast<uint8_t>(parameters.rank_); axis > 0; --axis) {
    const auto index = static_cast<uint8_t>(axis - 1);
    const auto coordinate = static_cast<Index>(linear_index % parameters.shape_[index]);
    linear_index = static_cast<Index>(linear_index / parameters.shape_[index]);
    byte_offset = static_cast<Index>(byte_offset + coordinate * parameters.strides_bytes_[operand][index]);
  }
  return parameters.pointers_[operand] + byte_offset;
}

template <typename Parameters, typename Operation>
__global__ void ElementwiseKernel(Parameters parameters, Operation operation) {
  using Index = ElementwiseIndexType<Parameters>;
  auto index = (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto grid_stride = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  while (index < parameters.num_elements_) {
    operation(parameters, index);
    index = static_cast<Index>(index + grid_stride);
  }
}

[[nodiscard]] inline auto GetElementwiseBlockCount(uint64_t work_items) noexcept -> uint32_t {
  const auto blocks = (work_items + ELEMENTWISE_THREADS_PER_BLOCK - 1) / ELEMENTWISE_THREADS_PER_BLOCK;
  return static_cast<uint32_t>(blocks < ELEMENTWISE_MAX_BLOCK_COUNT ? blocks : ELEMENTWISE_MAX_BLOCK_COUNT);
}

}  // namespace ttl::internal
