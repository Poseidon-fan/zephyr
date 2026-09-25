#include "layer/activation.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include <ttl/runtime/cuda_dtype.hpp>
#include <ttl/runtime/cuda_math.cuh>

namespace zephyr::layer {

struct SiluLayout final {
  int64_t dimensions_[ttl::TTL_MAX_RANK];
  int64_t strides_[ttl::TTL_MAX_RANK];
  size_t rank_;
};

template <ttl::CudaStorageType T>
__global__ void SiluKernel(const T *input, T *output, int64_t num_elements, SiluLayout layout) {
  for (auto index = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x; index < num_elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    auto remaining = index;
    int64_t offset = 0;
    for (auto axis = layout.rank_; axis > 0; --axis) {
      offset += (remaining % layout.dimensions_[axis - 1]) * layout.strides_[axis - 1];
      remaining /= layout.dimensions_[axis - 1];
    }
    // The packed gate projection may be strided. Round the activation before its product with the up projection.
    const auto value = ttl::ToElementwiseFloat(input[offset]);
    output[index] = static_cast<T>(value / (1.0F + expf(-value)));
  }
}

void LaunchSilu(cudaStream_t stream, ttl::DType dtype, const void *input, void *output, const ttl::Shape &shape,
                const ttl::Strides &strides) {
  const auto num_elements = shape.GetNumElements();
  if (num_elements == 0) {
    return;
  }
  auto layout = SiluLayout{};
  layout.rank_ = shape.GetRank();
  for (size_t axis = 0; axis < layout.rank_; ++axis) {
    layout.dimensions_[axis] = shape.GetDimension(axis);
    layout.strides_[axis] = strides.GetStride(axis);
  }
  constexpr int threads = 256;
  const auto blocks = static_cast<unsigned int>(std::min<int64_t>(((num_elements - 1) / threads) + 1, 65535));
  ttl::DispatchCudaFloatingDType(dtype, "Silu", [&]<typename T>(std::type_identity<T>) {
    SiluKernel<<<blocks, threads, 0, stream>>>(static_cast<const T *>(input), static_cast<T *>(output), num_elements,
                                               layout);
  });
}

}  // namespace zephyr::layer
