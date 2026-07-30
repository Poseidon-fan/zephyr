#include "ttl/internal/elementwise_launch.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/internal/cuda_dtype.hpp"
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/internal/elementwise_kernel.cuh"
#include "ttl/scalar.hpp"

namespace ttl::internal {
namespace {

template <typename T, size_t width>
struct alignas(sizeof(T) * width) AlignedVector final {
  T values_[width];
};

template <typename T, size_t width, typename Index>
__global__ void ContiguousFillKernel(T *output, Index num_elements, T value) {
  using Vector = AlignedVector<T, width>;
  const auto vector_count = static_cast<Index>(num_elements / width);
  auto vector_index =
      (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto grid_stride = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);

  Vector vector_value;
#pragma unroll
  for (size_t element = 0; element < width; ++element) {
    vector_value.values_[element] = value;
  }

  auto *vector_output = reinterpret_cast<Vector *>(output);
  while (vector_index < vector_count) {
    vector_output[vector_index] = vector_value;
    vector_index = static_cast<Index>(vector_index + grid_stride);
  }

  auto tail_index = (vector_count * static_cast<Index>(width)) +
                    (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  while (tail_index < num_elements) {
    output[tail_index] = value;
    tail_index = static_cast<Index>(tail_index + grid_stride);
  }
}

template <typename T>
struct FillOperation final {
  T value_;

  template <typename Parameters>
  __device__ void operator()(const Parameters &parameters, ElementwiseIndexType<Parameters> index) const {
    *reinterpret_cast<T *>(GetElementwisePointer(parameters, 0, index)) = value_;
  }
};

template <typename T>
struct CopyOperation final {
  template <typename Parameters>
  __device__ void operator()(const Parameters &parameters, ElementwiseIndexType<Parameters> index) const {
    *reinterpret_cast<T *>(GetElementwisePointer(parameters, 0, index)) =
        *reinterpret_cast<const T *>(GetElementwisePointer(parameters, 1, index));
  }
};

template <CudaStorageType T>
[[nodiscard]] auto ConvertScalar(const Scalar &value, std::source_location location) -> T {
  using HostType = StorageTypeForT<CUDA_DTYPE_OF<T>>;
  static_assert(sizeof(HostType) == sizeof(T));
  return std::bit_cast<T>(value.Cast<HostType>(location));
}

template <typename T, size_t width, typename Parameters>
void LaunchContiguousFill(cudaStream_t stream, const Parameters &parameters, T value) {
  const auto work_items = static_cast<uint64_t>(parameters.num_elements_ / width);
  const auto block_count = GetElementwiseBlockCount(work_items == 0 ? 1 : work_items);
  auto *output = reinterpret_cast<T *>(parameters.pointers_[0]);
  ContiguousFillKernel<T, width>
      <<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(output, parameters.num_elements_, value);
}

template <typename T, typename Parameters>
void DispatchContiguousFillWidth(cudaStream_t stream, const Parameters &parameters, uint8_t width, T value,
                                 std::source_location location) {
  switch (width) {
    case 1:
      LaunchContiguousFill<T, 1>(stream, parameters, value);
      return;
    case 2:
      LaunchContiguousFill<T, 2>(stream, parameters, value);
      return;
    case 4:
      LaunchContiguousFill<T, 4>(stream, parameters, value);
      return;
    case 8:
      if constexpr (sizeof(T) <= 2) {
        LaunchContiguousFill<T, 8>(stream, parameters, value);
        return;
      }
      break;
    case 16:
      if constexpr (sizeof(T) == 1) {
        LaunchContiguousFill<T, 16>(stream, parameters, value);
        return;
      }
      break;
    default:
      break;
  }
  throw InternalError("elementwise iterator selected an invalid fill vector width", location);
}

template <typename T, typename Parameters>
void LaunchFillWithParameters(cudaStream_t stream, const Parameters &parameters, IteratorPath path, uint8_t width,
                              T value, std::source_location location) {
  if (path == IteratorPath::CONTIGUOUS) {
    DispatchContiguousFillWidth(stream, parameters, width, value, location);
    return;
  }

  const auto block_count = GetElementwiseBlockCount(parameters.num_elements_);
  ElementwiseKernel<<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(parameters, FillOperation<T>{value});
}

template <typename T, typename Parameters>
void LaunchCopyWithParameters(cudaStream_t stream, const Parameters &parameters) {
  const auto block_count = GetElementwiseBlockCount(parameters.num_elements_);
  ElementwiseKernel<<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(parameters, CopyOperation<T>{});
}

template <CudaStorageType T>
void LaunchTypedFill(cudaStream_t stream, const ElementwiseIterator &iterator, const Scalar &value,
                     std::source_location location) {
  const auto converted = ConvertScalar<T>(value, location);
  if (iterator.GetIndexWidth() == IndexWidth::UINT32) {
    LaunchFillWithParameters(stream, iterator.MakeParameters32(location), iterator.GetPath(),
                             iterator.GetVectorWidthElements(), converted, location);
    return;
  }
  LaunchFillWithParameters(stream, iterator.MakeParameters64(), iterator.GetPath(), iterator.GetVectorWidthElements(),
                           converted, location);
}

template <CudaStorageType T>
void LaunchTypedCopy(cudaStream_t stream, const ElementwiseIterator &iterator, std::source_location location) {
  if (iterator.GetIndexWidth() == IndexWidth::UINT32) {
    LaunchCopyWithParameters<T>(stream, iterator.MakeParameters32(location));
    return;
  }
  LaunchCopyWithParameters<T>(stream, iterator.MakeParameters64());
}

}  // namespace

void LaunchFill(cudaStream_t stream, DType dtype, const ElementwiseIterator &iterator, const Scalar &value,
                std::source_location location) {
  if (stream == nullptr || iterator.GetOperandCount() != 1 || iterator.GetNumElements() <= 0) {
    throw InternalError("invalid elementwise fill launch plan", location);
  }
  DispatchCudaDType(dtype, "FillOut", [&]<CudaStorageType T>(std::type_identity<T>) {
    LaunchTypedFill<T>(stream, iterator, value, location);
  });
}

void LaunchCopy(cudaStream_t stream, DType dtype, const ElementwiseIterator &iterator, std::source_location location) {
  if (stream == nullptr || iterator.GetOperandCount() != 2 || iterator.GetNumElements() <= 0) {
    throw InternalError("invalid elementwise copy launch plan", location);
  }
  DispatchCudaDType(dtype, "CopyOut",
                    [&]<CudaStorageType T>(std::type_identity<T>) { LaunchTypedCopy<T>(stream, iterator, location); });
}

}  // namespace ttl::internal
