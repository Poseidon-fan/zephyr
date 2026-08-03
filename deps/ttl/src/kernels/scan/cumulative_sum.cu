#include "ttl/internal/ops/scan.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_math.cuh"
#include "ttl/internal/runtime/library/cuda_dtype.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t SCAN_THREADS_PER_BLOCK = 256;
constexpr uint32_t MAXIMUM_SCAN_BLOCKS = 65535;

template <CudaStorageType T>
using ScanAccumulator = std::conditional_t<IsCudaFloatingType<T>(), float, uint64_t>;

template <CudaStorageType T>
__device__ auto Lift(T value) -> ScanAccumulator<T> {
  if constexpr (IsCudaFloatingType<T>()) {
    return ToElementwiseFloat(value);
  } else if constexpr (std::same_as<T, uint8_t>) {
    return value;
  } else if constexpr (std::same_as<T, int32_t>) {
    return static_cast<uint32_t>(value);
  } else {
    static_assert(std::same_as<T, int64_t>);
    return __builtin_bit_cast(uint64_t, value);
  }
}

template <CudaStorageType T>
__device__ auto Project(ScanAccumulator<T> value) -> T {
  if constexpr (IsCudaFloatingType<T>()) {
    return FromElementwiseFloat<T>(value);
  } else if constexpr (std::same_as<T, uint8_t>) {
    return static_cast<uint8_t>(value);
  } else if constexpr (std::same_as<T, int32_t>) {
    return __builtin_bit_cast(int32_t, static_cast<uint32_t>(value));
  } else {
    static_assert(std::same_as<T, int64_t>);
    return __builtin_bit_cast(int64_t, value);
  }
}

template <typename Destination>
auto ConvertParameters(const CumulativeSumParameters64 &source, std::source_location location) -> Destination {
  using Index = std::remove_cvref_t<decltype(Destination::slice_count_)>;
  Destination destination{
      .output_ = source.output_,
      .input_ = source.input_,
      .slice_count_ = CheckedNarrow<Index>(source.slice_count_, "cumulative sum slice count", location),
      .axis_size_ = CheckedNarrow<Index>(source.axis_size_, "cumulative sum axis size", location),
      .rank_ = source.rank_,
      .axis_ = source.axis_,
  };
  for (size_t axis = 0; axis < TTL_MAX_RANK; ++axis) {
    destination.shape_[axis] = CheckedNarrow<Index>(source.shape_[axis], "cumulative sum shape", location);
    destination.output_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.output_strides_bytes_[axis], "cumulative sum output stride", location);
    destination.input_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.input_strides_bytes_[axis], "cumulative sum input stride", location);
  }
  return destination;
}

template <CudaStorageType T, typename Parameters>
__global__ void CumulativeSumKernel(Parameters parameters) {
  using Index = std::remove_cvref_t<decltype(parameters.slice_count_)>;
  auto slice = (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto step = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  while (slice < parameters.slice_count_) {
    auto remaining = slice;
    auto output_base = Index{0};
    auto input_base = Index{0};
    for (size_t remaining_rank = parameters.rank_; remaining_rank > 0; --remaining_rank) {
      const auto axis = remaining_rank - 1;
      if (axis == parameters.axis_) {
        continue;
      }
      const auto coordinate = static_cast<Index>(remaining % parameters.shape_[axis]);
      remaining = static_cast<Index>(remaining / parameters.shape_[axis]);
      output_base = static_cast<Index>(output_base + (coordinate * parameters.output_strides_bytes_[axis]));
      input_base = static_cast<Index>(input_base + (coordinate * parameters.input_strides_bytes_[axis]));
    }

    auto accumulator = ScanAccumulator<T>{0};
    for (Index axis_index = 0; axis_index < parameters.axis_size_; ++axis_index) {
      const auto input_offset =
          static_cast<Index>(input_base + (axis_index * parameters.input_strides_bytes_[parameters.axis_]));
      const auto output_offset =
          static_cast<Index>(output_base + (axis_index * parameters.output_strides_bytes_[parameters.axis_]));
      accumulator = static_cast<ScanAccumulator<T>>(
          accumulator + Lift(*reinterpret_cast<const T *>(parameters.input_ + input_offset)));
      *reinterpret_cast<T *>(parameters.output_ + output_offset) = Project<T>(accumulator);
    }
    if (step >= parameters.slice_count_ - slice) {
      break;
    }
    slice = static_cast<Index>(slice + step);
  }
}

template <CudaStorageType T, typename Parameters>
void LaunchTyped(cudaStream_t stream, const Parameters &parameters) {
  const auto blocks =
      (static_cast<uint64_t>(parameters.slice_count_) + SCAN_THREADS_PER_BLOCK - 1) / SCAN_THREADS_PER_BLOCK;
  const auto block_count = static_cast<uint32_t>(blocks < MAXIMUM_SCAN_BLOCKS ? blocks : MAXIMUM_SCAN_BLOCKS);
  CumulativeSumKernel<T><<<block_count, SCAN_THREADS_PER_BLOCK, 0, stream>>>(parameters);
}

}  // namespace

void LaunchCumulativeSum(cudaStream_t stream, DType dtype, IndexWidth index_width,
                         const CumulativeSumParameters64 &parameters, std::source_location location) {
  if (stream == nullptr || parameters.slice_count_ == 0 || parameters.axis_size_ == 0) {
    throw InternalError("invalid cumulative sum launch parameters", location);
  }
  DispatchCudaNumericDType(dtype, "CumulativeSumOut", [&]<CudaStorageType T>(std::type_identity<T>) {
    if (index_width == IndexWidth::UINT32) {
      LaunchTyped<T>(stream, ConvertParameters<CumulativeSumParameters32>(parameters, location));
    } else {
      LaunchTyped<T>(stream, ConvertParameters<CumulativeSumParameters64>(parameters, location));
    }
  });
}

}  // namespace ttl::internal
