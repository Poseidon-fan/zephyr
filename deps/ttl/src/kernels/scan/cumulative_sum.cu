#include "ttl/internal/ops/scan.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

#include <ttl/runtime/cuda_dtype.hpp>
#include <ttl/runtime/cuda_math.cuh>
#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t SCAN_THREADS_PER_BLOCK = 256;
constexpr uint32_t MAXIMUM_SCAN_BLOCKS = 65535;

struct ScanRow final {
  int32_t axis_size_;

  __host__ __device__ auto operator()(int32_t index) const noexcept -> int32_t { return index / axis_size_; }
};

using ScanRowIterator = thrust::transform_iterator<ScanRow, thrust::counting_iterator<int32_t>>;

[[nodiscard]] auto MakeScanRows(int32_t axis_size) -> ScanRowIterator {
  return ScanRowIterator{thrust::counting_iterator<int32_t>{0}, ScanRow{axis_size}};
}

template <CudaStorageType T>
using ScanAccumulator = std::conditional_t<IsCudaFloatingType<T>(), float, uint64_t>;

/** Carries the preceding tiles' aggregate into each block scan. CUB broadcasts thread zero's returned prefix. */
template <typename Accumulator>
struct ScanPrefix final {
  Accumulator value_{0};

  __device__ auto operator()(Accumulator aggregate) -> Accumulator {
    const auto prefix = value_;
    value_ += aggregate;
    return prefix;
  }
};

// Integer scans accumulate in unsigned storage. Projection preserves the low N bits, giving defined modular arithmetic
// for signed types instead of relying on signed-overflow behavior in device code.
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
  // Each block owns an axis slice and scans it in tiles, keeping scratch independent of the axis length.
  using Accumulator = ScanAccumulator<T>;
  using BlockScan = cub::BlockScan<Accumulator, SCAN_THREADS_PER_BLOCK>;
  __shared__ typename BlockScan::TempStorage scan_storage;

  using Index = std::remove_cvref_t<decltype(parameters.slice_count_)>;
  auto slice = static_cast<Index>(blockIdx.x);
  const auto step = static_cast<Index>(gridDim.x);
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

    auto prefix = ScanPrefix<Accumulator>{};
    for (Index first = 0;; first = static_cast<Index>(first + SCAN_THREADS_PER_BLOCK)) {
      const auto axis_index = static_cast<Index>(first + static_cast<Index>(threadIdx.x));
      auto value = Accumulator{0};
      if (axis_index < parameters.axis_size_) {
        const auto input_offset =
            static_cast<Index>(input_base + (axis_index * parameters.input_strides_bytes_[parameters.axis_]));
        value = Lift(*reinterpret_cast<const T *>(parameters.input_ + input_offset));
      }
      BlockScan(scan_storage).InclusiveSum(value, value, prefix);
      if (axis_index < parameters.axis_size_) {
        const auto output_offset =
            static_cast<Index>(output_base + (axis_index * parameters.output_strides_bytes_[parameters.axis_]));
        *reinterpret_cast<T *>(parameters.output_ + output_offset) = Project<T>(value);
      }
      // All input values are loaded before their tile is overwritten; finish using shared storage before its reuse.
      __syncthreads();
      if (parameters.axis_size_ - first <= SCAN_THREADS_PER_BLOCK) {
        break;
      }
    }
    if (step >= parameters.slice_count_ - slice) {
      break;
    }
    slice = static_cast<Index>(slice + step);
  }
}

template <CudaStorageType T, typename Parameters>
void LaunchTyped(cudaStream_t stream, const Parameters &parameters) {
  const auto blocks = static_cast<uint64_t>(parameters.slice_count_);
  const auto block_count = static_cast<uint32_t>(blocks < MAXIMUM_SCAN_BLOCKS ? blocks : MAXIMUM_SCAN_BLOCKS);
  CumulativeSumKernel<T><<<block_count, SCAN_THREADS_PER_BLOCK, 0, stream>>>(parameters);
}

}  // namespace

auto GetCumulativeSumWorkspaceBytes(int32_t num_items, int32_t axis_size, std::source_location location) -> size_t {
  auto workspace_bytes = size_t{0};
  CheckCuda(
      cub::DeviceScan::InclusiveSumByKey(nullptr, workspace_bytes, MakeScanRows(axis_size),
                                         static_cast<const float *>(nullptr), static_cast<float *>(nullptr), num_items),
      "cub::DeviceScan::InclusiveSumByKey workspace query", location);
  return workspace_bytes;
}

void LaunchCumulativeSumContiguous(cudaStream_t stream, const CumulativeSumParameters64 &parameters, void *workspace,
                                   size_t workspace_bytes, std::source_location location) {
  // Generated row keys reset the prefix at each boundary without a key buffer or one launch per row.
  const auto num_items = static_cast<int32_t>(parameters.slice_count_ * parameters.axis_size_);
  const auto axis_size = static_cast<int32_t>(parameters.axis_size_);
  CheckCuda(cub::DeviceScan::InclusiveSumByKey(
                workspace, workspace_bytes, MakeScanRows(axis_size), reinterpret_cast<const float *>(parameters.input_),
                reinterpret_cast<float *>(parameters.output_), num_items, cub::Equality{}, stream),
            "cub::DeviceScan::InclusiveSumByKey", location);
}

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
