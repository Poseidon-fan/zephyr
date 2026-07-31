#include "ttl/internal/indexing.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>
#include <utility>

#include <cuda_runtime.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/cuda_dtype.hpp"
#include "ttl/internal/device_error.cuh"
#include "ttl/internal/device_error.hpp"
#include "ttl/internal/index_width.hpp"
#include "ttl/shape.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t INDEXING_THREADS_PER_BLOCK = 256;
constexpr uint32_t MAXIMUM_INDEXING_BLOCKS = 65535;

template <typename T>
__device__ auto GetIndexBits(T value) -> uint64_t {
  if constexpr (std::same_as<T, int32_t>) {
    return static_cast<uint32_t>(value);
  } else {
    static_assert(std::same_as<T, int64_t>);
    return __builtin_bit_cast(uint64_t, value);
  }
}

template <typename Index>
[[nodiscard]] auto GetBlockCount(Index elements) noexcept -> uint32_t {
  const auto blocks = (static_cast<uint64_t>(elements) + INDEXING_THREADS_PER_BLOCK - 1) / INDEXING_THREADS_PER_BLOCK;
  return static_cast<uint32_t>(blocks < MAXIMUM_INDEXING_BLOCKS ? blocks : MAXIMUM_INDEXING_BLOCKS);
}

template <typename Destination>
auto ConvertIndexingParameters(const IndexingParameters64 &source, std::source_location location) -> Destination {
  using Index = std::remove_cvref_t<decltype(Destination::num_elements_)>;
  Destination destination{
      .output_ = source.output_,
      .input_ = source.input_,
      .indices_ = source.indices_,
      .num_elements_ = CheckedNarrow<Index>(source.num_elements_, "indexing element count", location),
      .axis_bound_ = source.axis_bound_,
      .rank_ = source.rank_,
      .axis_ = source.axis_,
  };
  for (size_t axis = 0; axis < TTL_MAX_RANK; ++axis) {
    destination.shape_[axis] = CheckedNarrow<Index>(source.shape_[axis], "indexing shape", location);
    destination.output_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.output_strides_bytes_[axis], "indexing output stride", location);
    destination.input_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.input_strides_bytes_[axis], "indexing input stride", location);
    destination.index_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.index_strides_bytes_[axis], "indexing index stride", location);
  }
  return destination;
}

template <typename Destination>
auto ConvertGatherRowsParameters(const GatherRowsParameters64 &source, std::source_location location) -> Destination {
  using Index = std::remove_cvref_t<decltype(Destination::num_elements_)>;
  Destination destination{
      .output_ = source.output_,
      .table_ = source.table_,
      .indices_ = source.indices_,
      .table_row_stride_bytes_ =
          CheckedNarrow<Index>(source.table_row_stride_bytes_, "gather rows table row stride", location),
      .num_elements_ = CheckedNarrow<Index>(source.num_elements_, "gather rows element count", location),
      .row_count_ = source.row_count_,
      .output_rank_ = source.output_rank_,
      .index_rank_ = source.index_rank_,
  };
  for (size_t axis = 0; axis < TTL_MAX_RANK; ++axis) {
    destination.output_shape_[axis] =
        CheckedNarrow<Index>(source.output_shape_[axis], "gather rows output shape", location);
    destination.output_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.output_strides_bytes_[axis], "gather rows output stride", location);
    destination.table_tail_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.table_tail_strides_bytes_[axis], "gather rows table stride", location);
    destination.index_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.index_strides_bytes_[axis], "gather rows index stride", location);
  }
  return destination;
}

template <CudaStorageType T, CudaStorageType IndexValue, typename Parameters>
__global__ void IndexingKernel(Parameters parameters, DeviceErrorLaunchContext error_context) {
  using Index = std::remove_cvref_t<decltype(parameters.num_elements_)>;
  auto linear_index =
      (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto step = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  while (linear_index < parameters.num_elements_) {
    auto remaining = linear_index;
    auto output_offset = Index{0};
    auto input_offset = Index{0};
    auto index_offset = Index{0};
    for (size_t remaining_rank = parameters.rank_; remaining_rank > 0; --remaining_rank) {
      const auto axis = remaining_rank - 1;
      const auto coordinate = static_cast<Index>(remaining % parameters.shape_[axis]);
      remaining = static_cast<Index>(remaining / parameters.shape_[axis]);
      output_offset = static_cast<Index>(output_offset + (coordinate * parameters.output_strides_bytes_[axis]));
      index_offset = static_cast<Index>(index_offset + (coordinate * parameters.index_strides_bytes_[axis]));
      if (axis != parameters.axis_) {
        input_offset = static_cast<Index>(input_offset + (coordinate * parameters.input_strides_bytes_[axis]));
      }
    }
    const auto selected = *reinterpret_cast<const IndexValue *>(parameters.indices_ + index_offset);
    if (selected < 0 || selected >= parameters.axis_bound_) {
      ReportDeviceError(error_context, DeviceErrorCode::INDEX_OUT_OF_BOUNDS, static_cast<int64_t>(linear_index),
                        GetIndexBits(selected), parameters.axis_bound_);
    } else {
      input_offset = static_cast<Index>(
          input_offset + (static_cast<Index>(selected) * parameters.input_strides_bytes_[parameters.axis_]));
      *reinterpret_cast<T *>(parameters.output_ + output_offset) =
          *reinterpret_cast<const T *>(parameters.input_ + input_offset);
    }
    if (step >= parameters.num_elements_ - linear_index) {
      break;
    }
    linear_index = static_cast<Index>(linear_index + step);
  }
}

template <CudaStorageType T, CudaStorageType IndexValue, typename Parameters>
__global__ void GatherRowsKernel(Parameters parameters, DeviceErrorLaunchContext error_context) {
  using Index = std::remove_cvref_t<decltype(parameters.num_elements_)>;
  auto linear_index =
      (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto step = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  while (linear_index < parameters.num_elements_) {
    auto remaining = linear_index;
    auto output_offset = Index{0};
    auto index_offset = Index{0};
    auto table_tail_offset = Index{0};
    for (size_t remaining_rank = parameters.output_rank_; remaining_rank > 0; --remaining_rank) {
      const auto axis = remaining_rank - 1;
      const auto coordinate = static_cast<Index>(remaining % parameters.output_shape_[axis]);
      remaining = static_cast<Index>(remaining / parameters.output_shape_[axis]);
      output_offset = static_cast<Index>(output_offset + (coordinate * parameters.output_strides_bytes_[axis]));
      if (axis < parameters.index_rank_) {
        index_offset = static_cast<Index>(index_offset + (coordinate * parameters.index_strides_bytes_[axis]));
      } else {
        table_tail_offset = static_cast<Index>(
            table_tail_offset + (coordinate * parameters.table_tail_strides_bytes_[axis - parameters.index_rank_]));
      }
    }
    const auto selected = *reinterpret_cast<const IndexValue *>(parameters.indices_ + index_offset);
    if (selected < 0 || selected >= parameters.row_count_) {
      ReportDeviceError(error_context, DeviceErrorCode::INDEX_OUT_OF_BOUNDS, static_cast<int64_t>(linear_index),
                        GetIndexBits(selected), parameters.row_count_);
    } else {
      const auto table_offset =
          static_cast<Index>((static_cast<Index>(selected) * parameters.table_row_stride_bytes_) + table_tail_offset);
      *reinterpret_cast<T *>(parameters.output_ + output_offset) =
          *reinterpret_cast<const T *>(parameters.table_ + table_offset);
    }
    if (step >= parameters.num_elements_ - linear_index) {
      break;
    }
    linear_index = static_cast<Index>(linear_index + step);
  }
}

template <CudaStorageType T, CudaStorageType IndexValue, typename Parameters>
void LaunchIndexingTyped(cudaStream_t stream, const Parameters &parameters,
                         const DeviceErrorLaunchContext &error_context) {
  IndexingKernel<T, IndexValue>
      <<<GetBlockCount(parameters.num_elements_), INDEXING_THREADS_PER_BLOCK, 0, stream>>>(parameters, error_context);
}

template <CudaStorageType T, CudaStorageType IndexValue, typename Parameters>
void LaunchGatherRowsTyped(cudaStream_t stream, const Parameters &parameters,
                           const DeviceErrorLaunchContext &error_context) {
  GatherRowsKernel<T, IndexValue>
      <<<GetBlockCount(parameters.num_elements_), INDEXING_THREADS_PER_BLOCK, 0, stream>>>(parameters, error_context);
}

template <typename Function>
void DispatchIndexType(DType index_dtype, Function &&function, std::source_location location) {
  switch (index_dtype) {
    case DType::INT32:
      std::forward<Function>(function)(std::type_identity<int32_t>{});
      return;
    case DType::INT64:
      std::forward<Function>(function)(std::type_identity<int64_t>{});
      return;
    case DType::BOOL:
    case DType::UINT8:
    case DType::FLOAT16:
    case DType::BFLOAT16:
    case DType::FLOAT32:
      break;
  }
  throw InternalError("indexing launcher received an unsupported index dtype", location);
}

}  // namespace

void LaunchIndexing(cudaStream_t stream, DType dtype, DType index_dtype, IndexWidth index_width,
                    const IndexingParameters64 &parameters, const DeviceErrorLaunchContext &error_context,
                    std::source_location location) {
  if (stream == nullptr || parameters.num_elements_ == 0 || error_context.record_ == nullptr) {
    throw InternalError("invalid indexing launch parameters", location);
  }
  DispatchCudaDType(dtype, "IndexingOut", [&]<CudaStorageType T>(std::type_identity<T>) {
    DispatchIndexType(
        index_dtype,
        [&]<CudaStorageType IndexValue>(std::type_identity<IndexValue>) {
          if (index_width == IndexWidth::UINT32) {
            LaunchIndexingTyped<T, IndexValue>(
                stream, ConvertIndexingParameters<IndexingParameters32>(parameters, location), error_context);
          } else {
            LaunchIndexingTyped<T, IndexValue>(
                stream, ConvertIndexingParameters<IndexingParameters64>(parameters, location), error_context);
          }
        },
        location);
  });
}

void LaunchGatherRows(cudaStream_t stream, DType dtype, DType index_dtype, IndexWidth index_width,
                      const GatherRowsParameters64 &parameters, const DeviceErrorLaunchContext &error_context,
                      std::source_location location) {
  if (stream == nullptr || parameters.num_elements_ == 0 || error_context.record_ == nullptr) {
    throw InternalError("invalid gather rows launch parameters", location);
  }
  DispatchCudaDType(dtype, "GatherRowsOut", [&]<CudaStorageType T>(std::type_identity<T>) {
    DispatchIndexType(
        index_dtype,
        [&]<CudaStorageType IndexValue>(std::type_identity<IndexValue>) {
          if (index_width == IndexWidth::UINT32) {
            LaunchGatherRowsTyped<T, IndexValue>(
                stream, ConvertGatherRowsParameters<GatherRowsParameters32>(parameters, location), error_context);
          } else {
            LaunchGatherRowsTyped<T, IndexValue>(
                stream, ConvertGatherRowsParameters<GatherRowsParameters64>(parameters, location), error_context);
          }
        },
        location);
  });
}

}  // namespace ttl::internal
