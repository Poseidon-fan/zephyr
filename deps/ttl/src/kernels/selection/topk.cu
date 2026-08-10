#include "ttl/internal/ops/topk.hpp"

#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_math.cuh"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/library/cuda_dtype.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t TOPK_THREADS_PER_BLOCK = 256;
constexpr uint32_t TOPK_SMALL_CAPACITY = 1024;
constexpr uint32_t MAXIMUM_TOPK_BLOCKS = 65535;
constexpr uint32_t MAXIMUM_UINT32 = ~uint32_t{0};
constexpr uint64_t MAXIMUM_UINT64 = ~uint64_t{0};
constexpr int64_t MAXIMUM_INT64 = INT64_MAX;

struct SegmentOffset final {
  int32_t axis_size_;

  __host__ __device__ auto operator()(int32_t segment) const noexcept -> int32_t { return segment * axis_size_; }
};

using SegmentCountingIterator = thrust::counting_iterator<int32_t>;
using SegmentOffsetIterator = thrust::transform_iterator<SegmentOffset, SegmentCountingIterator>;

[[nodiscard]] auto MakeSegmentOffsetIterator(int32_t start, int32_t axis_size) -> SegmentOffsetIterator {
  return SegmentOffsetIterator{SegmentCountingIterator{start}, SegmentOffset{axis_size}};
}

template <CudaStorageType T>
__device__ auto GetOrderedKey(T value) noexcept -> uint64_t {
  // Map every supported scalar to an unsigned key whose natural order matches TopK order. Both float zeros share a key,
  // and NaNs map above all numeric values; the source index supplies the stable tie break.
  if constexpr (CUDA_DTYPE_OF<T> == DType::UINT8) {
    return value;
  } else if constexpr (CUDA_DTYPE_OF<T> == DType::INT32) {
    return static_cast<uint32_t>(value) ^ uint32_t { 0x80000000U };
  } else if constexpr (CUDA_DTYPE_OF<T> == DType::INT64) {
    return static_cast<uint64_t>(value) ^ uint64_t { 0x8000000000000000ULL };
  } else {
    const auto converted = ToElementwiseFloat(value);
    if (isnan(converted)) {
      return MAXIMUM_UINT32;
    }
    auto bits = converted == 0.0F ? uint32_t{0} : __builtin_bit_cast(uint32_t, converted);
    bits = (bits & uint32_t{0x80000000U}) != 0 ? ~bits : bits ^ uint32_t{0x80000000U};
    return bits;
  }
}

__device__ auto ComesBefore(uint64_t lhs_key, int64_t lhs_index, uint64_t rhs_key, int64_t rhs_index,
                            bool largest) noexcept -> bool {
  if (lhs_key != rhs_key) {
    return largest ? lhs_key > rhs_key : lhs_key < rhs_key;
  }
  return lhs_index < rhs_index;
}

__device__ auto GetSliceOffset(const TopKParameters &parameters, uint64_t slice, const uint64_t *strides) noexcept
    -> uint64_t {
  auto offset = uint64_t{0};
  for (size_t remaining_rank = parameters.rank_; remaining_rank > 0; --remaining_rank) {
    const auto axis = remaining_rank - 1;
    if (axis == parameters.axis_) {
      continue;
    }
    const auto coordinate = slice % parameters.shape_[axis];
    slice /= parameters.shape_[axis];
    offset += coordinate * strides[axis];
  }
  return offset;
}

template <CudaStorageType T>
__device__ void WriteTopKResult(const TopKParameters &parameters, uint64_t slice, uint64_t output_index,
                                int64_t selected_index) {
  const auto value_base = GetSliceOffset(parameters, slice, parameters.value_strides_bytes_);
  const auto index_base = GetSliceOffset(parameters, slice, parameters.index_strides_elements_);
  const auto input_base = GetSliceOffset(parameters, slice, parameters.input_strides_bytes_);
  const auto input_offset =
      input_base + (static_cast<uint64_t>(selected_index) * parameters.input_strides_bytes_[parameters.axis_]);
  const auto value_offset = value_base + (output_index * parameters.value_strides_bytes_[parameters.axis_]);
  const auto index_offset = index_base + (output_index * parameters.index_strides_elements_[parameters.axis_]);
  *reinterpret_cast<T *>(parameters.values_ + value_offset) =
      *reinterpret_cast<const T *>(parameters.input_ + input_offset);
  parameters.indices_[index_offset] = selected_index;
}

template <CudaStorageType T>
__global__ void SmallTopKKernel(TopKParameters parameters) {
  __shared__ uint64_t keys[TOPK_SMALL_CAPACITY];
  __shared__ int64_t source_indices[TOPK_SMALL_CAPACITY];

  for (auto slice = static_cast<uint64_t>(blockIdx.x); slice < parameters.slice_count_; slice += gridDim.x) {
    const auto input_base = GetSliceOffset(parameters, slice, parameters.input_strides_bytes_);
    for (auto index = static_cast<uint64_t>(threadIdx.x); index < TOPK_SMALL_CAPACITY; index += blockDim.x) {
      if (index < parameters.axis_size_) {
        const auto input_offset = input_base + (index * parameters.input_strides_bytes_[parameters.axis_]);
        keys[index] = GetOrderedKey(*reinterpret_cast<const T *>(parameters.input_ + input_offset));
        source_indices[index] = static_cast<int64_t>(index);
      } else {
        keys[index] = parameters.largest_ ? uint64_t{0} : MAXIMUM_UINT64;
        source_indices[index] = MAXIMUM_INT64;
      }
    }
    __syncthreads();

    // Sort a fixed power-of-two shared-memory array. Sentinel entries pad short axes without entering the first k
    // items.
    for (uint32_t stage = 2; stage <= TOPK_SMALL_CAPACITY; stage <<= 1U) {
      for (uint32_t distance = stage >> 1U; distance > 0; distance >>= 1U) {
        for (auto index = static_cast<uint32_t>(threadIdx.x); index < TOPK_SMALL_CAPACITY; index += blockDim.x) {
          const auto partner = index ^ distance;
          if (partner <= index) {
            continue;
          }
          const auto forward = (index & stage) == 0;
          const auto left_before = ComesBefore(keys[index], source_indices[index], keys[partner],
                                               source_indices[partner], parameters.largest_);
          if ((forward && !left_before) || (!forward && left_before)) {
            const auto key = keys[index];
            const auto source_index = source_indices[index];
            keys[index] = keys[partner];
            source_indices[index] = source_indices[partner];
            keys[partner] = key;
            source_indices[partner] = source_index;
          }
        }
        __syncthreads();
      }
    }

    for (auto index = static_cast<uint64_t>(threadIdx.x); index < parameters.k_; index += blockDim.x) {
      WriteTopKResult<T>(parameters, slice, index, source_indices[index]);
    }
    __syncthreads();
  }
}

template <CudaStorageType T>
__global__ void InitializeSortInputKernel(TopKParameters parameters, uint64_t *keys, int64_t *indices) {
  auto linear_index = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const auto num_items = parameters.sort_item_count_;
  const auto step = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (linear_index < num_items) {
    const auto slice = linear_index / parameters.axis_size_;
    const auto axis_index = linear_index % parameters.axis_size_;
    const auto input_offset = GetSliceOffset(parameters, slice, parameters.input_strides_bytes_) +
                              (axis_index * parameters.input_strides_bytes_[parameters.axis_]);
    keys[linear_index] = GetOrderedKey(*reinterpret_cast<const T *>(parameters.input_ + input_offset));
    indices[linear_index] = static_cast<int64_t>(axis_index);
    linear_index += step;
  }
}

template <CudaStorageType T>
__global__ void GatherSortedTopKKernel(TopKParameters parameters, const int64_t *sorted_indices) {
  auto linear_index = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const auto output_items = parameters.output_item_count_;
  const auto step = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (linear_index < output_items) {
    const auto slice = linear_index / parameters.k_;
    const auto output_index = linear_index % parameters.k_;
    const auto selected_index = sorted_indices[(slice * parameters.axis_size_) + output_index];
    WriteTopKResult<T>(parameters, slice, output_index, selected_index);
    linear_index += step;
  }
}

template <CudaStorageType T>
__global__ void SerialTopKKernel(TopKParameters parameters) {
  auto slice = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const auto step = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  while (slice < parameters.slice_count_) {
    const auto input_base = GetSliceOffset(parameters, slice, parameters.input_strides_bytes_);
    const auto index_base = GetSliceOffset(parameters, slice, parameters.index_strides_elements_);
    for (uint64_t output_index = 0; output_index < parameters.k_; ++output_index) {
      auto best_key = parameters.largest_ ? uint64_t{0} : MAXIMUM_UINT64;
      auto best_index = MAXIMUM_INT64;
      for (uint64_t candidate = 0; candidate < parameters.axis_size_; ++candidate) {
        auto already_selected = false;
        for (uint64_t previous = 0; previous < output_index; ++previous) {
          const auto previous_offset = index_base + (previous * parameters.index_strides_elements_[parameters.axis_]);
          if (parameters.indices_[previous_offset] == static_cast<int64_t>(candidate)) {
            already_selected = true;
            break;
          }
        }
        if (already_selected) {
          continue;
        }
        const auto input_offset = input_base + (candidate * parameters.input_strides_bytes_[parameters.axis_]);
        const auto key = GetOrderedKey(*reinterpret_cast<const T *>(parameters.input_ + input_offset));
        if (best_index == MAXIMUM_INT64 ||
            ComesBefore(key, static_cast<int64_t>(candidate), best_key, best_index, parameters.largest_)) {
          best_key = key;
          best_index = static_cast<int64_t>(candidate);
        }
      }
      WriteTopKResult<T>(parameters, slice, output_index, best_index);
    }
    slice += step;
  }
}

[[nodiscard]] auto GetBlockCount(uint64_t work_items, std::source_location location) -> uint32_t {
  const auto blocks =
      CeilDivide(work_items, static_cast<uint64_t>(TOPK_THREADS_PER_BLOCK), "TopK block count", location);
  return static_cast<uint32_t>(blocks < MAXIMUM_TOPK_BLOCKS ? blocks : MAXIMUM_TOPK_BLOCKS);
}

template <CudaStorageType T>
void LaunchSmallTyped(cudaStream_t stream, const TopKParameters &parameters) {
  const auto blocks = static_cast<uint32_t>(parameters.slice_count_ < MAXIMUM_TOPK_BLOCKS ? parameters.slice_count_
                                                                                          : MAXIMUM_TOPK_BLOCKS);
  SmallTopKKernel<T><<<blocks, TOPK_THREADS_PER_BLOCK, 0, stream>>>(parameters);
}

template <CudaStorageType T>
void LaunchSortTyped(cudaStream_t stream, const TopKParameters &parameters, uint64_t *keys_input, uint64_t *keys_output,
                     int64_t *indices_input, int64_t *indices_output, void *workspace, size_t workspace_bytes,
                     std::source_location location) {
  auto launch_parameters = parameters;
  launch_parameters.sort_item_count_ =
      CheckedMultiply(parameters.slice_count_, parameters.axis_size_, "TopK sort item count", location);
  launch_parameters.output_item_count_ =
      CheckedMultiply(parameters.slice_count_, parameters.k_, "TopK output item count", location);
  const auto num_items = launch_parameters.sort_item_count_;
  const auto narrowed_num_items = CheckedNarrow<int32_t>(num_items, "TopK item count", location);
  const auto narrowed_segments = CheckedNarrow<int32_t>(parameters.slice_count_, "TopK segment count", location);
  const auto narrowed_axis_size = CheckedNarrow<int32_t>(parameters.axis_size_, "TopK axis size", location);
  InitializeSortInputKernel<T><<<GetBlockCount(num_items, location), TOPK_THREADS_PER_BLOCK, 0, stream>>>(
      launch_parameters, keys_input, indices_input);
  const auto begin_offsets = MakeSegmentOffsetIterator(0, narrowed_axis_size);
  const auto end_offsets = MakeSegmentOffsetIterator(1, narrowed_axis_size);
  auto required_bytes = workspace_bytes;
  // Each logical slice is one CUB segment. Sorting (key, source-index) pairs preserves the lowest source index for
  // equal keys because CUB's radix sort is stable and indices enter in ascending order.
  const auto status =
      parameters.largest_
          ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
                workspace, required_bytes, keys_input, keys_output, indices_input, indices_output, narrowed_num_items,
                narrowed_segments, begin_offsets, end_offsets, 0, sizeof(uint64_t) * 8, stream)
          : cub::DeviceSegmentedRadixSort::SortPairs(workspace, required_bytes, keys_input, keys_output, indices_input,
                                                     indices_output, narrowed_num_items, narrowed_segments,
                                                     begin_offsets, end_offsets, 0, sizeof(uint64_t) * 8, stream);
  CheckCuda(status, "cub::DeviceSegmentedRadixSort::SortPairs", location);
  const auto output_items = launch_parameters.output_item_count_;
  GatherSortedTopKKernel<T>
      <<<GetBlockCount(output_items, location), TOPK_THREADS_PER_BLOCK, 0, stream>>>(launch_parameters, indices_output);
}

template <CudaStorageType T>
void LaunchSerialTyped(cudaStream_t stream, const TopKParameters &parameters, std::source_location location) {
  // This path needs no temporary storage and is intentionally quadratic in k; it is the fallback when a sort workspace
  // cannot be represented or reserved by the host plan.
  SerialTopKKernel<T>
      <<<GetBlockCount(parameters.slice_count_, location), TOPK_THREADS_PER_BLOCK, 0, stream>>>(parameters);
}

}  // namespace

auto GetTopKSortWorkspaceBytes(int32_t num_items, int32_t num_segments, int32_t axis_size, bool largest,
                               std::source_location location) -> size_t {
  if (num_items <= 0 || num_segments <= 0 || axis_size <= 0) {
    throw InternalError("invalid TopK workspace query parameters", location);
  }
  const auto begin_offsets = MakeSegmentOffsetIterator(0, axis_size);
  const auto end_offsets = MakeSegmentOffsetIterator(1, axis_size);
  auto workspace_bytes = size_t{0};
  const auto status = largest
                          ? cub::DeviceSegmentedRadixSort::SortPairsDescending(
                                nullptr, workspace_bytes, static_cast<const uint64_t *>(nullptr),
                                static_cast<uint64_t *>(nullptr), static_cast<const int64_t *>(nullptr),
                                static_cast<int64_t *>(nullptr), num_items, num_segments, begin_offsets, end_offsets)
                          : cub::DeviceSegmentedRadixSort::SortPairs(
                                nullptr, workspace_bytes, static_cast<const uint64_t *>(nullptr),
                                static_cast<uint64_t *>(nullptr), static_cast<const int64_t *>(nullptr),
                                static_cast<int64_t *>(nullptr), num_items, num_segments, begin_offsets, end_offsets);
  CheckCuda(status, "cub::DeviceSegmentedRadixSort::SortPairs workspace query", location);
  return workspace_bytes;
}

void LaunchTopKSmall(cudaStream_t stream, DType dtype, const TopKParameters &parameters,
                     std::source_location location) {
  if (stream == nullptr || parameters.slice_count_ == 0 || parameters.axis_size_ == 0 ||
      parameters.axis_size_ > TOPK_SMALL_CAPACITY || parameters.k_ == 0) {
    throw InternalError("invalid small TopK launch parameters", location);
  }
  DispatchCudaNumericDType(
      dtype, "TopKOut", [&]<CudaStorageType T>(std::type_identity<T>) { LaunchSmallTyped<T>(stream, parameters); },
      location);
}

void LaunchTopKSort(cudaStream_t stream, DType dtype, const TopKParameters &parameters, uint64_t *keys_input,
                    uint64_t *keys_output, int64_t *indices_input, int64_t *indices_output, void *workspace,
                    size_t workspace_bytes, std::source_location location) {
  if (stream == nullptr || keys_input == nullptr || keys_output == nullptr || indices_input == nullptr ||
      indices_output == nullptr || workspace == nullptr || workspace_bytes == 0 || parameters.slice_count_ == 0 ||
      parameters.axis_size_ == 0 || parameters.k_ == 0 || parameters.k_ > parameters.axis_size_) {
    throw InternalError("invalid sorted TopK launch parameters", location);
  }
  DispatchCudaNumericDType(
      dtype, "TopKOut",
      [&]<CudaStorageType T>(std::type_identity<T>) {
        LaunchSortTyped<T>(stream, parameters, keys_input, keys_output, indices_input, indices_output, workspace,
                           workspace_bytes, location);
      },
      location);
}

void LaunchTopKSerial(cudaStream_t stream, DType dtype, const TopKParameters &parameters,
                      std::source_location location) {
  if (stream == nullptr || parameters.slice_count_ == 0 || parameters.axis_size_ == 0 || parameters.k_ == 0) {
    throw InternalError("invalid serial TopK launch parameters", location);
  }
  DispatchCudaNumericDType(
      dtype, "TopKOut",
      [&]<CudaStorageType T>(std::type_identity<T>) { LaunchSerialTyped<T>(stream, parameters, location); }, location);
}

}  // namespace ttl::internal
