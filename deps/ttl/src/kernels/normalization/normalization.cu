#include "ttl/internal/ops/normalization.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/common/index_width.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_math.cuh"
#include "ttl/internal/kernels/reduction/rowwise_kernel.cuh"
#include "ttl/internal/ops/rowwise.hpp"
#include "ttl/internal/runtime/library/cuda_dtype.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t NORMALIZATION_THREADS_PER_BLOCK = 256;
constexpr uint32_t NORMALIZATION_WARP_SIZE = 32;
constexpr uint32_t NORMALIZATION_WARPS_PER_BLOCK = NORMALIZATION_THREADS_PER_BLOCK / NORMALIZATION_WARP_SIZE;

struct alignas(8) WelfordStatistics final {
  float mean_;
  float squared_deviation_;
  uint64_t count_;
};

struct WelfordOperation final {
  using Accumulator = WelfordStatistics;

  __device__ auto Identity() const -> Accumulator { return {}; }
  __device__ auto Lift(float value) const -> Accumulator {
    return {.mean_ = value, .squared_deviation_ = 0.0F, .count_ = 1};
  }
  __device__ auto operator()(Accumulator lhs, Accumulator rhs) const -> Accumulator {
    if (lhs.count_ == 0) {
      return rhs;
    }
    if (rhs.count_ == 0) {
      return lhs;
    }
    const auto count = lhs.count_ + rhs.count_;
    const auto delta = rhs.mean_ - lhs.mean_;
    const auto rhs_fraction = static_cast<float>(rhs.count_) / static_cast<float>(count);
    const auto cross = delta * delta * (static_cast<float>(lhs.count_) * rhs_fraction);
    return {
        .mean_ = lhs.mean_ + (delta * rhs_fraction),
        .squared_deviation_ = lhs.squared_deviation_ + rhs.squared_deviation_ + cross,
        .count_ = count,
    };
  }
  __device__ auto Normalize(float value, Accumulator statistics, float epsilon) const -> float {
    const auto variance = statistics.squared_deviation_ / static_cast<float>(statistics.count_);
    return (value - statistics.mean_) * rsqrtf(variance + epsilon);
  }
};

struct alignas(8) ScaledSquaresStatistics final {
  float scale_;
  float scaled_squares_;
  uint64_t count_;
};

struct ScaledSquaresOperation final {
  using Accumulator = ScaledSquaresStatistics;

  __device__ auto Identity() const -> Accumulator { return {}; }
  __device__ auto Lift(float value) const -> Accumulator {
    const auto magnitude = fabsf(value);
    if (isnan(magnitude)) {
      return {.scale_ = CUDART_NAN_F, .scaled_squares_ = CUDART_NAN_F, .count_ = 1};
    }
    if (magnitude == 0.0F) {
      return {.scale_ = 0.0F, .scaled_squares_ = 0.0F, .count_ = 1};
    }
    return {.scale_ = magnitude, .scaled_squares_ = 1.0F, .count_ = 1};
  }
  __device__ auto operator()(Accumulator lhs, Accumulator rhs) const -> Accumulator {
    if (lhs.count_ == 0) {
      return rhs;
    }
    if (rhs.count_ == 0) {
      return lhs;
    }
    const auto count = lhs.count_ + rhs.count_;
    if (isnan(lhs.scale_) || isnan(rhs.scale_)) {
      return {.scale_ = CUDART_NAN_F, .scaled_squares_ = CUDART_NAN_F, .count_ = count};
    }
    if (isinf(lhs.scale_) || isinf(rhs.scale_)) {
      return {.scale_ = CUDART_INF_F, .scaled_squares_ = 1.0F, .count_ = count};
    }
    if (lhs.scale_ == 0.0F) {
      rhs.count_ = count;
      return rhs;
    }
    if (rhs.scale_ == 0.0F) {
      lhs.count_ = count;
      return lhs;
    }
    if (lhs.scale_ >= rhs.scale_) {
      const auto ratio = rhs.scale_ / lhs.scale_;
      return {
          .scale_ = lhs.scale_,
          .scaled_squares_ = lhs.scaled_squares_ + (rhs.scaled_squares_ * ratio * ratio),
          .count_ = count,
      };
    }
    const auto ratio = lhs.scale_ / rhs.scale_;
    return {
        .scale_ = rhs.scale_,
        .scaled_squares_ = rhs.scaled_squares_ + (lhs.scaled_squares_ * ratio * ratio),
        .count_ = count,
    };
  }
  __device__ auto Normalize(float value, Accumulator statistics, float epsilon) const -> float {
    if (statistics.scale_ == 0.0F) {
      return value * rsqrtf(epsilon);
    }
    if (isnan(statistics.scale_)) {
      return CUDART_NAN_F;
    }
    const auto scaled_epsilon = sqrtf(epsilon) / statistics.scale_;
    const auto scaled_mean_square =
        (statistics.scaled_squares_ / static_cast<float>(statistics.count_)) + (scaled_epsilon * scaled_epsilon);
    return (value / statistics.scale_) * rsqrtf(scaled_mean_square);
  }
};

template <typename Index>
struct NormalizationParameters final {
  RowwiseParameters<Index> rowwise_;
  NormalizationAuxiliaryParameters<Index> auxiliary_;
};

template <typename Index>
__device__ auto GetAuxiliaryOffset(const RowwiseParameters<Index> &rowwise, const Index (&strides)[TTL_MAX_RANK],
                                   Index reduction_index) -> Index {
  auto offset = Index{0};
  for (size_t remaining = rowwise.reduction_rank_; remaining > 0; --remaining) {
    const auto axis = remaining - 1;
    const auto coordinate = static_cast<Index>(reduction_index % rowwise.reduction_shape_[axis]);
    reduction_index = static_cast<Index>(reduction_index / rowwise.reduction_shape_[axis]);
    offset = static_cast<Index>(offset + (coordinate * strides[axis]));
  }
  return offset;
}

template <CudaStorageType T, typename Operation, typename Parameters>
__device__ auto ReduceNormalizationThread(const Parameters &parameters,
                                          RowwiseIndexType<decltype(parameters.rowwise_)> group_index,
                                          RowwiseIndexType<decltype(parameters.rowwise_)> first_index,
                                          RowwiseIndexType<decltype(parameters.rowwise_)> step, Operation operation) ->
    typename Operation::Accumulator {
  using Index = RowwiseIndexType<decltype(parameters.rowwise_)>;
  const auto &rowwise = parameters.rowwise_;
  const auto group_offset = GetRowwiseGroupInputOffset(rowwise, group_index);
  auto statistics = operation.Identity();
  for (auto index = first_index; index < rowwise.reduction_count_; index = static_cast<Index>(index + step)) {
    const auto offset = GetRowwiseReductionOffset(rowwise, index, false);
    const auto value = ToElementwiseFloat(*reinterpret_cast<const T *>(rowwise.input_ + group_offset + offset));
    statistics = operation(statistics, operation.Lift(value));
    if (step >= rowwise.reduction_count_ - index) {
      break;
    }
  }
  return statistics;
}

template <CudaStorageType T, typename Operation, typename Parameters>
__device__ void StoreNormalizationValues(const Parameters &parameters,
                                         RowwiseIndexType<decltype(parameters.rowwise_)> group_index,
                                         RowwiseIndexType<decltype(parameters.rowwise_)> first_index,
                                         RowwiseIndexType<decltype(parameters.rowwise_)> step,
                                         const typename Operation::Accumulator &statistics, Operation operation) {
  using Index = RowwiseIndexType<decltype(parameters.rowwise_)>;
  const auto &rowwise = parameters.rowwise_;
  const auto &auxiliary = parameters.auxiliary_;
  const auto input_group_offset = GetRowwiseGroupInputOffset(rowwise, group_index);
  const auto output_group_offset = GetRowwiseGroupOutputOffset(rowwise, group_index);
  for (auto index = first_index; index < rowwise.reduction_count_; index = static_cast<Index>(index + step)) {
    const auto input_offset = GetRowwiseReductionOffset(rowwise, index, false);
    const auto output_offset = GetRowwiseReductionOffset(rowwise, index, true);
    const auto value =
        ToElementwiseFloat(*reinterpret_cast<const T *>(rowwise.input_ + input_group_offset + input_offset));
    auto result = operation.Normalize(value, statistics, auxiliary.epsilon_);
    if (auxiliary.weight_ != nullptr) {
      const auto weight_offset = GetAuxiliaryOffset(rowwise, auxiliary.weight_strides_bytes_, index);
      result *= ToElementwiseFloat(*reinterpret_cast<const T *>(auxiliary.weight_ + weight_offset));
    }
    if (auxiliary.bias_ != nullptr) {
      const auto bias_offset = GetAuxiliaryOffset(rowwise, auxiliary.bias_strides_bytes_, index);
      result += ToElementwiseFloat(*reinterpret_cast<const T *>(auxiliary.bias_ + bias_offset));
    }
    *reinterpret_cast<T *>(rowwise.output_ + output_group_offset + output_offset) = FromElementwiseFloat<T>(result);
    if (step >= rowwise.reduction_count_ - index) {
      break;
    }
  }
}

template <CudaStorageType T, typename Operation, typename Parameters>
__global__ void WarpNormalizationKernel(Parameters parameters, Operation operation) {
  using Accumulator = typename Operation::Accumulator;
  using WarpReduce = cub::WarpReduce<Accumulator>;
  __shared__ typename WarpReduce::TempStorage warp_storage[NORMALIZATION_WARPS_PER_BLOCK];
  __shared__ Accumulator shared_statistics[NORMALIZATION_WARPS_PER_BLOCK];

  using Index = RowwiseIndexType<decltype(parameters.rowwise_)>;
  const auto lane = static_cast<uint32_t>(threadIdx.x) % NORMALIZATION_WARP_SIZE;
  const auto warp = static_cast<uint32_t>(threadIdx.x) / NORMALIZATION_WARP_SIZE;
  auto group_index = static_cast<Index>((static_cast<uint64_t>(blockIdx.x) * NORMALIZATION_WARPS_PER_BLOCK) + warp);
  const auto group_step = static_cast<Index>(static_cast<uint64_t>(gridDim.x) * NORMALIZATION_WARPS_PER_BLOCK);
  while (group_index < parameters.rowwise_.group_count_) {
    const auto local = ReduceNormalizationThread<T>(parameters, group_index, static_cast<Index>(lane),
                                                    static_cast<Index>(NORMALIZATION_WARP_SIZE), operation);
    const auto aggregate = WarpReduce(warp_storage[warp]).Reduce(local, operation);
    if (lane == 0) {
      shared_statistics[warp] = aggregate;
    }
    __syncwarp();
    StoreNormalizationValues<T>(parameters, group_index, static_cast<Index>(lane),
                                static_cast<Index>(NORMALIZATION_WARP_SIZE), shared_statistics[warp], operation);
    __syncwarp();
    if (group_step >= parameters.rowwise_.group_count_ - group_index) {
      break;
    }
    group_index = static_cast<Index>(group_index + group_step);
  }
}

template <CudaStorageType T, typename Operation, typename Parameters>
__global__ void BlockNormalizationKernel(Parameters parameters, Operation operation) {
  using Accumulator = typename Operation::Accumulator;
  using BlockReduce = cub::BlockReduce<Accumulator, NORMALIZATION_THREADS_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage block_storage;
  __shared__ Accumulator shared_statistics;

  using Index = RowwiseIndexType<decltype(parameters.rowwise_)>;
  auto group_index = static_cast<Index>(blockIdx.x);
  const auto group_step = static_cast<Index>(gridDim.x);
  while (group_index < parameters.rowwise_.group_count_) {
    const auto local = ReduceNormalizationThread<T>(parameters, group_index, static_cast<Index>(threadIdx.x),
                                                    static_cast<Index>(NORMALIZATION_THREADS_PER_BLOCK), operation);
    const auto aggregate = BlockReduce(block_storage).Reduce(local, operation);
    if (threadIdx.x == 0) {
      shared_statistics = aggregate;
    }
    __syncthreads();
    StoreNormalizationValues<T>(parameters, group_index, static_cast<Index>(threadIdx.x),
                                static_cast<Index>(NORMALIZATION_THREADS_PER_BLOCK), shared_statistics, operation);
    __syncthreads();
    if (group_step >= parameters.rowwise_.group_count_ - group_index) {
      break;
    }
    group_index = static_cast<Index>(group_index + group_step);
  }
}

template <CudaStorageType T, typename Operation, typename Parameters>
__global__ void PartialNormalizationKernel(Parameters parameters, typename Operation::Accumulator *partials,
                                           Operation operation) {
  using Accumulator = typename Operation::Accumulator;
  using BlockReduce = cub::BlockReduce<Accumulator, NORMALIZATION_THREADS_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage block_storage;

  using Index = RowwiseIndexType<decltype(parameters.rowwise_)>;
  const auto task_count = static_cast<uint64_t>(parameters.rowwise_.group_count_) * parameters.rowwise_.partial_count_;
  auto task = static_cast<uint64_t>(blockIdx.x);
  const auto task_step = static_cast<uint64_t>(gridDim.x);
  while (task < task_count) {
    const auto group_index = static_cast<Index>(task / parameters.rowwise_.partial_count_);
    const auto partial_index = static_cast<Index>(task % parameters.rowwise_.partial_count_);
    const auto first = static_cast<Index>(partial_index * NORMALIZATION_THREADS_PER_BLOCK + threadIdx.x);
    const auto step = static_cast<Index>(parameters.rowwise_.partial_count_ * NORMALIZATION_THREADS_PER_BLOCK);
    const auto local = ReduceNormalizationThread<T>(parameters, group_index, first, step, operation);
    const auto aggregate = BlockReduce(block_storage).Reduce(local, operation);
    if (threadIdx.x == 0) {
      partials[task] = aggregate;
    }
    __syncthreads();
    task += task_step;
  }
}

template <CudaStorageType T, typename Operation, typename Parameters>
__global__ void FinalNormalizationKernel(Parameters parameters, const typename Operation::Accumulator *partials,
                                         Operation operation) {
  using Accumulator = typename Operation::Accumulator;
  using BlockReduce = cub::BlockReduce<Accumulator, NORMALIZATION_THREADS_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage block_storage;
  __shared__ Accumulator shared_statistics;

  using Index = RowwiseIndexType<decltype(parameters.rowwise_)>;
  auto group_index = static_cast<Index>(blockIdx.x);
  const auto group_step = static_cast<Index>(gridDim.x);
  while (group_index < parameters.rowwise_.group_count_) {
    auto local = operation.Identity();
    if (threadIdx.x < parameters.rowwise_.partial_count_) {
      const auto offset = static_cast<uint64_t>(group_index) * parameters.rowwise_.partial_count_ + threadIdx.x;
      local = partials[offset];
    }
    const auto aggregate = BlockReduce(block_storage).Reduce(local, operation);
    if (threadIdx.x == 0) {
      shared_statistics = aggregate;
    }
    __syncthreads();
    StoreNormalizationValues<T>(parameters, group_index, static_cast<Index>(threadIdx.x),
                                static_cast<Index>(NORMALIZATION_THREADS_PER_BLOCK), shared_statistics, operation);
    __syncthreads();
    if (group_step >= parameters.rowwise_.group_count_ - group_index) {
      break;
    }
    group_index = static_cast<Index>(group_index + group_step);
  }
}

template <typename Index>
auto ConvertAuxiliary(const NormalizationAuxiliaryParameters64 &source, std::source_location location)
    -> NormalizationAuxiliaryParameters<Index> {
  NormalizationAuxiliaryParameters<Index> destination{
      .weight_ = source.weight_,
      .bias_ = source.bias_,
      .epsilon_ = source.epsilon_,
  };
  for (size_t axis = 0; axis < TTL_MAX_RANK; ++axis) {
    destination.weight_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.weight_strides_bytes_[axis], "normalization weight stride", location);
    destination.bias_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.bias_strides_bytes_[axis], "normalization bias stride", location);
  }
  return destination;
}

template <CudaStorageType T, typename Operation, typename Index>
void LaunchWithParameters(cudaStream_t stream, const RowwisePlan &plan, RowwiseParameters<Index> rowwise,
                          const NormalizationAuxiliaryParameters64 &auxiliary, void *scratch, Operation operation,
                          std::source_location location) {
  const auto parameters =
      NormalizationParameters<Index>{.rowwise_ = rowwise, .auxiliary_ = ConvertAuxiliary<Index>(auxiliary, location)};
  const auto blocks = plan.GetLaunchBlockCount();
  switch (plan.GetPath()) {
    case RowwisePath::WARP:
      WarpNormalizationKernel<T><<<blocks, NORMALIZATION_THREADS_PER_BLOCK, 0, stream>>>(parameters, operation);
      return;
    case RowwisePath::BLOCK:
      BlockNormalizationKernel<T><<<blocks, NORMALIZATION_THREADS_PER_BLOCK, 0, stream>>>(parameters, operation);
      return;
    case RowwisePath::TWO_STAGE:
      if (scratch == nullptr || plan.GetScratchBytes() == 0 || plan.GetPartialCount() <= 1) {
        throw InternalError("two-stage normalization requires scratch storage", location);
      }
      PartialNormalizationKernel<T><<<blocks, NORMALIZATION_THREADS_PER_BLOCK, 0, stream>>>(
          parameters, static_cast<typename Operation::Accumulator *>(scratch), operation);
      FinalNormalizationKernel<T>
          <<<static_cast<uint32_t>(rowwise.group_count_ < blocks ? rowwise.group_count_ : blocks),
             NORMALIZATION_THREADS_PER_BLOCK, 0, stream>>>(
              parameters, static_cast<const typename Operation::Accumulator *>(scratch), operation);
      return;
  }
  throw InternalError("invalid normalization row-wise path", location);
}

template <CudaStorageType T, typename Operation>
void LaunchTyped(cudaStream_t stream, const RowwisePlan &plan, const NormalizationAuxiliaryParameters64 &auxiliary,
                 void *scratch, Operation operation, std::source_location location) {
  if (stream == nullptr || plan.GetGroupCount() == 0 || plan.GetReductionCount() == 0) {
    throw InternalError("invalid normalization launch plan", location);
  }
  if (plan.GetIndexWidth() == IndexWidth::UINT32) {
    LaunchWithParameters<T>(stream, plan, plan.MakeParameters32(location), auxiliary, scratch, operation, location);
  } else {
    LaunchWithParameters<T>(stream, plan, plan.MakeParameters64(), auxiliary, scratch, operation, location);
  }
}

}  // namespace

void LaunchNormalization(cudaStream_t stream, DType dtype, NormalizationOp operation, const RowwisePlan &plan,
                         const NormalizationAuxiliaryParameters64 &auxiliary, void *scratch,
                         std::source_location location) {
  DispatchCudaFloatingDType(
      dtype, operation == NormalizationOp::LAYER_NORM ? "LayerNormOut" : "RmsNormOut",
      [&]<CudaStorageType T>(std::type_identity<T>) {
        if (operation == NormalizationOp::LAYER_NORM) {
          LaunchTyped<T>(stream, plan, auxiliary, scratch, WelfordOperation{}, location);
        } else {
          LaunchTyped<T>(stream, plan, auxiliary, scratch, ScaledSquaresOperation{}, location);
        }
      },
      location);
}

}  // namespace ttl::internal
