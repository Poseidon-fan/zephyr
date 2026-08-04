#include "ttl/internal/ops/softmax.hpp"

#include <concepts>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/index_width.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_math.cuh"
#include "ttl/internal/kernels/reduction/rowwise_kernel.cuh"
#include "ttl/internal/ops/rowwise.hpp"
#include "ttl/internal/runtime/library/cuda_dtype.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t SOFTMAX_THREADS_PER_BLOCK = 256;
constexpr uint32_t SOFTMAX_WARP_SIZE = 32;
constexpr uint32_t SOFTMAX_WARPS_PER_BLOCK = SOFTMAX_THREADS_PER_BLOCK / SOFTMAX_WARP_SIZE;

struct SoftmaxStatistics final {
  float maximum_;
  float sum_;
};

static_assert(sizeof(SoftmaxStatistics) == 8);

struct SoftmaxStatisticsCombine final {
  __device__ auto operator()(SoftmaxStatistics lhs, SoftmaxStatistics rhs) const -> SoftmaxStatistics {
    if (lhs.sum_ == 0.0F) {
      return rhs;
    }
    if (rhs.sum_ == 0.0F) {
      return lhs;
    }
    if (isnan(lhs.maximum_) || isnan(rhs.maximum_)) {
      return {.maximum_ = CUDART_NAN_F, .sum_ = CUDART_NAN_F};
    }
    // Each partial sum is relative to its own maximum. Rescale both before merging so the operation remains stable
    // and associative enough for warp, block, and two-stage CUB reductions.
    const auto maximum = fmaxf(lhs.maximum_, rhs.maximum_);
    return {
        .maximum_ = maximum,
        .sum_ = lhs.sum_ * expf(lhs.maximum_ - maximum) + rhs.sum_ * expf(rhs.maximum_ - maximum),
    };
  }
};

[[nodiscard]] __device__ auto IdentityStatistics() -> SoftmaxStatistics {
  return {.maximum_ = -CUDART_INF_F, .sum_ = 0.0F};
}

template <CudaStorageType T, typename Parameters>
__device__ auto ReduceSoftmaxThread(const Parameters &parameters, RowwiseIndexType<Parameters> group_index,
                                    RowwiseIndexType<Parameters> first_index, RowwiseIndexType<Parameters> step)
    -> SoftmaxStatistics {
  using Index = RowwiseIndexType<Parameters>;
  const auto group_offset = GetRowwiseGroupInputOffset(parameters, group_index);
  auto statistics = IdentityStatistics();
  const auto combine = SoftmaxStatisticsCombine{};
  for (auto index = first_index; index < parameters.reduction_count_; index = static_cast<Index>(index + step)) {
    const auto offset = GetRowwiseReductionOffset(parameters, index, false);
    const auto value = ToElementwiseFloat(*reinterpret_cast<const T *>(parameters.input_ + group_offset + offset));
    statistics = combine(statistics, {.maximum_ = value, .sum_ = 1.0F});
    if (step >= parameters.reduction_count_ - index) {
      break;
    }
  }
  return statistics;
}

template <CudaStorageType T, SoftmaxOp operation, typename Parameters>
__device__ void StoreSoftmaxValues(const Parameters &parameters, RowwiseIndexType<Parameters> group_index,
                                   RowwiseIndexType<Parameters> first_index, RowwiseIndexType<Parameters> step,
                                   SoftmaxStatistics statistics) {
  using Index = RowwiseIndexType<Parameters>;
  const auto input_group_offset = GetRowwiseGroupInputOffset(parameters, group_index);
  const auto output_group_offset = GetRowwiseGroupOutputOffset(parameters, group_index);
  for (auto index = first_index; index < parameters.reduction_count_; index = static_cast<Index>(index + step)) {
    const auto input_offset = GetRowwiseReductionOffset(parameters, index, false);
    const auto output_offset = GetRowwiseReductionOffset(parameters, index, true);
    const auto value =
        ToElementwiseFloat(*reinterpret_cast<const T *>(parameters.input_ + input_group_offset + input_offset));
    float result;
    if constexpr (operation == SoftmaxOp::SOFTMAX) {
      result = expf(value - statistics.maximum_) / statistics.sum_;
    } else {
      result = value - statistics.maximum_ - logf(statistics.sum_);
    }
    *reinterpret_cast<T *>(parameters.output_ + output_group_offset + output_offset) = FromElementwiseFloat<T>(result);
    if (step >= parameters.reduction_count_ - index) {
      break;
    }
  }
}

template <CudaStorageType T, SoftmaxOp operation, typename Parameters>
__global__ void WarpSoftmaxKernel(Parameters parameters) {
  using WarpReduce = cub::WarpReduce<SoftmaxStatistics>;
  __shared__ typename WarpReduce::TempStorage warp_storage[SOFTMAX_WARPS_PER_BLOCK];
  __shared__ SoftmaxStatistics shared_statistics[SOFTMAX_WARPS_PER_BLOCK];

  using Index = RowwiseIndexType<Parameters>;
  const auto lane = static_cast<uint32_t>(threadIdx.x) % SOFTMAX_WARP_SIZE;
  const auto warp = static_cast<uint32_t>(threadIdx.x) / SOFTMAX_WARP_SIZE;
  auto group_index = static_cast<Index>((static_cast<uint64_t>(blockIdx.x) * SOFTMAX_WARPS_PER_BLOCK) + warp);
  const auto group_step = static_cast<Index>(static_cast<uint64_t>(gridDim.x) * SOFTMAX_WARPS_PER_BLOCK);
  while (group_index < parameters.group_count_) {
    const auto local = ReduceSoftmaxThread<T>(parameters, group_index, static_cast<Index>(lane),
                                              static_cast<Index>(SOFTMAX_WARP_SIZE));
    const auto aggregate = WarpReduce(warp_storage[warp]).Reduce(local, SoftmaxStatisticsCombine{});
    if (lane == 0) {
      shared_statistics[warp] = aggregate;
    }
    __syncwarp();
    StoreSoftmaxValues<T, operation>(parameters, group_index, static_cast<Index>(lane),
                                     static_cast<Index>(SOFTMAX_WARP_SIZE), shared_statistics[warp]);
    __syncwarp();
    if (group_step >= parameters.group_count_ - group_index) {
      break;
    }
    group_index = static_cast<Index>(group_index + group_step);
  }
}

template <CudaStorageType T, SoftmaxOp operation, typename Parameters>
__global__ void BlockSoftmaxKernel(Parameters parameters) {
  using BlockReduce = cub::BlockReduce<SoftmaxStatistics, SOFTMAX_THREADS_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage block_storage;
  __shared__ SoftmaxStatistics shared_statistics;

  using Index = RowwiseIndexType<Parameters>;
  auto group_index = static_cast<Index>(blockIdx.x);
  const auto group_step = static_cast<Index>(gridDim.x);
  while (group_index < parameters.group_count_) {
    const auto local = ReduceSoftmaxThread<T>(parameters, group_index, static_cast<Index>(threadIdx.x),
                                              static_cast<Index>(SOFTMAX_THREADS_PER_BLOCK));
    const auto aggregate = BlockReduce(block_storage).Reduce(local, SoftmaxStatisticsCombine{});
    if (threadIdx.x == 0) {
      shared_statistics = aggregate;
    }
    __syncthreads();
    StoreSoftmaxValues<T, operation>(parameters, group_index, static_cast<Index>(threadIdx.x),
                                     static_cast<Index>(SOFTMAX_THREADS_PER_BLOCK), shared_statistics);
    __syncthreads();
    if (group_step >= parameters.group_count_ - group_index) {
      break;
    }
    group_index = static_cast<Index>(group_index + group_step);
  }
}

template <CudaStorageType T, typename Parameters>
__global__ void PartialSoftmaxKernel(Parameters parameters, SoftmaxStatistics *partials) {
  using BlockReduce = cub::BlockReduce<SoftmaxStatistics, SOFTMAX_THREADS_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage block_storage;

  using Index = RowwiseIndexType<Parameters>;
  const auto task_count = static_cast<uint64_t>(parameters.group_count_) * parameters.partial_count_;
  auto task = static_cast<uint64_t>(blockIdx.x);
  const auto task_step = static_cast<uint64_t>(gridDim.x);
  while (task < task_count) {
    const auto group_index = static_cast<Index>(task / parameters.partial_count_);
    const auto partial_index = static_cast<Index>(task % parameters.partial_count_);
    const auto first = static_cast<Index>(partial_index * SOFTMAX_THREADS_PER_BLOCK + threadIdx.x);
    const auto step = static_cast<Index>(parameters.partial_count_ * SOFTMAX_THREADS_PER_BLOCK);
    const auto local = ReduceSoftmaxThread<T>(parameters, group_index, first, step);
    const auto aggregate = BlockReduce(block_storage).Reduce(local, SoftmaxStatisticsCombine{});
    if (threadIdx.x == 0) {
      partials[task] = aggregate;
    }
    __syncthreads();
    task += task_step;
  }
}

template <CudaStorageType T, SoftmaxOp operation, typename Parameters>
__global__ void FinalSoftmaxKernel(Parameters parameters, const SoftmaxStatistics *partials) {
  using BlockReduce = cub::BlockReduce<SoftmaxStatistics, SOFTMAX_THREADS_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage block_storage;
  __shared__ SoftmaxStatistics shared_statistics;

  using Index = RowwiseIndexType<Parameters>;
  auto group_index = static_cast<Index>(blockIdx.x);
  const auto group_step = static_cast<Index>(gridDim.x);
  while (group_index < parameters.group_count_) {
    auto local = IdentityStatistics();
    if (threadIdx.x < parameters.partial_count_) {
      const auto offset = static_cast<uint64_t>(group_index) * parameters.partial_count_ + threadIdx.x;
      local = partials[offset];
    }
    const auto aggregate = BlockReduce(block_storage).Reduce(local, SoftmaxStatisticsCombine{});
    if (threadIdx.x == 0) {
      shared_statistics = aggregate;
    }
    __syncthreads();
    StoreSoftmaxValues<T, operation>(parameters, group_index, static_cast<Index>(threadIdx.x),
                                     static_cast<Index>(SOFTMAX_THREADS_PER_BLOCK), shared_statistics);
    __syncthreads();
    if (group_step >= parameters.group_count_ - group_index) {
      break;
    }
    group_index = static_cast<Index>(group_index + group_step);
  }
}

template <CudaStorageType T, SoftmaxOp operation, typename Parameters>
void LaunchWithParameters(cudaStream_t stream, const RowwisePlan &plan, const Parameters &parameters, void *scratch,
                          std::source_location location) {
  const auto blocks = plan.GetLaunchBlockCount();
  switch (plan.GetPath()) {
    case RowwisePath::WARP:
      WarpSoftmaxKernel<T, operation><<<blocks, SOFTMAX_THREADS_PER_BLOCK, 0, stream>>>(parameters);
      return;
    case RowwisePath::BLOCK:
      BlockSoftmaxKernel<T, operation><<<blocks, SOFTMAX_THREADS_PER_BLOCK, 0, stream>>>(parameters);
      return;
    case RowwisePath::TWO_STAGE:
      if (scratch == nullptr || plan.GetScratchBytes() == 0 || plan.GetPartialCount() <= 1) {
        throw InternalError("two-stage softmax requires scratch storage", location);
      }
      PartialSoftmaxKernel<T>
          <<<blocks, SOFTMAX_THREADS_PER_BLOCK, 0, stream>>>(parameters, static_cast<SoftmaxStatistics *>(scratch));
      FinalSoftmaxKernel<T, operation>
          <<<static_cast<uint32_t>(parameters.group_count_ < blocks ? parameters.group_count_ : blocks),
             SOFTMAX_THREADS_PER_BLOCK, 0, stream>>>(parameters, static_cast<const SoftmaxStatistics *>(scratch));
      return;
  }
  throw InternalError("invalid softmax row-wise path", location);
}

template <CudaStorageType T, SoftmaxOp operation>
void LaunchTyped(cudaStream_t stream, const RowwisePlan &plan, void *scratch, std::source_location location) {
  if (stream == nullptr || plan.GetGroupCount() == 0 || plan.GetReductionCount() == 0) {
    throw InternalError("invalid softmax launch plan", location);
  }
  if (plan.GetIndexWidth() == IndexWidth::UINT32) {
    LaunchWithParameters<T, operation>(stream, plan, plan.MakeParameters32(location), scratch, location);
  } else {
    LaunchWithParameters<T, operation>(stream, plan, plan.MakeParameters64(), scratch, location);
  }
}

}  // namespace

void LaunchSoftmax(cudaStream_t stream, DType dtype, SoftmaxOp operation, const RowwisePlan &plan, void *scratch,
                   std::source_location location) {
  DispatchCudaFloatingDType(
      dtype, operation == SoftmaxOp::SOFTMAX ? "SoftmaxOut" : "LogSoftmaxOut",
      [&]<CudaStorageType T>(std::type_identity<T>) {
        if (operation == SoftmaxOp::SOFTMAX) {
          LaunchTyped<T, SoftmaxOp::SOFTMAX>(stream, plan, scratch, location);
        } else {
          LaunchTyped<T, SoftmaxOp::LOG_SOFTMAX>(stream, plan, scratch, location);
        }
      },
      location);
}

}  // namespace ttl::internal
