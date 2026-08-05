#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/index_width.hpp"
#include "ttl/internal/ops/reduction.hpp"

namespace ttl::internal {

inline constexpr uint32_t REDUCTION_THREADS_PER_BLOCK = 256;
inline constexpr uint32_t REDUCTION_WARP_SIZE = 32;
inline constexpr uint32_t REDUCTION_WARPS_PER_BLOCK = REDUCTION_THREADS_PER_BLOCK / REDUCTION_WARP_SIZE;

template <typename Parameters>
using ReductionIndexType = std::remove_cvref_t<decltype(Parameters::output_count_)>;

template <typename Parameters>
__device__ auto GetReductionInputBaseOffset(const Parameters &parameters, ReductionIndexType<Parameters> output_index)
    -> ReductionIndexType<Parameters> {
  using Index = ReductionIndexType<Parameters>;
  auto offset = Index{0};
  for (size_t remaining = parameters.output_rank_; remaining > 0; --remaining) {
    const auto axis = remaining - 1;
    const auto coordinate = static_cast<Index>(output_index % parameters.output_shape_[axis]);
    output_index = static_cast<Index>(output_index / parameters.output_shape_[axis]);
    offset = static_cast<Index>(offset + (coordinate * parameters.output_input_strides_bytes_[axis]));
  }
  return offset;
}

template <typename Parameters>
__device__ auto GetReductionOutputOffset(const Parameters &parameters, ReductionIndexType<Parameters> output_index)
    -> ReductionIndexType<Parameters> {
  using Index = ReductionIndexType<Parameters>;
  auto offset = Index{0};
  for (size_t remaining = parameters.output_rank_; remaining > 0; --remaining) {
    const auto axis = remaining - 1;
    const auto coordinate = static_cast<Index>(output_index % parameters.output_shape_[axis]);
    output_index = static_cast<Index>(output_index / parameters.output_shape_[axis]);
    offset = static_cast<Index>(offset + (coordinate * parameters.output_strides_bytes_[axis]));
  }
  return offset;
}

template <typename Parameters>
__device__ auto GetReductionElementOffset(const Parameters &parameters, ReductionIndexType<Parameters> reduction_index)
    -> ReductionIndexType<Parameters> {
  using Index = ReductionIndexType<Parameters>;
  if (parameters.contiguous_reduction_) {
    for (size_t remaining = parameters.reduction_rank_; remaining > 0; --remaining) {
      const auto axis = remaining - 1;
      if (parameters.reduction_shape_[axis] > 1) {
        return static_cast<Index>(reduction_index * parameters.reduction_strides_bytes_[axis]);
      }
    }
    return Index{0};
  }

  auto offset = Index{0};
  for (size_t remaining = parameters.reduction_rank_; remaining > 0; --remaining) {
    const auto axis = remaining - 1;
    const auto coordinate = static_cast<Index>(reduction_index % parameters.reduction_shape_[axis]);
    reduction_index = static_cast<Index>(reduction_index / parameters.reduction_shape_[axis]);
    offset = static_cast<Index>(offset + (coordinate * parameters.reduction_strides_bytes_[axis]));
  }
  return offset;
}

template <typename Input, typename Operation, typename Parameters>
__device__ auto ReduceThreadValues(const Parameters &parameters, ReductionIndexType<Parameters> output_index,
                                   ReductionIndexType<Parameters> first_reduction_index,
                                   ReductionIndexType<Parameters> reduction_step, Operation operation) ->
    typename Operation::Accumulator {
  using Index = ReductionIndexType<Parameters>;
  auto accumulator = operation.Identity();
  const auto input_base_offset = GetReductionInputBaseOffset(parameters, output_index);
  for (auto reduction_index = first_reduction_index; reduction_index < parameters.reduction_count_;
       reduction_index = static_cast<Index>(reduction_index + reduction_step)) {
    const auto element_offset = GetReductionElementOffset(parameters, reduction_index);
    const auto value = *reinterpret_cast<const Input *>(parameters.input_ + input_base_offset + element_offset);
    accumulator = operation.Combine(accumulator, operation.Lift(value, static_cast<int64_t>(reduction_index)));
    if (reduction_step >= parameters.reduction_count_ - reduction_index) {
      break;
    }
  }
  return accumulator;
}

template <typename Output, typename Operation, typename Parameters>
__device__ void StoreReductionOutput(const Parameters &parameters, ReductionIndexType<Parameters> output_index,
                                     const typename Operation::Accumulator &accumulator, Operation operation) {
  const auto output_offset = GetReductionOutputOffset(parameters, output_index);
  *reinterpret_cast<Output *>(parameters.output_ + output_offset) =
      operation.Project(accumulator, static_cast<uint64_t>(parameters.reduction_count_));
}

template <typename Input, typename Output, typename Operation, typename Parameters>
__global__ void WarpReductionKernel(Parameters parameters, Operation operation) {
  // One warp owns each output and combines its lane-local strided subsequences. Multiple warps share a block only to
  // amortize launch overhead; their CUB temporary storage remains independent.
  using Accumulator = typename Operation::Accumulator;
  using WarpReduce = cub::WarpReduce<Accumulator>;
  __shared__ typename WarpReduce::TempStorage warp_storage[REDUCTION_WARPS_PER_BLOCK];

  using Index = ReductionIndexType<Parameters>;
  const auto lane = static_cast<uint32_t>(threadIdx.x) % REDUCTION_WARP_SIZE;
  const auto warp = static_cast<uint32_t>(threadIdx.x) / REDUCTION_WARP_SIZE;
  auto output_index = static_cast<Index>((static_cast<uint64_t>(blockIdx.x) * REDUCTION_WARPS_PER_BLOCK) + warp);
  const auto output_step = static_cast<Index>(static_cast<uint64_t>(gridDim.x) * REDUCTION_WARPS_PER_BLOCK);

  while (output_index < parameters.output_count_) {
    const auto accumulator = ReduceThreadValues<Input>(parameters, output_index, static_cast<Index>(lane),
                                                       static_cast<Index>(REDUCTION_WARP_SIZE), operation);
    const auto aggregate = WarpReduce(warp_storage[warp]).Reduce(accumulator, operation);
    if (lane == 0) {
      StoreReductionOutput<Output>(parameters, output_index, aggregate, operation);
    }
    __syncwarp();
    if (output_step >= parameters.output_count_ - output_index) {
      break;
    }
    output_index = static_cast<Index>(output_index + output_step);
  }
}

template <typename Input, typename Output, typename Operation, typename Parameters>
__global__ void BlockReductionKernel(Parameters parameters, Operation operation) {
  // One block owns each output, allowing a larger reduction domain to be combined in shared memory without scratch.
  using Accumulator = typename Operation::Accumulator;
  using BlockReduce = cub::BlockReduce<Accumulator, REDUCTION_THREADS_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage block_storage;

  using Index = ReductionIndexType<Parameters>;
  auto output_index = static_cast<Index>(blockIdx.x);
  const auto output_step = static_cast<Index>(gridDim.x);
  while (output_index < parameters.output_count_) {
    const auto accumulator = ReduceThreadValues<Input>(parameters, output_index, static_cast<Index>(threadIdx.x),
                                                       static_cast<Index>(REDUCTION_THREADS_PER_BLOCK), operation);
    const auto aggregate = BlockReduce(block_storage).Reduce(accumulator, operation);
    if (threadIdx.x == 0) {
      StoreReductionOutput<Output>(parameters, output_index, aggregate, operation);
    }
    __syncthreads();
    if (output_step >= parameters.output_count_ - output_index) {
      break;
    }
    output_index = static_cast<Index>(output_index + output_step);
  }
}

template <typename Input, typename Operation, typename Parameters>
__global__ void PartialReductionKernel(Parameters parameters, typename Operation::Accumulator *partials,
                                       Operation operation) {
  // Each block owns one (output, partial) pair. Partial p consumes every partial_count-th 256-element tile so work is
  // balanced even when the reduction extent is not divisible by the number of cooperating blocks.
  using Accumulator = typename Operation::Accumulator;
  using BlockReduce = cub::BlockReduce<Accumulator, REDUCTION_THREADS_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage block_storage;

  using Index = ReductionIndexType<Parameters>;
  const auto task_count = static_cast<uint64_t>(parameters.output_count_) * parameters.partial_count_;
  auto task_index = static_cast<uint64_t>(blockIdx.x);
  const auto task_step = static_cast<uint64_t>(gridDim.x);
  while (task_index < task_count) {
    const auto output_index = static_cast<Index>(task_index / parameters.partial_count_);
    const auto partial_index = static_cast<Index>(task_index % parameters.partial_count_);
    const auto first_reduction_index =
        static_cast<Index>((partial_index * REDUCTION_THREADS_PER_BLOCK) + static_cast<Index>(threadIdx.x));
    const auto reduction_step =
        static_cast<Index>(static_cast<Index>(parameters.partial_count_) * REDUCTION_THREADS_PER_BLOCK);
    const auto accumulator =
        ReduceThreadValues<Input>(parameters, output_index, first_reduction_index, reduction_step, operation);
    const auto aggregate = BlockReduce(block_storage).Reduce(accumulator, operation);
    if (threadIdx.x == 0) {
      partials[task_index] = aggregate;
    }
    __syncthreads();
    task_index += task_step;
  }
}

template <typename Output, typename Operation, typename Parameters>
__global__ void FinalReductionKernel(Parameters parameters, const typename Operation::Accumulator *partials,
                                     Operation operation) {
  // One block folds every scratch partial for an output and performs the operation-specific final projection once.
  using Accumulator = typename Operation::Accumulator;
  using BlockReduce = cub::BlockReduce<Accumulator, REDUCTION_THREADS_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage block_storage;

  using Index = ReductionIndexType<Parameters>;
  auto output_index = static_cast<Index>(blockIdx.x);
  const auto output_step = static_cast<Index>(gridDim.x);
  while (output_index < parameters.output_count_) {
    auto accumulator = operation.Identity();
    for (auto partial_index = static_cast<uint32_t>(threadIdx.x); partial_index < parameters.partial_count_;
         partial_index += REDUCTION_THREADS_PER_BLOCK) {
      const auto partial_offset = (static_cast<uint64_t>(output_index) * parameters.partial_count_) + partial_index;
      accumulator = operation.Combine(accumulator, partials[partial_offset]);
    }
    const auto aggregate = BlockReduce(block_storage).Reduce(accumulator, operation);
    if (threadIdx.x == 0) {
      StoreReductionOutput<Output>(parameters, output_index, aggregate, operation);
    }
    __syncthreads();
    if (output_step >= parameters.output_count_ - output_index) {
      break;
    }
    output_index = static_cast<Index>(output_index + output_step);
  }
}

template <typename Input, typename Output, typename Operation, typename Parameters>
void LaunchReductionWithParameters(cudaStream_t stream, const ReductionPlan &plan, const Parameters &parameters,
                                   void *scratch, Operation operation, std::source_location location) {
  const auto block_count = plan.GetLaunchBlockCount();
  switch (plan.GetPath()) {
    case ReductionPath::WARP:
      WarpReductionKernel<Input, Output>
          <<<block_count, REDUCTION_THREADS_PER_BLOCK, 0, stream>>>(parameters, operation);
      return;
    case ReductionPath::BLOCK:
      BlockReductionKernel<Input, Output>
          <<<block_count, REDUCTION_THREADS_PER_BLOCK, 0, stream>>>(parameters, operation);
      return;
    case ReductionPath::TWO_STAGE: {
      if (scratch == nullptr || plan.GetScratchBytes() == 0 || plan.GetPartialCount() <= 1) {
        throw InternalError("two-stage reduction requires non-empty scratch storage", location);
      }
      auto *partials = static_cast<typename Operation::Accumulator *>(scratch);
      PartialReductionKernel<Input>
          <<<block_count, REDUCTION_THREADS_PER_BLOCK, 0, stream>>>(parameters, partials, operation);
      const auto final_block_count =
          static_cast<uint32_t>(parameters.output_count_ < block_count ? parameters.output_count_ : block_count);
      FinalReductionKernel<Output>
          <<<final_block_count, REDUCTION_THREADS_PER_BLOCK, 0, stream>>>(parameters, partials, operation);
      return;
    }
  }
  throw InternalError("invalid reduction path", location);
}

template <typename Input, typename Output, typename Operation>
void LaunchTypedReduction(cudaStream_t stream, const ReductionPlan &plan, void *scratch, Operation operation,
                          std::source_location location) {
  if (stream == nullptr || plan.GetOutputCount() == 0) {
    throw InternalError("invalid reduction launch plan", location);
  }
  if (plan.GetIndexWidth() == IndexWidth::UINT32) {
    LaunchReductionWithParameters<Input, Output>(stream, plan, plan.MakeParameters32(location), scratch, operation,
                                                 location);
    return;
  }
  LaunchReductionWithParameters<Input, Output>(stream, plan, plan.MakeParameters64(), scratch, operation, location);
}

}  // namespace ttl::internal
