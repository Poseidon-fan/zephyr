#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>

#include <cuda_runtime.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_kernel.cuh"
#include "ttl/internal/ops/elementwise_iterator.hpp"

namespace ttl::internal {

/** Result wrapper that lets an operation report a device error without overwriting the corresponding output element. */
template <typename T>
struct ElementwiseResult final {
  T value_;
  bool write_{true};
};

template <typename T, size_t width>
struct alignas(sizeof(T) * width) ElementwiseVector final {
  T values_[width];
};

template <typename Output, typename Input, typename Operation>
struct UnaryIteratorOperation final {
  Operation operation_;

  template <typename Parameters>
  __device__ void operator()(const Parameters &parameters, ElementwiseIndexType<Parameters> index) const {
    const auto value = *reinterpret_cast<const Input *>(GetElementwisePointer(parameters, 1, index));
    const auto result = operation_(value, static_cast<int64_t>(index));
    if (result.write_) {
      *reinterpret_cast<Output *>(GetElementwisePointer(parameters, 0, index)) = result.value_;
    }
  }
};

template <typename Output, typename Lhs, typename Rhs, typename Operation>
struct BinaryIteratorOperation final {
  Operation operation_;

  template <typename Parameters>
  __device__ void operator()(const Parameters &parameters, ElementwiseIndexType<Parameters> index) const {
    const auto lhs = *reinterpret_cast<const Lhs *>(GetElementwisePointer(parameters, 1, index));
    const auto rhs = *reinterpret_cast<const Rhs *>(GetElementwisePointer(parameters, 2, index));
    const auto result = operation_(lhs, rhs, static_cast<int64_t>(index));
    if (result.write_) {
      *reinterpret_cast<Output *>(GetElementwisePointer(parameters, 0, index)) = result.value_;
    }
  }
};

template <typename Output, typename First, typename Second, typename Third, typename Operation>
struct TernaryIteratorOperation final {
  Operation operation_;

  template <typename Parameters>
  __device__ void operator()(const Parameters &parameters, ElementwiseIndexType<Parameters> index) const {
    const auto first = *reinterpret_cast<const First *>(GetElementwisePointer(parameters, 1, index));
    const auto second = *reinterpret_cast<const Second *>(GetElementwisePointer(parameters, 2, index));
    const auto third = *reinterpret_cast<const Third *>(GetElementwisePointer(parameters, 3, index));
    const auto result = operation_(first, second, third, static_cast<int64_t>(index));
    if (result.write_) {
      *reinterpret_cast<Output *>(GetElementwisePointer(parameters, 0, index)) = result.value_;
    }
  }
};

template <typename Output, typename Input, size_t width, typename Index, typename Operation>
__global__ void ContiguousUnaryKernel(Output *output, const Input *input, Index num_elements, Operation operation) {
  using OutputVector = ElementwiseVector<Output, width>;
  using InputVector = ElementwiseVector<Input, width>;
  const auto vector_count = static_cast<Index>(num_elements / width);
  auto vector_index =
      (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto grid_stride = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  const auto *vector_input = reinterpret_cast<const InputVector *>(input);
  auto *vector_output = reinterpret_cast<OutputVector *>(output);

  while (vector_index < vector_count) {
    const auto input_values = vector_input[vector_index];
    OutputVector output_values;
    bool write_all = true;
#pragma unroll
    for (size_t element = 0; element < width; ++element) {
      const auto linear_index = static_cast<Index>((vector_index * static_cast<Index>(width)) + element);
      const auto result = operation(input_values.values_[element], static_cast<int64_t>(linear_index));
      output_values.values_[element] = result.value_;
      write_all = write_all && result.write_;
    }
    if (write_all) {
      vector_output[vector_index] = output_values;
    } else {
      // A vector store cannot preserve individual elements whose operations rejected their result. Re-evaluate the
      // vector and issue only permitted scalar stores; error reporting is sticky and remains idempotent.
#pragma unroll
      for (size_t element = 0; element < width; ++element) {
        const auto linear_index = static_cast<Index>((vector_index * static_cast<Index>(width)) + element);
        const auto result = operation(input_values.values_[element], static_cast<int64_t>(linear_index));
        if (result.write_) {
          output[linear_index] = result.value_;
        }
      }
    }
    vector_index = static_cast<Index>(vector_index + grid_stride);
  }

  auto tail_index = static_cast<Index>((vector_count * static_cast<Index>(width)) +
                                       (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) +
                                       static_cast<Index>(threadIdx.x));
  while (tail_index < num_elements) {
    const auto result = operation(input[tail_index], static_cast<int64_t>(tail_index));
    if (result.write_) {
      output[tail_index] = result.value_;
    }
    tail_index = static_cast<Index>(tail_index + grid_stride);
  }
}

template <typename Output, typename Lhs, typename Rhs, size_t width, typename Index, typename Operation>
__global__ void ContiguousBinaryKernel(Output *output, const Lhs *lhs, const Rhs *rhs, Index num_elements,
                                       Operation operation) {
  using OutputVector = ElementwiseVector<Output, width>;
  using LhsVector = ElementwiseVector<Lhs, width>;
  using RhsVector = ElementwiseVector<Rhs, width>;
  const auto vector_count = static_cast<Index>(num_elements / width);
  auto vector_index =
      (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto grid_stride = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  const auto *vector_lhs = reinterpret_cast<const LhsVector *>(lhs);
  const auto *vector_rhs = reinterpret_cast<const RhsVector *>(rhs);
  auto *vector_output = reinterpret_cast<OutputVector *>(output);

  while (vector_index < vector_count) {
    const auto lhs_values = vector_lhs[vector_index];
    const auto rhs_values = vector_rhs[vector_index];
    OutputVector output_values;
    bool write_all = true;
#pragma unroll
    for (size_t element = 0; element < width; ++element) {
      const auto linear_index = static_cast<Index>((vector_index * static_cast<Index>(width)) + element);
      const auto result =
          operation(lhs_values.values_[element], rhs_values.values_[element], static_cast<int64_t>(linear_index));
      output_values.values_[element] = result.value_;
      write_all = write_all && result.write_;
    }
    if (write_all) {
      vector_output[vector_index] = output_values;
    } else {
#pragma unroll
      for (size_t element = 0; element < width; ++element) {
        const auto linear_index = static_cast<Index>((vector_index * static_cast<Index>(width)) + element);
        const auto result =
            operation(lhs_values.values_[element], rhs_values.values_[element], static_cast<int64_t>(linear_index));
        if (result.write_) {
          output[linear_index] = result.value_;
        }
      }
    }
    vector_index = static_cast<Index>(vector_index + grid_stride);
  }

  auto tail_index = static_cast<Index>((vector_count * static_cast<Index>(width)) +
                                       (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) +
                                       static_cast<Index>(threadIdx.x));
  while (tail_index < num_elements) {
    const auto result = operation(lhs[tail_index], rhs[tail_index], static_cast<int64_t>(tail_index));
    if (result.write_) {
      output[tail_index] = result.value_;
    }
    tail_index = static_cast<Index>(tail_index + grid_stride);
  }
}

template <typename Output, typename First, typename Second, typename Third, size_t width, typename Index,
          typename Operation>
__global__ void ContiguousTernaryKernel(Output *output, const First *first, const Second *second, const Third *third,
                                        Index num_elements, Operation operation) {
  using OutputVector = ElementwiseVector<Output, width>;
  using FirstVector = ElementwiseVector<First, width>;
  using SecondVector = ElementwiseVector<Second, width>;
  using ThirdVector = ElementwiseVector<Third, width>;
  const auto vector_count = static_cast<Index>(num_elements / width);
  auto vector_index =
      (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto grid_stride = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  const auto *vector_first = reinterpret_cast<const FirstVector *>(first);
  const auto *vector_second = reinterpret_cast<const SecondVector *>(second);
  const auto *vector_third = reinterpret_cast<const ThirdVector *>(third);
  auto *vector_output = reinterpret_cast<OutputVector *>(output);

  while (vector_index < vector_count) {
    const auto first_values = vector_first[vector_index];
    const auto second_values = vector_second[vector_index];
    const auto third_values = vector_third[vector_index];
    OutputVector output_values;
#pragma unroll
    for (size_t element = 0; element < width; ++element) {
      const auto linear_index = static_cast<Index>((vector_index * static_cast<Index>(width)) + element);
      output_values.values_[element] = operation(first_values.values_[element], second_values.values_[element],
                                                 third_values.values_[element], static_cast<int64_t>(linear_index))
                                           .value_;
    }
    vector_output[vector_index] = output_values;
    vector_index = static_cast<Index>(vector_index + grid_stride);
  }

  auto tail_index = static_cast<Index>((vector_count * static_cast<Index>(width)) +
                                       (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) +
                                       static_cast<Index>(threadIdx.x));
  while (tail_index < num_elements) {
    output[tail_index] =
        operation(first[tail_index], second[tail_index], third[tail_index], static_cast<int64_t>(tail_index)).value_;
    tail_index = static_cast<Index>(tail_index + grid_stride);
  }
}

template <size_t width, typename... Types>
// NOLINTNEXTLINE(bugprone-dynamic-static-initializers): constexpr variable templates are constant-initialized.
inline constexpr bool VALID_ELEMENTWISE_VECTOR_WIDTH = ((sizeof(Types) * width <= 16) && ...);

template <typename Output, typename Input, size_t width, typename Parameters, typename Operation>
void LaunchContiguousUnary(cudaStream_t stream, const Parameters &parameters, Operation operation) {
  const auto work_items = static_cast<uint64_t>(parameters.num_elements_ / width);
  const auto block_count = GetElementwiseBlockCount(work_items == 0 ? 1 : work_items);
  ContiguousUnaryKernel<Output, Input, width><<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<Output *>(parameters.pointers_[0]), reinterpret_cast<const Input *>(parameters.pointers_[1]),
      parameters.num_elements_, operation);
}

template <typename Output, typename Lhs, typename Rhs, size_t width, typename Parameters, typename Operation>
void LaunchContiguousBinary(cudaStream_t stream, const Parameters &parameters, Operation operation) {
  const auto work_items = static_cast<uint64_t>(parameters.num_elements_ / width);
  const auto block_count = GetElementwiseBlockCount(work_items == 0 ? 1 : work_items);
  ContiguousBinaryKernel<Output, Lhs, Rhs, width><<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<Output *>(parameters.pointers_[0]), reinterpret_cast<const Lhs *>(parameters.pointers_[1]),
      reinterpret_cast<const Rhs *>(parameters.pointers_[2]), parameters.num_elements_, operation);
}

template <typename Output, typename First, typename Second, typename Third, size_t width, typename Parameters,
          typename Operation>
void LaunchContiguousTernary(cudaStream_t stream, const Parameters &parameters, Operation operation) {
  const auto work_items = static_cast<uint64_t>(parameters.num_elements_ / width);
  const auto block_count = GetElementwiseBlockCount(work_items == 0 ? 1 : work_items);
  ContiguousTernaryKernel<Output, First, Second, Third, width>
      <<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(
          reinterpret_cast<Output *>(parameters.pointers_[0]), reinterpret_cast<const First *>(parameters.pointers_[1]),
          reinterpret_cast<const Second *>(parameters.pointers_[2]),
          reinterpret_cast<const Third *>(parameters.pointers_[3]), parameters.num_elements_, operation);
}

template <typename Output, typename Input, typename Parameters, typename Operation>
void LaunchUnaryWithParameters(cudaStream_t stream, const Parameters &parameters, IteratorPath path, uint8_t width,
                               Operation operation, std::source_location location) {
  if (path != IteratorPath::CONTIGUOUS) {
    const auto block_count = GetElementwiseBlockCount(parameters.num_elements_);
    ElementwiseKernel<<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(
        parameters, UnaryIteratorOperation<Output, Input, Operation>{operation});
    return;
  }

#define TTL_LAUNCH_UNARY_WIDTH(width_value)                                             \
  case width_value:                                                                     \
    if constexpr (VALID_ELEMENTWISE_VECTOR_WIDTH<width_value, Output, Input>) {         \
      LaunchContiguousUnary<Output, Input, width_value>(stream, parameters, operation); \
      return;                                                                           \
    }                                                                                   \
    break
  switch (width) {
    TTL_LAUNCH_UNARY_WIDTH(1);
    TTL_LAUNCH_UNARY_WIDTH(2);
    TTL_LAUNCH_UNARY_WIDTH(4);
    TTL_LAUNCH_UNARY_WIDTH(8);
    TTL_LAUNCH_UNARY_WIDTH(16);
    default:
      break;
  }
#undef TTL_LAUNCH_UNARY_WIDTH
  throw InternalError("elementwise iterator selected an invalid unary vector width", location);
}

template <typename Output, typename Lhs, typename Rhs, typename Parameters, typename Operation>
void LaunchBinaryWithParameters(cudaStream_t stream, const Parameters &parameters, IteratorPath path, uint8_t width,
                                Operation operation, std::source_location location) {
  if (path != IteratorPath::CONTIGUOUS) {
    const auto block_count = GetElementwiseBlockCount(parameters.num_elements_);
    ElementwiseKernel<<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(
        parameters, BinaryIteratorOperation<Output, Lhs, Rhs, Operation>{operation});
    return;
  }

#define TTL_LAUNCH_BINARY_WIDTH(width_value)                                                \
  case width_value:                                                                         \
    if constexpr (VALID_ELEMENTWISE_VECTOR_WIDTH<width_value, Output, Lhs, Rhs>) {          \
      LaunchContiguousBinary<Output, Lhs, Rhs, width_value>(stream, parameters, operation); \
      return;                                                                               \
    }                                                                                       \
    break
  switch (width) {
    TTL_LAUNCH_BINARY_WIDTH(1);
    TTL_LAUNCH_BINARY_WIDTH(2);
    TTL_LAUNCH_BINARY_WIDTH(4);
    TTL_LAUNCH_BINARY_WIDTH(8);
    TTL_LAUNCH_BINARY_WIDTH(16);
    default:
      break;
  }
#undef TTL_LAUNCH_BINARY_WIDTH
  throw InternalError("elementwise iterator selected an invalid binary vector width", location);
}

template <typename Output, typename First, typename Second, typename Third, typename Parameters, typename Operation>
void LaunchTernaryWithParameters(cudaStream_t stream, const Parameters &parameters, IteratorPath path, uint8_t width,
                                 Operation operation, std::source_location location) {
  if (path != IteratorPath::CONTIGUOUS) {
    const auto block_count = GetElementwiseBlockCount(parameters.num_elements_);
    ElementwiseKernel<<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(
        parameters, TernaryIteratorOperation<Output, First, Second, Third, Operation>{operation});
    return;
  }

#define TTL_LAUNCH_TERNARY_WIDTH(width_value)                                                            \
  case width_value:                                                                                      \
    if constexpr (VALID_ELEMENTWISE_VECTOR_WIDTH<width_value, Output, First, Second, Third>) {           \
      LaunchContiguousTernary<Output, First, Second, Third, width_value>(stream, parameters, operation); \
      return;                                                                                            \
    }                                                                                                    \
    break
  switch (width) {
    TTL_LAUNCH_TERNARY_WIDTH(1);
    TTL_LAUNCH_TERNARY_WIDTH(2);
    TTL_LAUNCH_TERNARY_WIDTH(4);
    TTL_LAUNCH_TERNARY_WIDTH(8);
    TTL_LAUNCH_TERNARY_WIDTH(16);
    default:
      break;
  }
#undef TTL_LAUNCH_TERNARY_WIDTH
  throw InternalError("elementwise iterator selected an invalid ternary vector width", location);
}

template <typename Output, typename Input, typename Operation>
void LaunchUnary(cudaStream_t stream, const ElementwiseIterator &iterator, Operation operation,
                 std::source_location location) {
  if (iterator.GetIndexWidth() == IndexWidth::UINT32) {
    LaunchUnaryWithParameters<Output, Input>(stream, iterator.MakeParameters32(location), iterator.GetPath(),
                                             iterator.GetVectorWidthElements(), operation, location);
    return;
  }
  LaunchUnaryWithParameters<Output, Input>(stream, iterator.MakeParameters64(), iterator.GetPath(),
                                           iterator.GetVectorWidthElements(), operation, location);
}

template <typename Output, typename Lhs, typename Rhs, typename Operation>
void LaunchBinary(cudaStream_t stream, const ElementwiseIterator &iterator, Operation operation,
                  std::source_location location) {
  if (iterator.GetIndexWidth() == IndexWidth::UINT32) {
    LaunchBinaryWithParameters<Output, Lhs, Rhs>(stream, iterator.MakeParameters32(location), iterator.GetPath(),
                                                 iterator.GetVectorWidthElements(), operation, location);
    return;
  }
  LaunchBinaryWithParameters<Output, Lhs, Rhs>(stream, iterator.MakeParameters64(), iterator.GetPath(),
                                               iterator.GetVectorWidthElements(), operation, location);
}

template <typename Output, typename First, typename Second, typename Third, typename Operation>
void LaunchTernary(cudaStream_t stream, const ElementwiseIterator &iterator, Operation operation,
                   std::source_location location) {
  if (iterator.GetIndexWidth() == IndexWidth::UINT32) {
    LaunchTernaryWithParameters<Output, First, Second, Third>(stream, iterator.MakeParameters32(location),
                                                              iterator.GetPath(), iterator.GetVectorWidthElements(),
                                                              operation, location);
    return;
  }
  LaunchTernaryWithParameters<Output, First, Second, Third>(stream, iterator.MakeParameters64(), iterator.GetPath(),
                                                            iterator.GetVectorWidthElements(), operation, location);
}

}  // namespace ttl::internal
