#include "ttl/internal/ops/elementwise_ops.hpp"

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>

#include <ttl/runtime/cuda_dtype.hpp>
#include <ttl/runtime/cuda_math.cuh>
#include "ttl/common/error.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_apply.cuh"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/scalar.hpp"

namespace ttl::internal {
namespace {

template <CudaStorageType T, ComparisonElementwiseOp operation>
struct ComparisonOperation final {
  __device__ auto operator()(T lhs, T rhs, int64_t /*index*/) const -> ElementwiseResult<bool> {
    if constexpr (IsCudaFloatingType<T>()) {
      const auto lhs_value = ToElementwiseFloat(lhs);
      const auto rhs_value = ToElementwiseFloat(rhs);
      if constexpr (operation == ComparisonElementwiseOp::EQUAL) {
        return {.value_ = lhs_value == rhs_value};
      } else if constexpr (operation == ComparisonElementwiseOp::NOT_EQUAL) {
        return {.value_ = lhs_value != rhs_value};
      } else if constexpr (operation == ComparisonElementwiseOp::LESS) {
        return {.value_ = lhs_value < rhs_value};
      } else if constexpr (operation == ComparisonElementwiseOp::LESS_EQUAL) {
        return {.value_ = lhs_value <= rhs_value};
      } else if constexpr (operation == ComparisonElementwiseOp::GREATER) {
        return {.value_ = lhs_value > rhs_value};
      } else {
        static_assert(operation == ComparisonElementwiseOp::GREATER_EQUAL);
        return {.value_ = lhs_value >= rhs_value};
      }
    } else {
      if constexpr (operation == ComparisonElementwiseOp::EQUAL) {
        return {.value_ = lhs == rhs};
      } else if constexpr (operation == ComparisonElementwiseOp::NOT_EQUAL) {
        return {.value_ = lhs != rhs};
      } else if constexpr (operation == ComparisonElementwiseOp::LESS) {
        return {.value_ = lhs < rhs};
      } else if constexpr (operation == ComparisonElementwiseOp::LESS_EQUAL) {
        return {.value_ = lhs <= rhs};
      } else if constexpr (operation == ComparisonElementwiseOp::GREATER) {
        return {.value_ = lhs > rhs};
      } else {
        static_assert(operation == ComparisonElementwiseOp::GREATER_EQUAL);
        return {.value_ = lhs >= rhs};
      }
    }
  }
};

template <CudaStorageType T, ComparisonElementwiseOp operation>
struct ScalarComparisonOperation final {
  T scalar_;

  __device__ auto operator()(T input, int64_t index) const -> ElementwiseResult<bool> {
    return ComparisonOperation<T, operation>{}(input, scalar_, index);
  }
};

template <CudaStorageType T, ComparisonElementwiseOp operation>
void LaunchComparisonOperation(cudaStream_t stream, const ElementwiseIterator &iterator,
                               std::source_location location) {
  LaunchBinary<bool, T, T>(stream, iterator, ComparisonOperation<T, operation>{}, location);
}

template <CudaStorageType T, ComparisonElementwiseOp operation>
void LaunchScalarComparisonOperation(cudaStream_t stream, const ElementwiseIterator &iterator, const Scalar &scalar,
                                     std::source_location location) {
  LaunchUnary<bool, T>(stream, iterator,
                       ScalarComparisonOperation<T, operation>{ConvertElementwiseScalar<T>(scalar, location)},
                       location);
}

template <CudaStorageType T>
void DispatchComparisonOperation(cudaStream_t stream, ComparisonElementwiseOp operation,
                                 const ElementwiseIterator &iterator, std::source_location location) {
  switch (operation) {
    case ComparisonElementwiseOp::EQUAL:
      LaunchComparisonOperation<T, ComparisonElementwiseOp::EQUAL>(stream, iterator, location);
      return;
    case ComparisonElementwiseOp::NOT_EQUAL:
      LaunchComparisonOperation<T, ComparisonElementwiseOp::NOT_EQUAL>(stream, iterator, location);
      return;
    case ComparisonElementwiseOp::LESS:
      LaunchComparisonOperation<T, ComparisonElementwiseOp::LESS>(stream, iterator, location);
      return;
    case ComparisonElementwiseOp::LESS_EQUAL:
      LaunchComparisonOperation<T, ComparisonElementwiseOp::LESS_EQUAL>(stream, iterator, location);
      return;
    case ComparisonElementwiseOp::GREATER:
      LaunchComparisonOperation<T, ComparisonElementwiseOp::GREATER>(stream, iterator, location);
      return;
    case ComparisonElementwiseOp::GREATER_EQUAL:
      LaunchComparisonOperation<T, ComparisonElementwiseOp::GREATER_EQUAL>(stream, iterator, location);
      return;
  }
  throw InternalError("invalid comparison elementwise operation", location);
}

template <CudaStorageType T>
void DispatchScalarComparisonOperation(cudaStream_t stream, ComparisonElementwiseOp operation,
                                       const ElementwiseIterator &iterator, const Scalar &scalar,
                                       std::source_location location) {
  switch (operation) {
    case ComparisonElementwiseOp::EQUAL:
      LaunchScalarComparisonOperation<T, ComparisonElementwiseOp::EQUAL>(stream, iterator, scalar, location);
      return;
    case ComparisonElementwiseOp::NOT_EQUAL:
      LaunchScalarComparisonOperation<T, ComparisonElementwiseOp::NOT_EQUAL>(stream, iterator, scalar, location);
      return;
    case ComparisonElementwiseOp::LESS:
      LaunchScalarComparisonOperation<T, ComparisonElementwiseOp::LESS>(stream, iterator, scalar, location);
      return;
    case ComparisonElementwiseOp::LESS_EQUAL:
      LaunchScalarComparisonOperation<T, ComparisonElementwiseOp::LESS_EQUAL>(stream, iterator, scalar, location);
      return;
    case ComparisonElementwiseOp::GREATER:
      LaunchScalarComparisonOperation<T, ComparisonElementwiseOp::GREATER>(stream, iterator, scalar, location);
      return;
    case ComparisonElementwiseOp::GREATER_EQUAL:
      LaunchScalarComparisonOperation<T, ComparisonElementwiseOp::GREATER_EQUAL>(stream, iterator, scalar, location);
      return;
  }
  throw InternalError("invalid scalar comparison elementwise operation", location);
}

void ValidateComparisonLaunch(cudaStream_t stream, const ElementwiseIterator &iterator, size_t operand_count,
                              std::source_location location) {
  if (stream == nullptr || iterator.GetOperandCount() != operand_count || iterator.GetNumElements() <= 0) {
    throw InternalError("invalid comparison elementwise launch plan", location);
  }
}

}  // namespace

void LaunchComparisonElementwise(cudaStream_t stream, DType input_dtype, ComparisonElementwiseOp operation,
                                 const ElementwiseIterator &iterator, std::source_location location) {
  ValidateComparisonLaunch(stream, iterator, 3, location);
  DispatchCudaDType(input_dtype, "comparison elementwise", [&]<CudaStorageType T>(std::type_identity<T>) {
    DispatchComparisonOperation<T>(stream, operation, iterator, location);
  });
}

void LaunchScalarComparisonElementwise(cudaStream_t stream, DType input_dtype, ComparisonElementwiseOp operation,
                                       const ElementwiseIterator &iterator, const Scalar &scalar,
                                       std::source_location location) {
  ValidateComparisonLaunch(stream, iterator, 2, location);
  DispatchCudaDType(input_dtype, "scalar comparison elementwise", [&]<CudaStorageType T>(std::type_identity<T>) {
    DispatchScalarComparisonOperation<T>(stream, operation, iterator, scalar, location);
  });
}

}  // namespace ttl::internal
