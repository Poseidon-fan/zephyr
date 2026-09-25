#include "ttl/internal/ops/elementwise_ops.hpp"

#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda/std/type_traits>

#include <ttl/runtime/cuda_dtype.hpp>
#include <ttl/runtime/cuda_math.cuh>
#include "ttl/common/error.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_apply.cuh"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/runtime/execution/device_error.cuh"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/scalar.hpp"

namespace ttl::internal {
namespace {

template <typename T>
__device__ auto AddModulo(T lhs, T rhs) -> T {
  using Unsigned = cuda::std::make_unsigned_t<T>;
  const auto result = static_cast<Unsigned>(lhs) + static_cast<Unsigned>(rhs);
  return __builtin_bit_cast(T, result);
}

template <typename T>
__device__ auto SubtractModulo(T lhs, T rhs) -> T {
  using Unsigned = cuda::std::make_unsigned_t<T>;
  const auto result = static_cast<Unsigned>(lhs) - static_cast<Unsigned>(rhs);
  return __builtin_bit_cast(T, result);
}

template <typename T>
__device__ auto MultiplyModulo(T lhs, T rhs) -> T {
  using Unsigned = cuda::std::make_unsigned_t<T>;
  const auto result = static_cast<Unsigned>(lhs) * static_cast<Unsigned>(rhs);
  return __builtin_bit_cast(T, result);
}

template <CudaStorageType T, BinaryElementwiseOp operation>
struct BinaryOperation final {
  DeviceErrorLaunchContext error_context_;

  __device__ auto operator()(T lhs, T rhs, int64_t index) const -> ElementwiseResult<T> {
    if constexpr (IsCudaFloatingType<T>()) {
      const auto lhs_value = ToElementwiseFloat(lhs);
      const auto rhs_value = ToElementwiseFloat(rhs);
      if constexpr (operation == BinaryElementwiseOp::ADD) {
        return {.value_ = FromElementwiseFloat<T>(lhs_value + rhs_value)};
      } else if constexpr (operation == BinaryElementwiseOp::SUBTRACT) {
        return {.value_ = FromElementwiseFloat<T>(lhs_value - rhs_value)};
      } else if constexpr (operation == BinaryElementwiseOp::MULTIPLY) {
        return {.value_ = FromElementwiseFloat<T>(lhs_value * rhs_value)};
      } else if constexpr (operation == BinaryElementwiseOp::DIVIDE) {
        return {.value_ = FromElementwiseFloat<T>(lhs_value / rhs_value)};
      } else if constexpr (operation == BinaryElementwiseOp::MAXIMUM) {
        if (isnan(lhs_value)) {
          return {.value_ = lhs};
        }
        if (isnan(rhs_value)) {
          return {.value_ = rhs};
        }
        return {.value_ = lhs_value > rhs_value ? lhs : rhs};
      } else {
        static_assert(operation == BinaryElementwiseOp::MINIMUM);
        if (isnan(lhs_value)) {
          return {.value_ = lhs};
        }
        if (isnan(rhs_value)) {
          return {.value_ = rhs};
        }
        return {.value_ = lhs_value < rhs_value ? lhs : rhs};
      }
    } else {
      static_assert(IsCudaSignedIntegerType<T>());
      if constexpr (operation == BinaryElementwiseOp::ADD) {
        return {.value_ = AddModulo(lhs, rhs)};
      } else if constexpr (operation == BinaryElementwiseOp::SUBTRACT) {
        return {.value_ = SubtractModulo(lhs, rhs)};
      } else if constexpr (operation == BinaryElementwiseOp::MULTIPLY) {
        return {.value_ = MultiplyModulo(lhs, rhs)};
      } else if constexpr (operation == BinaryElementwiseOp::DIVIDE) {
        if (rhs == 0) {
          ReportDeviceError(error_context_, DeviceErrorCode::INTEGER_DIVIDE_BY_ZERO, index, uint64_t{0});
          return {.value_ = T{}, .write_ = false};
        }
        using Unsigned = cuda::std::make_unsigned_t<T>;
        const auto minimum = __builtin_bit_cast(T, Unsigned{1} << ((sizeof(T) * 8) - 1));
        if (lhs == minimum && rhs == T{-1}) {
          return {.value_ = minimum};
        }
        return {.value_ = static_cast<T>(lhs / rhs)};
      } else if constexpr (operation == BinaryElementwiseOp::MAXIMUM) {
        return {.value_ = lhs > rhs ? lhs : rhs};
      } else {
        static_assert(operation == BinaryElementwiseOp::MINIMUM);
        return {.value_ = lhs < rhs ? lhs : rhs};
      }
    }
  }
};

template <CudaStorageType T, BinaryElementwiseOp operation>
struct ScalarBinaryOperation final {
  BinaryOperation<T, operation> operation_;
  T scalar_;

  __device__ auto operator()(T input, int64_t index) const -> ElementwiseResult<T> {
    return operation_(input, scalar_, index);
  }
};

template <CudaStorageType T, BinaryElementwiseOp operation>
void LaunchBinaryOperation(cudaStream_t stream, const ElementwiseIterator &iterator,
                           const DeviceErrorLaunchContext &error_context, std::source_location location) {
  LaunchBinary<T, T, T>(stream, iterator, BinaryOperation<T, operation>{error_context}, location);
}

template <CudaStorageType T, BinaryElementwiseOp operation>
void LaunchScalarBinaryOperation(cudaStream_t stream, const ElementwiseIterator &iterator, const Scalar &scalar,
                                 const DeviceErrorLaunchContext &error_context, std::source_location location) {
  const auto converted = ConvertElementwiseScalar<T>(scalar, location);
  LaunchUnary<T, T>(stream, iterator,
                    ScalarBinaryOperation<T, operation>{
                        .operation_ = BinaryOperation<T, operation>{error_context},
                        .scalar_ = converted,
                    },
                    location);
}

template <CudaStorageType T>
void DispatchBinaryOperation(cudaStream_t stream, BinaryElementwiseOp operation, const ElementwiseIterator &iterator,
                             const DeviceErrorLaunchContext &error_context, std::source_location location) {
  switch (operation) {
    case BinaryElementwiseOp::ADD:
      LaunchBinaryOperation<T, BinaryElementwiseOp::ADD>(stream, iterator, error_context, location);
      return;
    case BinaryElementwiseOp::SUBTRACT:
      LaunchBinaryOperation<T, BinaryElementwiseOp::SUBTRACT>(stream, iterator, error_context, location);
      return;
    case BinaryElementwiseOp::MULTIPLY:
      LaunchBinaryOperation<T, BinaryElementwiseOp::MULTIPLY>(stream, iterator, error_context, location);
      return;
    case BinaryElementwiseOp::DIVIDE:
      LaunchBinaryOperation<T, BinaryElementwiseOp::DIVIDE>(stream, iterator, error_context, location);
      return;
    case BinaryElementwiseOp::MAXIMUM:
      LaunchBinaryOperation<T, BinaryElementwiseOp::MAXIMUM>(stream, iterator, error_context, location);
      return;
    case BinaryElementwiseOp::MINIMUM:
      LaunchBinaryOperation<T, BinaryElementwiseOp::MINIMUM>(stream, iterator, error_context, location);
      return;
  }
  throw InternalError("invalid binary elementwise operation", location);
}

template <CudaStorageType T>
void DispatchScalarBinaryOperation(cudaStream_t stream, BinaryElementwiseOp operation,
                                   const ElementwiseIterator &iterator, const Scalar &scalar,
                                   const DeviceErrorLaunchContext &error_context, std::source_location location) {
  switch (operation) {
    case BinaryElementwiseOp::ADD:
      LaunchScalarBinaryOperation<T, BinaryElementwiseOp::ADD>(stream, iterator, scalar, error_context, location);
      return;
    case BinaryElementwiseOp::SUBTRACT:
      LaunchScalarBinaryOperation<T, BinaryElementwiseOp::SUBTRACT>(stream, iterator, scalar, error_context, location);
      return;
    case BinaryElementwiseOp::MULTIPLY:
      LaunchScalarBinaryOperation<T, BinaryElementwiseOp::MULTIPLY>(stream, iterator, scalar, error_context, location);
      return;
    case BinaryElementwiseOp::DIVIDE:
      LaunchScalarBinaryOperation<T, BinaryElementwiseOp::DIVIDE>(stream, iterator, scalar, error_context, location);
      return;
    case BinaryElementwiseOp::MAXIMUM:
      LaunchScalarBinaryOperation<T, BinaryElementwiseOp::MAXIMUM>(stream, iterator, scalar, error_context, location);
      return;
    case BinaryElementwiseOp::MINIMUM:
      LaunchScalarBinaryOperation<T, BinaryElementwiseOp::MINIMUM>(stream, iterator, scalar, error_context, location);
      return;
  }
  throw InternalError("invalid scalar binary elementwise operation", location);
}

void ValidateBinaryLaunch(cudaStream_t stream, const ElementwiseIterator &iterator, size_t operand_count,
                          std::source_location location) {
  if (stream == nullptr || iterator.GetOperandCount() != operand_count || iterator.GetNumElements() <= 0) {
    throw InternalError("invalid binary elementwise launch plan", location);
  }
}

}  // namespace

void LaunchBinaryElementwise(cudaStream_t stream, DType dtype, BinaryElementwiseOp operation,
                             const ElementwiseIterator &iterator, const DeviceErrorLaunchContext &error_context,
                             std::source_location location) {
  ValidateBinaryLaunch(stream, iterator, 3, location);
  DispatchCudaDType(dtype, "binary elementwise", [&]<CudaStorageType T>(std::type_identity<T>) {
    if constexpr (IsCudaFloatingType<T>() || IsCudaSignedIntegerType<T>()) {
      DispatchBinaryOperation<T>(stream, operation, iterator, error_context, location);
    } else {
      throw InternalError("binary elementwise launch received an unsupported dtype", location);
    }
  });
}

void LaunchScalarBinaryElementwise(cudaStream_t stream, DType dtype, BinaryElementwiseOp operation,
                                   const ElementwiseIterator &iterator, const Scalar &scalar,
                                   const DeviceErrorLaunchContext &error_context, std::source_location location) {
  ValidateBinaryLaunch(stream, iterator, 2, location);
  DispatchCudaDType(dtype, "scalar binary elementwise", [&]<CudaStorageType T>(std::type_identity<T>) {
    if constexpr (IsCudaFloatingType<T>() || IsCudaSignedIntegerType<T>()) {
      DispatchScalarBinaryOperation<T>(stream, operation, iterator, scalar, error_context, location);
    } else {
      throw InternalError("scalar binary elementwise launch received an unsupported dtype", location);
    }
  });
}

}  // namespace ttl::internal
