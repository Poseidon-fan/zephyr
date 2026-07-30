#pragma once

#include <cstdint>
#include <optional>
#include <source_location>

#include <driver_types.h>

#include "ttl/dtype.hpp"
#include "ttl/internal/device_error.hpp"
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/scalar.hpp"

namespace ttl::internal {

enum class BinaryElementwiseOp : uint8_t {
  ADD,
  SUBTRACT,
  MULTIPLY,
  DIVIDE,
  MAXIMUM,
  MINIMUM,
};

enum class ComparisonElementwiseOp : uint8_t {
  EQUAL,
  NOT_EQUAL,
  LESS,
  LESS_EQUAL,
  GREATER,
  GREATER_EQUAL,
};

enum class UnaryElementwiseOp : uint8_t {
  NEGATE,
  ABS,
  EXP,
  LOG,
  SQRT,
  RSQRT,
  SIN,
  COS,
  TANH,
  SIGMOID,
  RELU,
  SILU,
};

enum class LogicalElementwiseOp : uint8_t {
  AND,
  OR,
  NOT,
};

void LaunchBinaryElementwise(cudaStream_t stream, DType dtype, BinaryElementwiseOp operation,
                             const ElementwiseIterator &iterator, const DeviceErrorLaunchContext &error_context,
                             std::source_location location);
void LaunchScalarBinaryElementwise(cudaStream_t stream, DType dtype, BinaryElementwiseOp operation,
                                   const ElementwiseIterator &iterator, const Scalar &scalar,
                                   const DeviceErrorLaunchContext &error_context, std::source_location location);
void LaunchComparisonElementwise(cudaStream_t stream, DType input_dtype, ComparisonElementwiseOp operation,
                                 const ElementwiseIterator &iterator, std::source_location location);
void LaunchScalarComparisonElementwise(cudaStream_t stream, DType input_dtype, ComparisonElementwiseOp operation,
                                       const ElementwiseIterator &iterator, const Scalar &scalar,
                                       std::source_location location);
void LaunchUnaryElementwise(cudaStream_t stream, DType dtype, UnaryElementwiseOp operation,
                            const ElementwiseIterator &iterator, std::source_location location);
void LaunchGeluElementwise(cudaStream_t stream, DType dtype, GeluApproximation approximation,
                           const ElementwiseIterator &iterator, std::source_location location);
void LaunchClampElementwise(cudaStream_t stream, DType dtype, const ElementwiseIterator &iterator,
                            const std::optional<Scalar> &minimum, const std::optional<Scalar> &maximum,
                            std::source_location location);
void LaunchLogicalElementwise(cudaStream_t stream, LogicalElementwiseOp operation, const ElementwiseIterator &iterator,
                              std::source_location location);
void LaunchWhereElementwise(cudaStream_t stream, DType value_dtype, const ElementwiseIterator &iterator,
                            std::source_location location);

}  // namespace ttl::internal
