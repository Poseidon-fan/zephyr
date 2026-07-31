#pragma once

#include <source_location>

#include "ttl/dtype.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/scalar.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

/** Allocate an uninitialized canonical contiguous tensor matching an input's shape and dtype. */
[[nodiscard]] auto EmptyLike(ExecutionContext &context, const Tensor &input,
                             std::source_location location = std::source_location::current()) -> Tensor;

/** Allocate a canonical contiguous tensor and fill it with a checked scalar conversion. */
[[nodiscard]] auto Full(ExecutionContext &context, const Shape &shape, Scalar value, DType dtype,
                        std::source_location location = std::source_location::current()) -> Tensor;

/** Allocate a canonical contiguous tensor filled with the zero value of its dtype. */
[[nodiscard]] auto Zeros(ExecutionContext &context, const Shape &shape, DType dtype,
                         std::source_location location = std::source_location::current()) -> Tensor;

/** Allocate a canonical contiguous tensor filled with the one value of its dtype. */
[[nodiscard]] auto Ones(ExecutionContext &context, const Shape &shape, DType dtype,
                        std::source_location location = std::source_location::current()) -> Tensor;

/**
 * Create a one-dimensional half-open arithmetic sequence.
 *
 * Supported output dtypes are INT32, INT64, and FLOAT32. The step must be nonzero; floating-point bounds and step
 * must be finite.
 */
[[nodiscard]] auto Arange(ExecutionContext &context, Scalar start, Scalar end, Scalar step, DType dtype,
                          std::source_location location = std::source_location::current()) -> Tensor;

/** Fill every logical element of a writable tensor on the context stream. */
void FillOut(ExecutionContext &context, Tensor &output, Scalar value,
             std::source_location location = std::source_location::current());

}  // namespace ttl
