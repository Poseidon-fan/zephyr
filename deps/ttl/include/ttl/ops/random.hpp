#pragma once

#include <source_location>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/generator.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/** @brief Bounds for a uniform floating distribution over `[low_, high_)`. */
struct UniformOptions final {
  double low_{0.0};
  double high_{1.0};
};

/** @brief Parameters for a floating normal distribution. */
struct NormalOptions final {
  double mean_{0.0};
  double standard_deviation_{1.0};
};

/**
 * @brief Fill a floating tensor with uniform samples from `[low_, high_)`.
 *
 * Bounds must be finite and satisfy `low_ < high_`. The generator must belong to the exact device and stream owned by
 * `context`; generator use is serialized and the operation is enqueued asynchronously.
 */
void UniformOut(ExecutionContext &context, Tensor &output, Generator &generator, const UniformOptions &options = {},
                std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous floating tensor and fill it with uniform samples. */
[[nodiscard]] auto Uniform(ExecutionContext &context, const Shape &shape, DType dtype, Generator &generator,
                           const UniformOptions &options = {},
                           std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Fill a floating tensor with normally distributed samples.
 *
 * The mean must be finite and the standard deviation must be finite and non-negative. The generator must belong to
 * the exact device and stream owned by `context`.
 */
void NormalOut(ExecutionContext &context, Tensor &output, Generator &generator, const NormalOptions &options = {},
               std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous floating tensor and fill it with normal samples. */
[[nodiscard]] auto Normal(ExecutionContext &context, const Shape &shape, DType dtype, Generator &generator,
                          const NormalOptions &options = {},
                          std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
