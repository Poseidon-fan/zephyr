#pragma once

#include <cstdint>
#include <optional>
#include <source_location>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/** @brief Selects the trailing normalized dimensions and numerical stabilizer. */
struct NormOptions final {
  /** Number of trailing input dimensions reduced by the normalization. */
  int64_t normalized_rank_;
  /** Non-negative value added to the variance or mean square before reciprocal square root. */
  float epsilon_{1.0e-5F};
};

/**
 * @brief Apply layer normalization over the trailing `normalized_rank_` dimensions.
 *
 * Input, output, and optional affine tensors use one floating dtype. `weight` and `bias`, when present, match the
 * normalized trailing shape. The output must be non-overlapping dense and disjoint from all inputs.
 */
void LayerNormOut(ExecutionContext &context, Tensor &output, const Tensor &input, const std::optional<Tensor> &weight,
                  const std::optional<Tensor> &bias, const NormOptions &options,
                  std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and apply layer normalization. */
[[nodiscard]] auto LayerNorm(ExecutionContext &context, const Tensor &input, const std::optional<Tensor> &weight,
                             const std::optional<Tensor> &bias, const NormOptions &options,
                             std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Apply root-mean-square normalization over the trailing `normalized_rank_` dimensions.
 *
 * Input, output, and optional weight use one floating dtype. `weight`, when present, matches the normalized trailing
 * shape. The output must be non-overlapping dense and disjoint from all inputs.
 */
void RmsNormOut(ExecutionContext &context, Tensor &output, const Tensor &input, const std::optional<Tensor> &weight,
                const NormOptions &options, std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and apply root-mean-square normalization. */
[[nodiscard]] auto RmsNorm(ExecutionContext &context, const Tensor &input, const std::optional<Tensor> &weight,
                           const NormOptions &options, std::source_location location = std::source_location::current())
    -> Tensor;

}  // namespace ttl
