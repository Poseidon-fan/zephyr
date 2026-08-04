#pragma once

#include <cstdint>
#include <optional>
#include <source_location>

#include "ttl/ops/elementwise.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/** @brief Controls the compute mode used by floating-point matrix multiplication. */
struct MatmulOptions final {
  /** Allow TensorFloat-32 inputs for FLOAT32 computation on supported NVIDIA GPUs. */
  bool allow_tf32_{true};
};

/** @brief Optional activation fused into `Linear`. */
enum class LinearActivation : uint8_t {
  NONE,
  RELU,
  GELU,
};

/** @brief Controls the fused activation and matrix-multiplication mode used by `Linear`. */
struct LinearOptions final {
  LinearActivation activation_{LinearActivation::NONE};
  GeluApproximation gelu_approximation_{GeluApproximation::NONE};
  MatmulOptions matmul_;
};

/**
 * @brief Multiply two rank-two floating tensors into a caller-provided output.
 *
 * For `lhs` shaped `[M, K]` and `rhs` shaped `[K, N]`, `output` must be `[M, N]`. All tensors use one dtype and the
 * output must be non-overlapping dense and disjoint from both inputs.
 */
void MatmulOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
               const MatmulOptions &options = {}, std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and multiply two rank-two tensors. */
[[nodiscard]] auto Matmul(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                          const MatmulOptions &options = {},
                          std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Multiply batches of rank-two matrices with broadcast batch dimensions.
 *
 * Both inputs must have rank at least three. Their trailing dimensions follow matrix multiplication rules, and their
 * leading dimensions broadcast. The output must be non-overlapping dense and disjoint from both inputs.
 */
void BatchedMatmulOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                      const MatmulOptions &options = {},
                      std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and perform broadcast batched matrix multiplication. */
[[nodiscard]] auto BatchedMatmul(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                                 const MatmulOptions &options = {},
                                 std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Apply a floating linear transform, optional bias, and optional activation.
 *
 * `input` has shape `[..., in_features]`, `weight` has shape `[out_features, in_features]`, and an optional `bias`
 * has shape `[out_features]`. The output shape is `[..., out_features]` and must not alias any input.
 */
void LinearOut(ExecutionContext &context, Tensor &output, const Tensor &input, const Tensor &weight,
               const std::optional<Tensor> &bias = std::nullopt, const LinearOptions &options = {},
               std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and apply a linear transform. */
[[nodiscard]] auto Linear(ExecutionContext &context, const Tensor &input, const Tensor &weight,
                          const std::optional<Tensor> &bias = std::nullopt, const LinearOptions &options = {},
                          std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
