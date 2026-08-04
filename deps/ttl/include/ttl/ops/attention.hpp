#pragma once

#include <cstdint>
#include <optional>
#include <source_location>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/** @brief Aligns a rectangular causal mask to the first or last query-key diagonal. */
enum class CausalAlignment : uint8_t {
  UPPER_LEFT,
  LOWER_RIGHT,
};

/** @brief Controls scaling and causal masking for scaled dot-product attention. */
struct SdpaOptions final {
  /** Multiplicative query-key scale. A missing value selects `1 / sqrt(head_dimension)`. */
  std::optional<float> scale_;
  /** Whether to apply a causal mask in addition to `mask`. */
  bool causal_{false};
  /** Placement of the causal diagonal when query and key sequence lengths differ. */
  CausalAlignment causal_alignment_{CausalAlignment::UPPER_LEFT};
};

/**
 * @brief Compute scaled dot-product attention into a caller-provided output tensor.
 *
 * `query`, `key`, and `value` have shapes `[B, Hq, Q, D]`, `[B, Hkv, K, D]`, and `[B, Hkv, K, Dv]`.
 * `Hq` must be divisible by `Hkv`, and `output` has shape `[B, Hq, Q, Dv]`. A boolean mask gates elements; a
 * floating mask is added to the attention scores and must broadcast to `[B, Hq, Q, K]`.
 *
 * All value tensors use one floating dtype. The output must be non-overlapping dense and must not alias an input.
 * Work is enqueued asynchronously on `context`.
 */
void ScaledDotProductAttentionOut(ExecutionContext &context, Tensor &output, const Tensor &query, const Tensor &key,
                                  const Tensor &value, const std::optional<Tensor> &mask = std::nullopt,
                                  const SdpaOptions &options = {},
                                  std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and compute scaled dot-product attention. */
[[nodiscard]] auto ScaledDotProductAttention(ExecutionContext &context, const Tensor &query, const Tensor &key,
                                             const Tensor &value, const std::optional<Tensor> &mask = std::nullopt,
                                             const SdpaOptions &options = {},
                                             std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
