#pragma once

#include <cstdint>
#include <utility>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

namespace zephyr::layer {

/** Cached rotary embeddings with NeoX half-pairing or adjacent-pair rotation. */
class RotaryEmbedding final {
 public:
  /** head_dim is the rotated prefix width; it may be smaller than the Q/K head dimension. */
  RotaryEmbedding(ttl::ExecutionContext &context, float base, int64_t head_dim, int64_t max_position_embeddings,
                  bool is_gpt_neox, ttl::DType dtype);

  /**
   * Q/K are [batch, heads, sequence, head_dim]; Q and K may have different head counts.
   * Positions is an INT32/INT64 device tensor [batch * sequence]. Values index the cached positions.
   * The token-major transpose shares storage when contiguous and is copied otherwise, then rotates in place.
   * Consequently contiguous token-major input views are modified. Returns contiguous [B,H,S,D] tensors.
   * Dimensions outside the rotated prefix remain unchanged. Invalid positions are reported asynchronously.
   */
  [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &query, const ttl::Tensor &key,
                             const ttl::Tensor &positions) const -> std::pair<ttl::Tensor, ttl::Tensor>;

 private:
  RotaryEmbedding(ttl::Tensor cos, ttl::Tensor sin, bool is_gpt_neox)
      : cos_(std::move(cos)), sin_(std::move(sin)), is_gpt_neox_(is_gpt_neox) {}

  [[nodiscard]] static auto Create(ttl::ExecutionContext &context, float base, int64_t head_dim,
                                   int64_t max_position_embeddings, bool is_gpt_neox, ttl::DType dtype)
      -> RotaryEmbedding;

  ttl::Tensor cos_;
  ttl::Tensor sin_;
  bool is_gpt_neox_;
};

}  // namespace zephyr::layer
