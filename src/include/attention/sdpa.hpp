#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

namespace zephyr::attention {

/** Distinguishes unmasked attention, kernel-provided causality, and an explicit additive mask. */
class AttentionMask final {
 public:
  [[nodiscard]] static auto None() -> AttentionMask { return AttentionMask{std::monostate{}}; }
  [[nodiscard]] static auto CausalFlash() -> AttentionMask { return AttentionMask{Causal{}}; }
  [[nodiscard]] static auto Custom(ttl::Tensor tensor) -> AttentionMask { return AttentionMask{std::move(tensor)}; }

  [[nodiscard]] auto IsNone() const noexcept -> bool { return std::holds_alternative<std::monostate>(value_); }
  [[nodiscard]] auto IsCustom() const noexcept -> bool { return std::holds_alternative<ttl::Tensor>(value_); }
  [[nodiscard]] auto AsOptionTensor() const noexcept -> const ttl::Tensor * {
    return std::get_if<ttl::Tensor>(&value_);
  }

 private:
  struct Causal {};
  using Value = std::variant<std::monostate, Causal, ttl::Tensor>;

  explicit AttentionMask(Value value) : value_(std::move(value)) {}

  Value value_;
};

/** Parameters of one attention call; the KV group count is query_heads / kv_heads. */
struct SdpaParams final {
  int64_t n_kv_groups_;
  float softmax_scale_;
};

/** Device cumulative key lengths and their maximum, for this execution context's device. */
struct FlashKMeta final {
  int32_t max_{0};
  std::optional<ttl::Tensor> cumulative_seqlens_;
};

/** Physical sequence boundaries; packed tensors concatenate logical sequences along their sequence axis. */
struct FlashParams final {
  int32_t max_q_{0};
  std::optional<ttl::Tensor> cumulative_seqlens_q_;
  FlashKMeta logical_k_;
  bool causal_{false};
  bool packed_{false};
};

/**
 * Scaled dot-product attention over [batch, heads, sequence, head_dim] tensors.
 * Returns [batch, query_heads, query_length, value_head_dim]. A custom mask contains additive scores.
 * Variable-length boundaries are read on the host to invoke TTL's rectangular SDPA once per sequence.
 */
[[nodiscard]] auto RunAttention(ttl::ExecutionContext &context, const ttl::Tensor &query, const ttl::Tensor &key,
                                const ttl::Tensor &value, const AttentionMask &mask, const FlashParams *flash_params,
                                const SdpaParams &sdpa_params) -> ttl::Tensor;

}  // namespace zephyr::attention
