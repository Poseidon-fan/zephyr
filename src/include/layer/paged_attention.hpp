#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

#include "attention/sdpa.hpp"

namespace zephyr::layer {

/**
 * Rank-local metadata shared by attention layers. Slot mappings may be flat or [batch, sequence].
 * Context lengths include this call's new tokens. Referenced cache pages must already be allocated,
 * and any cached prefix must have been computed. Host lengths and device tensors describe the same batch.
 */
struct PagedAttentionInputMetadata final {
  std::optional<ttl::Tensor> block_tables_;
  std::optional<ttl::Tensor> context_lens_;
  std::optional<std::vector<int64_t>> paged_context_lens_cpu_;
  ttl::Tensor slot_mappings_;
  std::optional<int64_t> max_context_len_;
  bool is_first_prompt_chunk_{false};
  std::optional<std::vector<int64_t>> num_cached_tokens_;
  std::optional<std::vector<int64_t>> query_lens_;
  std::optional<ttl::Tensor> cu_seqlens_kv_;
};

/** Selects prompt, cached-prefix, or paged decode execution for model-produced Q/K/V. */
class PagedAttention final {
 public:
  /**
   * Q/K/V are [batch, heads, sequence, head_dim]; Q/K already contain positional encoding.
   * Ordinary prompt returns [batch, heads, sequence, head_dim]. Standard decode returns
   * [batch * sequence, heads, head_dim]; gathered decode returns [batch * sequence, heads, 1, head_dim].
   * A custom mask or unsupported kernel shape selects gathered decode. Cached-prefix output follows the mask:
   * None returns token-major, otherwise head-major.
   * Cache handles share storage with their owners. Both caches may be absent for an uncached prompt.
   */
  [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &query, const ttl::Tensor &key,
                             const ttl::Tensor &value, const attention::AttentionMask &attention_mask,
                             std::optional<ttl::Tensor> key_cache, std::optional<ttl::Tensor> value_cache,
                             const PagedAttentionInputMetadata &input_metadata,
                             const attention::SdpaParams &sdpa_params, const attention::FlashParams *flash_params) const
      -> ttl::Tensor;

 private:
  /** Gather logical contexts and return head-major attention; prefix output layout is selected by Forward. */
  [[nodiscard]] static auto GatherAttention(ttl::ExecutionContext &context, const ttl::Tensor &query,
                                            const ttl::Tensor &key_cache, const ttl::Tensor &value_cache,
                                            const attention::AttentionMask &attention_mask,
                                            const PagedAttentionInputMetadata &input_metadata,
                                            const attention::SdpaParams &sdpa_params,
                                            const attention::FlashParams *flash_params, bool decode) -> ttl::Tensor;
};

}  // namespace zephyr::layer
