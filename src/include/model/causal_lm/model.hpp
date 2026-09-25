#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

#include "attention/sdpa.hpp"
#include "kv_cache/cache_engine.hpp"
#include "layer/paged_attention.hpp"

namespace zephyr::model::causal_lm {

/** One logical sequence's output positions, relative to the tokens supplied in this forward call. */
struct LogitsRange final {
  int64_t start_;
  int64_t length_;
};

/** Model dimensions and storage requirements for one tensor-parallel rank. */
struct ModelSpec final {
  int64_t max_seq_len_;
  int64_t vocab_size_;
  ttl::Device device_;
  ttl::DType dtype_;
  std::vector<kv_cache::LayerCacheSpec> layer_specs_;
};

/**
 * Borrowed inputs for one rank's forward call. The caller owns and prepares positions, page tables, and slots.
 * Positions are an INT32/INT64 device vector in physical token order, including logical resets in packed prompts.
 * Logits ranges have one entry per logical sequence and a common output length. They are not KV context lengths.
 * Prompt and prefix flash metadata must declare causal attention. A later chunk dispatched as decode supplies
 * one block-table/context-length row per query token; prefix gather instead uses logical sequence rows.
 * An absent cache is valid for an initial prompt; it computes attention without storing K/V.
 */
struct ModelForwardContext final {
  const ttl::Tensor &positions_;
  const layer::PagedAttentionInputMetadata &paged_attention_;
  const attention::FlashParams &flash_params_;
  std::span<const LogitsRange> logits_ranges_;
  const kv_cache::CacheEngine *cache_{nullptr};

  /** Select [logical_batch, output_length, hidden_size] before applying the vocabulary projection. */
  [[nodiscard]] auto SelectLogits(ttl::ExecutionContext &context, const ttl::Tensor &hidden_states) const
      -> ttl::Tensor;
};

/** Rank-local autoregressive language model. All TP ranks must execute the same batch in the same order. */
class CausalLM {
 public:
  virtual ~CausalLM() = default;

  /** Input IDs are [physical_batch, sequence]; returns [logical_batch, output_length, vocabulary_size]. */
  [[nodiscard]] virtual auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input_ids,
                                     const ModelForwardContext &forward_context) const -> ttl::Tensor = 0;
  [[nodiscard]] virtual auto GetSpec() const noexcept -> const ModelSpec & = 0;
};

}  // namespace zephyr::model::causal_lm
