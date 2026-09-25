#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "common/types.hpp"
#include "executor/execution.hpp"
#include "kv_cache/types.hpp"
#include "model/causal_lm/model.hpp"
#include "model/loader.hpp"

namespace zephyr::executor {

struct CausalLMOptions final {
  /** Absent capacity permits initial prompts without saving KV, but no incremental decoding. */
  std::optional<kv_cache::CacheCapacity> cache_capacity_;
};

/** Generation capabilities shared by all ranks; device placement is reported separately. */
struct CausalLMExecutionSpec final : ExecutionSpec {
  CausalLMExecutionSpec(const model::causal_lm::ModelSpec &model_spec,
                        std::optional<kv_cache::CacheCapacity> cache_capacity);

  int64_t max_seq_len_;
  int64_t vocab_size_;
  std::optional<kv_cache::CacheCapacity> cache_capacity_;
  bool supports_packed_prefill_{false};
};

enum class CausalLMPhase : uint8_t { PREFILL, DECODE };

/** One sequence's new tokens and the already allocated cache pages covering its complete context. */
struct CausalLMInput final {
  std::vector<token_id_t> token_ids_;
  int64_t num_computed_tokens_;
  std::vector<kv_cache::block_id_t> block_ids_;
  /** Output positions relative to token_ids_; a zero length advances the cache without producing logits. */
  model::causal_lm::LogitsRange logits_range_;
};

struct CausalLMBatch final : ExecutionBatch {
  CausalLMPhase phase_{CausalLMPhase::PREFILL};
  /** Applies to PREFILL only; chunk boundaries are determined by the caller. */
  bool is_final_prompt_chunk_{false};
  std::vector<CausalLMInput> inputs_;
};

/** Bind causal-model loading and cache capacity before starting the rank workers. */
[[nodiscard]] auto CreateCausalLMFactory(CausalLMOptions options = {},
                                         model::ModelLoader<model::causal_lm::CausalLM> loader = model::LoadCausalLM)
    -> ExecutionFactory;

}  // namespace zephyr::executor
