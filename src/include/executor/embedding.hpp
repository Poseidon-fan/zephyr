#pragma once

#include <cstdint>
#include <vector>

#include "common/types.hpp"
#include "executor/execution.hpp"
#include "model/embedding/model.hpp"
#include "model/loader.hpp"

namespace zephyr::executor {

/** Full token sequences to embed; inputs may have different lengths. */
struct EmbeddingBatch final : ExecutionBatch {
  std::vector<std::vector<token_id_t>> token_ids_;
};

struct EmbeddingExecutionSpec final : ExecutionSpec {
  explicit EmbeddingExecutionSpec(const model::embedding::ModelSpec &spec)
      : ExecutionSpec(spec.device_, spec.dtype_),
        max_seq_len_(spec.max_seq_len_),
        vocab_size_(spec.vocab_size_),
        embedding_size_(spec.embedding_size_),
        causal_attention_(spec.causal_attention_) {}

  int64_t max_seq_len_;
  int64_t vocab_size_;
  int64_t embedding_size_;
  bool causal_attention_;
};

/** Pair a model loader with embedding input preparation and rank execution. */
[[nodiscard]] auto CreateEmbeddingFactory(model::ModelLoader<model::embedding::Embedding> loader) -> ExecutionFactory;

}  // namespace zephyr::executor
