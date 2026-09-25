#pragma once

#include <cstdint>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

#include "attention/sdpa.hpp"

namespace zephyr::model::embedding {

/** Model dimensions, computation dtype, and attention semantics for one tensor-parallel rank. */
struct ModelSpec final {
  int64_t max_seq_len_;
  int64_t vocab_size_;
  int64_t embedding_size_;
  ttl::Device device_;
  ttl::DType dtype_;
  bool causal_attention_;
};

/**
 * Complete text embedding model, including its configured pooling, projection, and normalization.
 * These transformations belong to the model; embeddings need not be normalized or match its hidden size.
 * All tensor-parallel ranks execute identical inputs and produce complete embedding vectors.
 */
class Embedding {
 public:
  virtual ~Embedding() = default;

  /**
   * Input IDs are [batch, sequence] complete, equally long sequences without padding or cached prefixes.
   * Each sequence starts at position zero. Returns floating [batch, embedding_size] on the model's device;
   * postprocessing may use a higher precision than the model's computation dtype.
   */
  [[nodiscard]] virtual auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input_ids,
                                     const attention::FlashParams &flash_params) const -> ttl::Tensor = 0;
  [[nodiscard]] virtual auto GetSpec() const noexcept -> const ModelSpec & = 0;
};

}  // namespace zephyr::model::embedding
