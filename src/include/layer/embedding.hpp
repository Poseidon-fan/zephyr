#pragma once

#include <cstdint>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

#include "weight/weight_builder.hpp"

namespace zephyr::layer {

/** Replicated token embedding table with shape [vocabulary_size, hidden_size]. */
class Embedding final {
 public:
  Embedding(ttl::Tensor embeddings, int64_t hidden_size);

  [[nodiscard]] static auto Load(ttl::ExecutionContext &context, int64_t vocabulary_size, int64_t hidden_size,
                                 const weight::WeightBuilder &builder) -> Embedding;

  /** Preserve the token ID dimensions and append the hidden dimension. */
  [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &token_ids) const -> ttl::Tensor;

  /** The output projection may share this tensor without copying its storage. */
  [[nodiscard]] auto GetWeight() const noexcept -> const ttl::Tensor & { return embeddings_; }

 private:
  ttl::Tensor embeddings_;
  int64_t hidden_size_;
};

}  // namespace zephyr::layer
