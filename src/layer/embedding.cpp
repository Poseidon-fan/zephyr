#include "layer/embedding.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <utility>

#include <ttl/ops/indexing.hpp>
#include <ttl/tensor/layout.hpp>
#include <ttl/tensor/shape.hpp>

namespace zephyr::layer {

Embedding::Embedding(ttl::Tensor embeddings, int64_t hidden_size)
    : embeddings_(std::move(embeddings)), hidden_size_(hidden_size) {}

auto Embedding::Load(ttl::ExecutionContext &context, int64_t vocabulary_size, int64_t hidden_size,
                     const weight::WeightBuilder &builder) -> Embedding {
  return Embedding{builder.Get(context, ttl::Shape{vocabulary_size, hidden_size}, "weight"), hidden_size};
}

auto Embedding::Forward(ttl::ExecutionContext &context, const ttl::Tensor &token_ids) const -> ttl::Tensor {
  auto output = ttl::Embedding(context, embeddings_, token_ids);
  auto dimensions = std::array<int64_t, ttl::TTL_MAX_RANK>{};
  const auto shape = output.GetShape();
  for (size_t axis = 0; axis + 1 < shape.GetRank(); ++axis) {
    dimensions[axis] = shape.GetDimension(axis);
  }
  dimensions[shape.GetRank() - 1] = hidden_size_;
  return ttl::View(output, ttl::Shape{std::span<const int64_t>{dimensions.data(), shape.GetRank()}});
}

}  // namespace zephyr::layer
