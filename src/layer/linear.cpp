#include "layer/linear.hpp"

#include <cstddef>
#include <limits>
#include <utility>

#include <ttl/ops/elementwise.hpp>
#include <ttl/ops/matmul.hpp>
#include <ttl/tensor/shape.hpp>

#include "common/exception.hpp"

namespace zephyr::layer {

auto Linear::Load(ttl::ExecutionContext &context, int64_t in_features, int64_t out_features, bool bias,
                  const weight::WeightBuilder &builder) -> Linear {
  auto weight = builder.Get(context, ttl::Shape{out_features, in_features}, "weight");
  auto bias_tensor = bias ? std::optional{builder.Get(context, ttl::Shape{out_features}, "bias")} : std::nullopt;
  return Linear{std::move(weight), std::move(bias_tensor)};
}

auto Linear::Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor {
  return ttl::Linear(context, input, weight_, bias_);
}

auto ColumnParallelLayer::Load(ttl::ExecutionContext &context, int64_t in_features, int64_t out_features, bool bias,
                               const parallel::TpRankContext &rank, const weight::WeightBuilder &builder)
    -> ColumnParallelLayer {
  return LoadWithShard(context, in_features, out_features, bias, rank,
                       weight::Shard::Uniform(0, rank.Rank(), rank.WorldSize()), builder);
}

auto ColumnParallelLayer::LoadWithShard(ttl::ExecutionContext &context, int64_t in_features, int64_t out_features,
                                        bool bias, [[maybe_unused]] const parallel::TpRankContext &rank,
                                        weight::Shard shard, const weight::WeightBuilder &builder)
    -> ColumnParallelLayer {
  auto weight = builder.Get(context, ttl::Shape{out_features, in_features}, "weight", shard);
  auto bias_tensor = bias && builder.GetParameterInfo("bias").has_value()
                         ? std::optional{builder.Get(context, ttl::Shape{out_features}, "bias", shard)}
                         : std::nullopt;
  return ColumnParallelLayer{Linear{std::move(weight), std::nullopt}, std::move(bias_tensor)};
}

auto ColumnParallelLayer::Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor {
  auto output = projection_.Forward(context, input);
  if (bias_.has_value()) {
    ttl::AddOut(context, output, output, *bias_);
  }
  return output;
}

auto RowParallelLayer::Load(ttl::ExecutionContext &context, int64_t in_features, int64_t out_features, bool bias,
                            const parallel::TpRankContext &rank, const weight::WeightBuilder &builder)
    -> RowParallelLayer {
  const auto shard = weight::Shard::Uniform(1, rank.Rank(), rank.WorldSize());
  auto weight = builder.Get(context, ttl::Shape{out_features, in_features}, "weight", shard);
  auto bias_tensor = bias && builder.GetParameterInfo("bias").has_value()
                         ? std::optional{builder.Get(context, ttl::Shape{out_features}, "bias")}
                         : std::nullopt;
  return RowParallelLayer{Linear{std::move(weight), std::nullopt}, std::move(bias_tensor), rank};
}

auto RowParallelLayer::Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor {
  auto output = projection_.Forward(context, input);
  if (rank_.WorldSize() > 1) {
    rank_.AllReduceSum(context, output, output);
  }
  if (bias_.has_value()) {
    ttl::AddOut(context, output, output, *bias_);
  }
  return output;
}

auto ComputeKvShard(int64_t num_kv_heads, int64_t head_dim, const parallel::TpRankContext &rank) -> weight::Shard {
  if (num_kv_heads <= 0 || head_dim <= 0 || num_kv_heads > std::numeric_limits<int64_t>::max() / head_dim) {
    throw InvalidArgumentException("KV head count and dimension must form a positive, representable projection");
  }
  const auto heads = static_cast<size_t>(num_kv_heads);
  const auto world_size = rank.WorldSize();
  if (heads >= world_size) {
    if (heads % world_size != 0) {
      throw InvalidArgumentException("KV head count must be divisible by tensor parallel size when partitioned");
    }
    return weight::Shard::Uniform(0, rank.Rank(), world_size);
  }
  if (world_size % heads != 0) {
    throw InvalidArgumentException("tensor parallel size must be divisible by KV head count when replicated");
  }
  const auto head = rank.Rank() / (world_size / heads);
  return weight::Shard::Range(0, head * static_cast<size_t>(head_dim), static_cast<size_t>(head_dim));
}

auto ComputeNumKvGroups(int64_t num_kv_heads, int64_t num_attention_heads, const parallel::TpRankContext &rank)
    -> int64_t {
  static_cast<void>(ComputeKvShard(num_kv_heads, 1, rank));
  const auto world_size = static_cast<int64_t>(rank.WorldSize());
  if (num_attention_heads <= 0 || num_attention_heads % world_size != 0) {
    throw InvalidArgumentException("query head count must be positive and divisible by tensor parallel size");
  }
  const auto replication = world_size > num_kv_heads ? world_size / num_kv_heads : 1;
  return (num_attention_heads / num_kv_heads) / replication;
}

}  // namespace zephyr::layer
