#include "layer/mlp.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

#include <ttl/ops/composition.hpp>
#include <ttl/tensor/layout.hpp>
#include <ttl/tensor/shape.hpp>

#include "layer/activation.hpp"

namespace zephyr::layer {

auto Mlp::Load(ttl::ExecutionContext &context, const weight::WeightBuilder &builder, int64_t hidden_size,
               int64_t intermediate_size, const parallel::TpRankContext &rank) -> Mlp {
  const auto shard = weight::Shard::Uniform(0, rank.Rank(), rank.WorldSize());
  const auto shape = ttl::Shape{intermediate_size, hidden_size};
  // Shard each projection before packing so every rank owns corresponding gate/up rows.
  const std::array weights{builder.PushPrefix("gate_proj").Get(context, shape, "weight", shard),
                           builder.PushPrefix("up_proj").Get(context, shape, "weight", shard)};
  auto gate_up = Linear{ttl::Concat(context, weights, 0), std::nullopt};
  auto down =
      RowParallelLayer::Load(context, intermediate_size, hidden_size, false, rank, builder.PushPrefix("down_proj"));
  return Mlp{std::move(gate_up), std::move(down)};
}

auto Mlp::Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor {
  const auto gate_up = gate_up_.Forward(context, input);
  const auto axis = static_cast<int64_t>(gate_up.GetRank()) - 1;
  const auto split_size = gate_up.GetShape().GetDimension(static_cast<size_t>(axis)) / 2;
  const auto gate = ttl::Narrow(gate_up, axis, 0, split_size);
  const auto up = ttl::Narrow(gate_up, axis, split_size, split_size);
  return down_.Forward(context, SiluAndMul(context, gate, up));
}

}  // namespace zephyr::layer
