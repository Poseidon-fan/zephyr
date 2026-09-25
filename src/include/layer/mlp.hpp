#pragma once

#include <cstdint>
#include <utility>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

#include "layer/linear.hpp"
#include "parallel/tp.hpp"
#include "weight/weight_builder.hpp"

namespace zephyr::layer {

/** Bias-free gated feed-forward layer: down(silu(gate(input)) * up(input)). */
class Mlp final {
 public:
  /** Load separate gate_proj/up_proj/down_proj weights without biases. */
  [[nodiscard]] static auto Load(ttl::ExecutionContext &context, const weight::WeightBuilder &builder,
                                 int64_t hidden_size, int64_t intermediate_size, const parallel::TpRankContext &rank)
      -> Mlp;
  /** Input and output are replicated; intermediate activations remain local to the TP rank. */
  [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor;

 private:
  Mlp(Linear gate_up, RowParallelLayer down) : gate_up_(std::move(gate_up)), down_(std::move(down)) {}

  Linear gate_up_;
  RowParallelLayer down_;
};

}  // namespace zephyr::layer
