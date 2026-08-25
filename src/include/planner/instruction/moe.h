#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "common/operator_attributes.h"
#include "common/types.h"
#include "planner/buffer.h"
#include "planner/instruction.h"

namespace zephyr::planner {

/** The three weight views consumed by one local gated expert. */
struct ExpertWeightViews final {
  BufferView gate_;
  BufferView up_;
  BufferView down_;
};

/** Executes a complete routed gated MoE on one Worker. */
class Moe final : public CloneableInstruction<Moe> {
 public:
  Moe(BufferView input, BufferView router_logits, std::optional<BufferView> selection_bias,
      std::vector<ExpertWeightViews> experts, BufferView output, RoutingScoreFunction score_function, int64_t top_k,
      RoutingWeightNormalization weight_normalization, float routing_scale,
      std::optional<ExpertGroupRouting> group_routing, GatedActivation activation)
      : input_(std::move(input)),
        router_logits_(std::move(router_logits)),
        selection_bias_(std::move(selection_bias)),
        experts_(std::move(experts)),
        output_(std::move(output)),
        score_function_(score_function),
        top_k_(top_k),
        weight_normalization_(weight_normalization),
        routing_scale_(routing_scale),
        group_routing_(group_routing),
        activation_(activation) {}

  BufferView input_;
  BufferView router_logits_;
  std::optional<BufferView> selection_bias_;
  std::vector<ExpertWeightViews> experts_;
  BufferView output_;
  RoutingScoreFunction score_function_;
  int64_t top_k_;
  RoutingWeightNormalization weight_normalization_;
  float routing_scale_;
  std::optional<ExpertGroupRouting> group_routing_;
  GatedActivation activation_;
};

/** Computes global expert indices and routing weights. */
class MoeRouting final : public CloneableInstruction<MoeRouting> {
 public:
  MoeRouting(BufferView router_logits, std::optional<BufferView> selection_bias, BufferView expert_indices,
             BufferView expert_weights, RoutingScoreFunction score_function, int64_t top_k,
             RoutingWeightNormalization weight_normalization, float routing_scale,
             std::optional<ExpertGroupRouting> group_routing)
      : router_logits_(std::move(router_logits)),
        selection_bias_(std::move(selection_bias)),
        expert_indices_(std::move(expert_indices)),
        expert_weights_(std::move(expert_weights)),
        score_function_(score_function),
        top_k_(top_k),
        weight_normalization_(weight_normalization),
        routing_scale_(routing_scale),
        group_routing_(group_routing) {}

  BufferView router_logits_;
  std::optional<BufferView> selection_bias_;
  BufferView expert_indices_;
  BufferView expert_weights_;
  RoutingScoreFunction score_function_;
  int64_t top_k_;
  RoutingWeightNormalization weight_normalization_;
  float routing_scale_;
  std::optional<ExpertGroupRouting> group_routing_;
};

/** Packs locally owned token assignments for an expert-parallel exchange. */
class MoeDispatch final : public CloneableInstruction<MoeDispatch> {
 public:
  MoeDispatch(BufferView input, BufferView expert_indices, BufferView expert_weights, std::vector<rank_t> expert_ranks,
              std::vector<int32_t> expert_to_local, BufferView packed_inputs, BufferView packed_weights,
              BufferView packed_local_experts, BufferView packed_token_indices, BufferView send_counts)
      : input_(std::move(input)),
        expert_indices_(std::move(expert_indices)),
        expert_weights_(std::move(expert_weights)),
        expert_ranks_(std::move(expert_ranks)),
        expert_to_local_(std::move(expert_to_local)),
        packed_inputs_(std::move(packed_inputs)),
        packed_weights_(std::move(packed_weights)),
        packed_local_experts_(std::move(packed_local_experts)),
        packed_token_indices_(std::move(packed_token_indices)),
        send_counts_(std::move(send_counts)) {}

  BufferView input_;
  BufferView expert_indices_;
  BufferView expert_weights_;
  std::vector<rank_t> expert_ranks_;
  std::vector<int32_t> expert_to_local_;
  BufferView packed_inputs_;
  BufferView packed_weights_;
  BufferView packed_local_experts_;
  BufferView packed_token_indices_;
  BufferView send_counts_;
};

/** Executes the experts owned by one expert-parallel rank. */
class MoeExperts final : public CloneableInstruction<MoeExperts> {
 public:
  MoeExperts(BufferView packed_inputs, BufferView packed_local_experts, std::vector<ExpertWeightViews> local_experts,
             BufferView packed_outputs, GatedActivation activation)
      : packed_inputs_(std::move(packed_inputs)),
        packed_local_experts_(std::move(packed_local_experts)),
        local_experts_(std::move(local_experts)),
        packed_outputs_(std::move(packed_outputs)),
        activation_(activation) {}

  BufferView packed_inputs_;
  BufferView packed_local_experts_;
  std::vector<ExpertWeightViews> local_experts_;
  BufferView packed_outputs_;
  GatedActivation activation_;
};

/** Combines returned expert assignments into the token rows owned by this rank. */
class MoeCombine final : public CloneableInstruction<MoeCombine> {
 public:
  MoeCombine(BufferView expert_outputs, BufferView packed_weights, BufferView packed_token_indices, BufferView output)
      : expert_outputs_(std::move(expert_outputs)),
        packed_weights_(std::move(packed_weights)),
        packed_token_indices_(std::move(packed_token_indices)),
        output_(std::move(output)) {}

  BufferView expert_outputs_;
  BufferView packed_weights_;
  BufferView packed_token_indices_;
  BufferView output_;
};

}  // namespace zephyr::planner
