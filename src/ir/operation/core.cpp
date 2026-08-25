#include "ir/operation/core.hpp"

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

namespace zephyr::ir {
namespace {

auto FloatToString(float value) -> std::string {
  auto stream = std::ostringstream{};
  stream << std::setprecision(std::numeric_limits<float>::max_digits10) << std::defaultfloat << value;
  return stream.str();
}

auto InferLinearType(const Value &input, const Parameter *weight) -> TensorType {
  const auto &input_type = GetType(input);
  const auto output_features = std::get<int64_t>(weight->type_.shape_[0]);
  auto shape = input_type.shape_;
  shape.back() = output_features;
  return TensorType{.dtype_ = input_type.dtype_, .shape_ = std::move(shape)};
}

auto MakeLinearOperands(Value input, const Parameter *weight, const Parameter *bias) -> std::vector<Value> {
  auto operands = std::vector<Value>{input, weight};
  if (bias != nullptr) {
    operands.emplace_back(bias);
  }
  return operands;
}

auto InferEmbeddingType(const Value &indices, const Parameter *weight) -> TensorType {
  const auto &index_type = GetType(indices);
  const auto hidden = std::get<int64_t>(weight->type_.shape_[1]);

  auto shape = index_type.shape_;
  shape.emplace_back(hidden);
  return TensorType{.dtype_ = weight->type_.dtype_, .shape_ = std::move(shape)};
}

auto InferRotaryTypes(const Value &query, const Value &key) -> std::vector<TensorType> {
  return {GetType(query), GetType(key)};
}

auto MakeMoeOperands(Value input, Value router_logits, const Parameter *selection_bias,
                     const std::vector<MoeExpertParameters> &experts) -> std::vector<Value> {
  auto operands = std::vector<Value>{input, router_logits};
  if (selection_bias != nullptr) {
    operands.emplace_back(selection_bias);
  }
  for (const auto &expert : experts) {
    operands.emplace_back(expert.gate_weight_);
    operands.emplace_back(expert.up_weight_);
    operands.emplace_back(expert.down_weight_);
  }
  return operands;
}

}  // namespace

Linear::Linear(Value input, const Parameter *weight, const Parameter *bias)
    : Operation(MakeLinearOperands(input, weight, bias), {InferLinearType(input, weight)}),
      weight_(weight),
      bias_(bias) {}

Embedding::Embedding(Value indices, const Parameter *weight)
    : Operation({indices, weight}, {InferEmbeddingType(indices, weight)}), weight_(weight) {}

RmsNorm::RmsNorm(Value input, const Parameter *weight, float epsilon)
    : Operation({input, weight}, {GetType(input)}), weight_(weight), epsilon_(epsilon) {}

auto RmsNorm::ToString(const OperationIndices &operation_indices) const -> std::string {
  return Format(operation_indices, " {epsilon = " + FloatToString(epsilon_) + "}");
}

RotaryEmbedding::RotaryEmbedding(Value query, Value key, Value positions, float theta, int64_t rotary_dimension,
                                 RotaryLayout layout)
    : Operation({query, key, positions}, InferRotaryTypes(query, key)),
      theta_(theta),
      rotary_dimension_(rotary_dimension),
      layout_(layout) {}

auto RotaryEmbedding::ToString(const OperationIndices &operation_indices) const -> std::string {
  auto attributes = std::string{" {theta = "} + FloatToString(theta_) +
                    ", rotary_dimension = " + std::to_string(rotary_dimension_) +
                    ", layout = " + (layout_ == RotaryLayout::INTERLEAVED ? "interleaved" : "split_half") + "}";
  return Format(operation_indices, attributes);
}

SelfAttention::SelfAttention(Value query, Value key, Value value, AttentionMaskKind mask_kind,
                             std::optional<AttentionWindow> window, float scale, std::optional<float> softcap)
    : Operation({query, key, value}, {GetType(query)}),
      mask_kind_(mask_kind),
      window_(window),
      scale_(scale),
      softcap_(softcap) {}

auto SelfAttention::ToString(const OperationIndices &operation_indices) const -> std::string {
  auto attributes = std::string{" {mask_kind = "} +
                    (mask_kind_ == AttentionMaskKind::CAUSAL ? "causal" : "bidirectional") + ", window = ";
  if (window_.has_value()) {
    attributes += "[" + std::to_string(window_->left_) + ", " + std::to_string(window_->right_) + "]";
  } else {
    attributes += "none";
  }
  attributes += ", scale = " + FloatToString(scale_) + ", softcap = ";
  attributes += softcap_.has_value() ? FloatToString(*softcap_) : "none";
  attributes += "}";
  return Format(operation_indices, attributes);
}

Moe::Moe(Value input, Value router_logits, const Parameter *selection_bias, std::vector<MoeExpertParameters> experts,
         RoutingScoreFunction score_function, int64_t top_k, RoutingWeightNormalization weight_normalization,
         float routing_scale, std::optional<ExpertGroupRouting> group_routing, GatedActivation activation)
    : Operation(MakeMoeOperands(input, router_logits, selection_bias, experts), {GetType(input)}),
      selection_bias_(selection_bias),
      experts_(std::move(experts)),
      score_function_(score_function),
      top_k_(top_k),
      weight_normalization_(weight_normalization),
      routing_scale_(routing_scale),
      group_routing_(group_routing),
      activation_(activation) {}

auto Moe::ToString(const OperationIndices &operation_indices) const -> std::string {
  auto attributes =
      std::string{" {score_function = "} + (score_function_ == RoutingScoreFunction::SIGMOID ? "sigmoid" : "softmax") +
      ", top_k = " + std::to_string(top_k_) +
      ", weight_normalization = " + (weight_normalization_ == RoutingWeightNormalization::SUM ? "sum" : "none") +
      ", routing_scale = " + FloatToString(routing_scale_) + ", group_count = ";
  if (group_routing_.has_value()) {
    attributes += std::to_string(group_routing_->group_count_) +
                  ", selected_group_count = " + std::to_string(group_routing_->selected_group_count_) +
                  ", group_score_function = " +
                  (group_routing_->score_function_ == GroupScoreFunction::TOP2_SUM ? "top2_sum" : "max");
  } else {
    attributes += "none, selected_group_count = none, group_score_function = none";
  }
  attributes += ", activation = " + std::string{activation_ == GatedActivation::GELU ? "gelu" : "silu"} + "}";
  return Format(operation_indices, attributes);
}

}  // namespace zephyr::ir
