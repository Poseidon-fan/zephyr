#include "ir/operation/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "common/exception.h"

namespace zephyr::ir {
namespace {

auto FormatOperands(const Operation &operation, const OperationIndices &operation_indices) -> std::string {
  auto text = std::string{"("};
  for (size_t index = 0; index < operation.GetOperands().size(); index++) {
    if (index != 0) {
      text += ", ";
    }
    text += operation.GetOperands()[index]->ToString(operation_indices);
  }
  return text + ")";
}

auto FormatResultTypes(const Operation &operation) -> std::string {
  const auto result_types = operation.GetResultTypes();
  auto text = std::string{" : "};
  if (result_types.size() == 1) {
    return text + result_types[0].ToString();
  }
  text += "(";
  for (size_t index = 0; index < result_types.size(); index++) {
    if (index != 0) {
      text += ", ";
    }
    text += result_types[index].ToString();
  }
  return text + ")";
}

auto FormatResultNames(const Operation &operation, const OperationIndices &operation_indices) -> std::string {
  const auto iterator = operation_indices.find(&operation);
  if (iterator == operation_indices.end()) {
    throw ConfigurationException{"operation is missing from the model result numbering"};
  }
  auto text = std::string{};
  for (size_t index = 0; index < operation.GetResultTypes().size(); index++) {
    if (index != 0) {
      text += ", ";
    }
    text += "%" + std::to_string(iterator->second + index);
  }
  return text;
}

auto FloatToString(float value) -> std::string {
  auto stream = std::ostringstream{};
  stream << std::setprecision(std::numeric_limits<float>::max_digits10) << std::defaultfloat << value;
  return stream.str();
}

void RequireFloating(const TensorType &type, std::string_view operation) {
  if (!ttl::IsFloating(type.dtype_)) {
    throw InvalidArgumentException{std::string{operation} + " requires floating-point tensors"};
  }
}

auto RequireValue(const Value *value, std::string_view operation) -> const Value & {
  if (value == nullptr) {
    throw InvalidArgumentException{std::string{operation} + " operand must not be null"};
  }
  return *value;
}

void VerifyResultTypes(const Operation &operation, std::span<const TensorType> expected) {
  const auto actual = operation.GetResultTypes();
  if (!std::ranges::equal(actual, expected)) {
    throw InvalidArgumentException{std::string{operation.GetName()} + " has inconsistent result types"};
  }
}

auto InferLinearType(const Value &input, const Parameter *weight, const Parameter *bias) -> TensorType {
  const auto &input_type = input.GetType();
  internal::VerifyTensorType(input_type);
  if (weight == nullptr) {
    throw InvalidArgumentException{"core.linear weight must not be null"};
  }
  internal::VerifyParameterType(*weight);
  if (input_type.shape_.empty() || weight->GetType().shape_.size() != 2) {
    throw InvalidArgumentException{"core.linear requires a ranked input and rank-2 weight"};
  }
  RequireFloating(input_type, Linear::NAME);
  internal::VerifySameDType(input_type, weight->GetType(), Linear::NAME);

  const auto output_features = GetStaticExtent(weight->GetType().shape_[0], "linear output features");
  const auto input_features = GetStaticExtent(weight->GetType().shape_[1], "linear input features");
  if (GetStaticExtent(input_type.shape_.back(), "linear input features") != input_features) {
    throw InvalidArgumentException{"core.linear input and weight feature dimensions do not match"};
  }
  if (bias != nullptr) {
    internal::VerifyParameterType(*bias);
    internal::VerifySameDType(input_type, bias->GetType(), Linear::NAME);
    if (bias->GetType().shape_.size() != 1 ||
        GetStaticExtent(bias->GetType().shape_[0], "linear bias features") != output_features) {
      throw InvalidArgumentException{"core.linear bias shape does not match output features"};
    }
  }

  auto shape = input_type.shape_;
  shape.back() = output_features;
  return TensorType{.dtype_ = input_type.dtype_, .shape_ = std::move(shape)};
}

auto MakeLinearOperands(const Value *input, const Parameter *weight, const Parameter *bias)
    -> std::vector<const Value *> {
  auto operands = std::vector<const Value *>{input, weight};
  if (bias != nullptr) {
    operands.emplace_back(bias);
  }
  return operands;
}

auto InferEmbeddingType(const Value &indices, const Parameter *weight) -> TensorType {
  const auto &index_type = indices.GetType();
  internal::VerifyTensorType(index_type);
  if (!ttl::IsIntegral(index_type.dtype_)) {
    throw InvalidArgumentException{"core.embedding indices must use an integral dtype"};
  }
  if (weight == nullptr) {
    throw InvalidArgumentException{"core.embedding weight must not be null"};
  }
  internal::VerifyParameterType(*weight);
  if (weight->GetType().shape_.size() != 2) {
    throw InvalidArgumentException{"core.embedding weight must have rank 2"};
  }
  RequireFloating(weight->GetType(), Embedding::NAME);
  static_cast<void>(GetStaticExtent(weight->GetType().shape_[0], "embedding vocabulary"));
  const auto hidden = GetStaticExtent(weight->GetType().shape_[1], "embedding hidden size");

  auto shape = index_type.shape_;
  shape.emplace_back(hidden);
  return TensorType{.dtype_ = weight->GetType().dtype_, .shape_ = std::move(shape)};
}

auto InferRmsNormType(const Value &input, const Parameter *weight, float epsilon) -> TensorType {
  const auto &input_type = input.GetType();
  internal::VerifyTensorType(input_type);
  RequireFloating(input_type, RmsNorm::NAME);
  if (weight == nullptr) {
    throw InvalidArgumentException{"core.rms_norm weight must not be null"};
  }
  internal::VerifyParameterType(*weight);
  internal::VerifySameDType(input_type, weight->GetType(), RmsNorm::NAME);
  if (input_type.shape_.empty() || weight->GetType().shape_.size() != 1 ||
      GetStaticExtent(input_type.shape_.back(), "rms norm features") !=
          GetStaticExtent(weight->GetType().shape_[0], "rms norm weight features")) {
    throw InvalidArgumentException{"core.rms_norm feature dimensions do not match"};
  }
  if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
    throw InvalidArgumentException{"core.rms_norm epsilon must be finite and positive"};
  }
  return input_type;
}

auto InferRotaryTypes(const Value &query, const Value &key, const Value &positions, float theta,
                      int64_t rotary_dimension, RotaryLayout layout) -> std::vector<TensorType> {
  const auto &query_type = query.GetType();
  const auto &key_type = key.GetType();
  const auto &position_type = positions.GetType();
  internal::VerifyTensorType(query_type);
  internal::VerifyTensorType(key_type);
  internal::VerifyTensorType(position_type);
  RequireFloating(query_type, RotaryEmbedding::NAME);
  internal::VerifySameDType(query_type, key_type, RotaryEmbedding::NAME);
  if (!ttl::IsIntegral(position_type.dtype_)) {
    throw InvalidArgumentException{"core.rotary_embedding positions must use an integral dtype"};
  }
  if (query_type.shape_.size() != 3 || key_type.shape_.size() != 3 || position_type.shape_.size() != 1) {
    throw InvalidArgumentException{"core.rotary_embedding requires rank-3 query/key and rank-1 positions"};
  }
  if (query_type.shape_[0] != key_type.shape_[0] || query_type.shape_[0] != position_type.shape_[0]) {
    throw InvalidArgumentException{"core.rotary_embedding token dimensions do not match"};
  }
  static_cast<void>(GetStaticExtent(query_type.shape_[1], "rotary query head count"));
  static_cast<void>(GetStaticExtent(key_type.shape_[1], "rotary key head count"));
  const auto head_dimension = GetStaticExtent(query_type.shape_[2], "rotary head dimension");
  if (GetStaticExtent(key_type.shape_[2], "rotary key head dimension") != head_dimension) {
    throw InvalidArgumentException{"core.rotary_embedding head dimensions do not match"};
  }
  if (rotary_dimension <= 0 || rotary_dimension > head_dimension || rotary_dimension % 2 != 0) {
    throw InvalidArgumentException{"core.rotary_embedding rotary dimension is invalid"};
  }
  if (!std::isfinite(theta) || theta <= 0.0F) {
    throw InvalidArgumentException{"core.rotary_embedding theta must be finite and positive"};
  }
  if (layout != RotaryLayout::SPLIT_HALF && layout != RotaryLayout::INTERLEAVED) {
    throw InvalidArgumentException{"core.rotary_embedding layout is invalid"};
  }
  return {query_type, key_type};
}

auto InferAttentionType(const Value &query, const Value &key, const Value &value, AttentionMaskKind mask_kind,
                        const std::optional<AttentionWindow> &window, float scale, const std::optional<float> &softcap)
    -> TensorType {
  const auto &query_type = query.GetType();
  const auto &key_type = key.GetType();
  const auto &value_type = value.GetType();
  internal::VerifyTensorType(query_type);
  internal::VerifyTensorType(key_type);
  internal::VerifyTensorType(value_type);
  RequireFloating(query_type, SelfAttention::NAME);
  internal::VerifySameDType(query_type, key_type, SelfAttention::NAME);
  internal::VerifySameDType(query_type, value_type, SelfAttention::NAME);
  if (query_type.shape_.size() != 3 || key_type.shape_.size() != 3 || value_type.shape_.size() != 3) {
    throw InvalidArgumentException{"core.self_attention operands must have rank 3"};
  }
  if (query_type.shape_[0] != key_type.shape_[0] || query_type.shape_[0] != value_type.shape_[0]) {
    throw InvalidArgumentException{"core.self_attention token dimensions do not match"};
  }
  const auto query_heads = GetStaticExtent(query_type.shape_[1], "attention query head count");
  const auto key_value_heads = GetStaticExtent(key_type.shape_[1], "attention key/value head count");
  if (value_type.shape_[1] != key_type.shape_[1] || query_heads % key_value_heads != 0) {
    throw InvalidArgumentException{"core.self_attention head counts are incompatible"};
  }
  const auto head_dimension = GetStaticExtent(query_type.shape_[2], "attention head dimension");
  if (GetStaticExtent(key_type.shape_[2], "attention key head dimension") != head_dimension ||
      GetStaticExtent(value_type.shape_[2], "attention value head dimension") != head_dimension) {
    throw InvalidArgumentException{"core.self_attention head dimensions do not match"};
  }
  if (mask_kind != AttentionMaskKind::BIDIRECTIONAL && mask_kind != AttentionMaskKind::CAUSAL) {
    throw InvalidArgumentException{"core.self_attention mask kind is invalid"};
  }
  if (window.has_value() &&
      (window->left_ < 0 || window->right_ < 0 || (mask_kind == AttentionMaskKind::CAUSAL && window->right_ != 0))) {
    throw InvalidArgumentException{"core.self_attention window is invalid"};
  }
  if (!std::isfinite(scale) || scale <= 0.0F) {
    throw InvalidArgumentException{"core.self_attention scale must be finite and positive"};
  }
  if (softcap.has_value() && (!std::isfinite(*softcap) || *softcap <= 0.0F)) {
    throw InvalidArgumentException{"core.self_attention softcap must be finite and positive"};
  }
  return query_type;
}

auto MakeMoeOperands(const Value *input, const Value *router_logits, const Parameter *selection_bias,
                     const std::vector<MoeExpertParameters> &experts) -> std::vector<const Value *> {
  auto operands = std::vector<const Value *>{input, router_logits};
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

auto InferMoeType(const Value &input, const Value &router_logits, const Parameter *selection_bias,
                  const std::vector<MoeExpertParameters> &experts, RoutingScoreFunction score_function, int64_t top_k,
                  RoutingWeightNormalization weight_normalization, float routing_scale,
                  const std::optional<ExpertGroupRouting> &group_routing, GatedActivation activation) -> TensorType {
  const auto &input_type = input.GetType();
  const auto &router_type = router_logits.GetType();
  internal::VerifyTensorType(input_type);
  internal::VerifyTensorType(router_type);
  RequireFloating(input_type, Moe::NAME);
  RequireFloating(router_type, Moe::NAME);
  if (input_type.shape_.empty() || router_type.shape_.size() != input_type.shape_.size() ||
      !std::ranges::equal(input_type.shape_.begin(), input_type.shape_.end() - 1, router_type.shape_.begin(),
                          router_type.shape_.end() - 1)) {
    throw InvalidArgumentException{"core.moe input and router prefix shapes do not match"};
  }
  const auto hidden = GetStaticExtent(input_type.shape_.back(), "moe hidden size");
  const auto expert_count = GetStaticExtent(router_type.shape_.back(), "moe expert count");
  if (experts.empty() || experts.size() != static_cast<size_t>(expert_count)) {
    throw InvalidArgumentException{"core.moe expert parameters do not match router expert count"};
  }
  if (top_k <= 0 || top_k > expert_count) {
    throw InvalidArgumentException{"core.moe top_k is out of range"};
  }
  if (!std::isfinite(routing_scale) || routing_scale < 0.0F) {
    throw InvalidArgumentException{"core.moe routing scale must be finite and non-negative"};
  }
  if (score_function != RoutingScoreFunction::SOFTMAX && score_function != RoutingScoreFunction::SIGMOID) {
    throw InvalidArgumentException{"core.moe routing score function is invalid"};
  }
  if (weight_normalization != RoutingWeightNormalization::NONE &&
      weight_normalization != RoutingWeightNormalization::SUM) {
    throw InvalidArgumentException{"core.moe routing weight normalization is invalid"};
  }
  if (activation != GatedActivation::SILU && activation != GatedActivation::GELU) {
    throw InvalidArgumentException{"core.moe activation is invalid"};
  }

  if (selection_bias != nullptr) {
    internal::VerifyParameterType(*selection_bias);
    if (selection_bias->GetType().dtype_ != ttl::DType::FLOAT32 || selection_bias->GetType().shape_.size() != 1 ||
        GetStaticExtent(selection_bias->GetType().shape_[0], "moe selection bias") != expert_count) {
      throw InvalidArgumentException{"core.moe selection bias must be float32[expert_count]"};
    }
  }

  int64_t intermediate = 0;
  for (const auto &expert : experts) {
    if (expert.gate_weight_ == nullptr || expert.up_weight_ == nullptr || expert.down_weight_ == nullptr) {
      throw InvalidArgumentException{"core.moe expert parameters must not be null"};
    }
    internal::VerifyParameterType(*expert.gate_weight_);
    internal::VerifyParameterType(*expert.up_weight_);
    internal::VerifyParameterType(*expert.down_weight_);
    internal::VerifySameDType(input_type, expert.gate_weight_->GetType(), Moe::NAME);
    internal::VerifySameDType(input_type, expert.up_weight_->GetType(), Moe::NAME);
    internal::VerifySameDType(input_type, expert.down_weight_->GetType(), Moe::NAME);
    if (expert.gate_weight_->GetType().shape_.size() != 2 || expert.up_weight_->GetType().shape_.size() != 2 ||
        expert.down_weight_->GetType().shape_.size() != 2) {
      throw InvalidArgumentException{"core.moe expert weights must have rank 2"};
    }
    const auto current_intermediate =
        GetStaticExtent(expert.gate_weight_->GetType().shape_[0], "moe intermediate size");
    if (intermediate == 0) {
      intermediate = current_intermediate;
    }
    if (current_intermediate != intermediate || expert.up_weight_->GetType().shape_[0] != Dimension{intermediate} ||
        expert.gate_weight_->GetType().shape_[1] != Dimension{hidden} ||
        expert.up_weight_->GetType().shape_[1] != Dimension{hidden} ||
        expert.down_weight_->GetType().shape_[0] != Dimension{hidden} ||
        expert.down_weight_->GetType().shape_[1] != Dimension{intermediate}) {
      throw InvalidArgumentException{"core.moe expert weight shapes are inconsistent"};
    }
  }

  if (group_routing.has_value()) {
    const auto &group = *group_routing;
    if (group.group_count_ <= 0 || expert_count % group.group_count_ != 0 || group.selected_group_count_ <= 0 ||
        group.selected_group_count_ > group.group_count_) {
      throw InvalidArgumentException{"core.moe expert group configuration is invalid"};
    }
    const auto experts_per_group = expert_count / group.group_count_;
    if ((group.score_function_ != GroupScoreFunction::MAX && group.score_function_ != GroupScoreFunction::TOP2_SUM) ||
        (group.score_function_ == GroupScoreFunction::TOP2_SUM && experts_per_group < 2) ||
        group.selected_group_count_ * experts_per_group < top_k) {
      throw InvalidArgumentException{"core.moe expert group selection is invalid"};
    }
  }
  return input_type;
}

}  // namespace

Linear::Linear(const Value *input, const Parameter *weight, const Parameter *bias)
    : Operation(MakeLinearOperands(input, weight, bias), {InferLinearType(RequireValue(input, NAME), weight, bias)}),
      weight_(weight),
      bias_(bias) {}

void Linear::Verify() const {
  const auto expected = InferLinearType(*GetInput(), weight_, bias_);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto Linear::ToString(const OperationIndices &operation_indices) const -> std::string {
  return FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
         FormatOperands(*this, operation_indices) + FormatResultTypes(*this);
}

Embedding::Embedding(const Value *indices, const Parameter *weight)
    : Operation({indices, weight}, {InferEmbeddingType(RequireValue(indices, NAME), weight)}), weight_(weight) {}

void Embedding::Verify() const {
  const auto expected = InferEmbeddingType(*GetIndices(), weight_);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto Embedding::ToString(const OperationIndices &operation_indices) const -> std::string {
  return FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
         FormatOperands(*this, operation_indices) + FormatResultTypes(*this);
}

RmsNorm::RmsNorm(const Value *input, const Parameter *weight, float epsilon)
    : Operation({input, weight}, {InferRmsNormType(RequireValue(input, NAME), weight, epsilon)}),
      weight_(weight),
      epsilon_(epsilon) {}

void RmsNorm::Verify() const {
  const auto expected = InferRmsNormType(*GetInput(), weight_, epsilon_);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto RmsNorm::ToString(const OperationIndices &operation_indices) const -> std::string {
  return FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
         FormatOperands(*this, operation_indices) + " {epsilon = " + FloatToString(epsilon_) + "}" +
         FormatResultTypes(*this);
}

RotaryEmbedding::RotaryEmbedding(const Value *query, const Value *key, const Value *positions, float theta,
                                 int64_t rotary_dimension, RotaryLayout layout)
    : Operation({query, key, positions},
                InferRotaryTypes(RequireValue(query, NAME), RequireValue(key, NAME), RequireValue(positions, NAME),
                                 theta, rotary_dimension, layout)),
      theta_(theta),
      rotary_dimension_(rotary_dimension),
      layout_(layout) {}

void RotaryEmbedding::Verify() const {
  const auto operands = GetOperands();
  const auto expected = InferRotaryTypes(*operands[0], *operands[1], *operands[2], theta_, rotary_dimension_, layout_);
  VerifyResultTypes(*this, expected);
}

auto RotaryEmbedding::ToString(const OperationIndices &operation_indices) const -> std::string {
  return FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
         FormatOperands(*this, operation_indices) + " {theta = " + FloatToString(theta_) +
         ", rotary_dimension = " + std::to_string(rotary_dimension_) +
         ", layout = " + (layout_ == RotaryLayout::INTERLEAVED ? "interleaved" : "split_half") + "}" +
         FormatResultTypes(*this);
}

SelfAttention::SelfAttention(const Value *query, const Value *key, const Value *value, AttentionMaskKind mask_kind,
                             std::optional<AttentionWindow> window, float scale, std::optional<float> softcap)
    : Operation({query, key, value},
                {InferAttentionType(RequireValue(query, NAME), RequireValue(key, NAME), RequireValue(value, NAME),
                                    mask_kind, window, scale, softcap)}),
      mask_kind_(mask_kind),
      window_(window),
      scale_(scale),
      softcap_(softcap) {}

void SelfAttention::Verify() const {
  const auto operands = GetOperands();
  const auto expected =
      InferAttentionType(*operands[0], *operands[1], *operands[2], mask_kind_, window_, scale_, softcap_);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto SelfAttention::ToString(const OperationIndices &operation_indices) const -> std::string {
  auto text = FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
              FormatOperands(*this, operation_indices) +
              " {mask_kind = " + (mask_kind_ == AttentionMaskKind::CAUSAL ? "causal" : "bidirectional") + ", window = ";
  if (window_.has_value()) {
    text += "[" + std::to_string(window_->left_) + ", " + std::to_string(window_->right_) + "]";
  } else {
    text += "none";
  }
  text += ", scale = " + FloatToString(scale_) + ", softcap = ";
  text += softcap_.has_value() ? FloatToString(*softcap_) : "none";
  return text + "}" + FormatResultTypes(*this);
}

Moe::Moe(const Value *input, const Value *router_logits, const Parameter *selection_bias,
         std::vector<MoeExpertParameters> experts, RoutingScoreFunction score_function, int64_t top_k,
         RoutingWeightNormalization weight_normalization, float routing_scale,
         std::optional<ExpertGroupRouting> group_routing, GatedActivation activation)
    : Operation(MakeMoeOperands(input, router_logits, selection_bias, experts),
                {InferMoeType(RequireValue(input, NAME), RequireValue(router_logits, NAME), selection_bias, experts,
                              score_function, top_k, weight_normalization, routing_scale, group_routing, activation)}),
      selection_bias_(selection_bias),
      experts_(std::move(experts)),
      score_function_(score_function),
      top_k_(top_k),
      weight_normalization_(weight_normalization),
      routing_scale_(routing_scale),
      group_routing_(group_routing),
      activation_(activation) {}

void Moe::Verify() const {
  const auto operands = GetOperands();
  const auto expected = InferMoeType(*operands[0], *operands[1], selection_bias_, experts_, score_function_, top_k_,
                                     weight_normalization_, routing_scale_, group_routing_, activation_);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto Moe::ToString(const OperationIndices &operation_indices) const -> std::string {
  auto text = FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
              FormatOperands(*this, operation_indices) +
              " {score_function = " + (score_function_ == RoutingScoreFunction::SIGMOID ? "sigmoid" : "softmax") +
              ", top_k = " + std::to_string(top_k_) + ", weight_normalization = " +
              (weight_normalization_ == RoutingWeightNormalization::SUM ? "sum" : "none") +
              ", routing_scale = " + FloatToString(routing_scale_) + ", group_count = ";
  if (group_routing_.has_value()) {
    text += std::to_string(group_routing_->group_count_) +
            ", selected_group_count = " + std::to_string(group_routing_->selected_group_count_) +
            ", group_score_function = " +
            (group_routing_->score_function_ == GroupScoreFunction::TOP2_SUM ? "top2_sum" : "max");
  } else {
    text += "none, selected_group_count = none, group_score_function = none";
  }
  return text + ", activation = " + (activation_ == GatedActivation::GELU ? "gelu" : "silu") + "}" +
         FormatResultTypes(*this);
}

}  // namespace zephyr::ir
