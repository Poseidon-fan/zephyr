#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "common/operator_attributes.h"
#include "ir/value.h"

namespace zephyr::ir {

/** Matrix projection with an optional bias. */
class Linear final : public Operation {
 public:
  static constexpr std::string_view NAME = "core.linear";

  Linear(const Value *input, const Parameter *weight, const Parameter *bias = nullptr);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetInput() const -> const Value * { return GetOperands()[0]; }
  [[nodiscard]] auto GetWeight() const -> const Parameter * { return weight_; }
  [[nodiscard]] auto GetBias() const -> const Parameter * { return bias_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;

 private:
  const Parameter *weight_;
  const Parameter *bias_;
};

/** Looks up rows from a vocabulary parameter. */
class Embedding final : public Operation {
 public:
  static constexpr std::string_view NAME = "core.embedding";

  Embedding(const Value *indices, const Parameter *weight);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetIndices() const -> const Value * { return GetOperands()[0]; }
  [[nodiscard]] auto GetWeight() const -> const Parameter * { return weight_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;

 private:
  const Parameter *weight_;
};

/** Root-mean-square normalization over the last dimension. */
class RmsNorm final : public Operation {
 public:
  static constexpr std::string_view NAME = "core.rms_norm";

  RmsNorm(const Value *input, const Parameter *weight, float epsilon);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetInput() const -> const Value * { return GetOperands()[0]; }
  [[nodiscard]] auto GetWeight() const -> const Parameter * { return weight_; }
  [[nodiscard]] auto GetEpsilon() const -> float { return epsilon_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;

 private:
  const Parameter *weight_;
  float epsilon_;
};

/** Applies rotary position encoding to query and key tensors. */
class RotaryEmbedding final : public Operation {
 public:
  static constexpr std::string_view NAME = "core.rotary_embedding";

  RotaryEmbedding(const Value *query, const Value *key, const Value *positions, float theta, int64_t rotary_dimension,
                  RotaryLayout layout);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetTheta() const -> float { return theta_; }
  [[nodiscard]] auto GetRotaryDimension() const -> int64_t { return rotary_dimension_; }
  [[nodiscard]] auto GetLayout() const -> RotaryLayout { return layout_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;

 private:
  float theta_;
  int64_t rotary_dimension_;
  RotaryLayout layout_;
};

/** Computes scaled self-attention over packed query, key, and value tensors. */
class SelfAttention final : public Operation {
 public:
  static constexpr std::string_view NAME = "core.self_attention";

  SelfAttention(const Value *query, const Value *key, const Value *value, AttentionMaskKind mask_kind,
                std::optional<AttentionWindow> window, float scale, std::optional<float> softcap);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetMaskKind() const -> AttentionMaskKind { return mask_kind_; }
  [[nodiscard]] auto GetWindow() const -> const std::optional<AttentionWindow> & { return window_; }
  [[nodiscard]] auto GetScale() const -> float { return scale_; }
  [[nodiscard]] auto GetSoftcap() const -> const std::optional<float> & { return softcap_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;

 private:
  AttentionMaskKind mask_kind_;
  std::optional<AttentionWindow> window_;
  float scale_;
  std::optional<float> softcap_;
};

/** The three static matrices used by one gated routed expert. */
struct MoeExpertParameters final {
  const Parameter *gate_weight_;
  const Parameter *up_weight_;
  const Parameter *down_weight_;
};

/** Routed gated mixture-of-experts computation. */
class Moe final : public Operation {
 public:
  static constexpr std::string_view NAME = "core.moe";

  Moe(const Value *input, const Value *router_logits, const Parameter *selection_bias,
      std::vector<MoeExpertParameters> experts, RoutingScoreFunction score_function, int64_t top_k,
      RoutingWeightNormalization weight_normalization, float routing_scale,
      std::optional<ExpertGroupRouting> group_routing, GatedActivation activation);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetSelectionBias() const -> const Parameter * { return selection_bias_; }
  [[nodiscard]] auto GetExperts() const -> std::span<const MoeExpertParameters> { return experts_; }
  [[nodiscard]] auto GetScoreFunction() const -> RoutingScoreFunction { return score_function_; }
  [[nodiscard]] auto GetTopK() const -> int64_t { return top_k_; }
  [[nodiscard]] auto GetWeightNormalization() const -> RoutingWeightNormalization { return weight_normalization_; }
  [[nodiscard]] auto GetRoutingScale() const -> float { return routing_scale_; }
  [[nodiscard]] auto GetGroupRouting() const -> const std::optional<ExpertGroupRouting> & { return group_routing_; }
  [[nodiscard]] auto GetActivation() const -> GatedActivation { return activation_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;

 private:
  const Parameter *selection_bias_;
  std::vector<MoeExpertParameters> experts_;
  RoutingScoreFunction score_function_;
  int64_t top_k_;
  RoutingWeightNormalization weight_normalization_;
  float routing_scale_;
  std::optional<ExpertGroupRouting> group_routing_;
  GatedActivation activation_;
};

}  // namespace zephyr::ir
