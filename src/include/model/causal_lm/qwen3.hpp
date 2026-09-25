#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "layer/embedding.hpp"
#include "layer/linear.hpp"
#include "layer/mlp.hpp"
#include "layer/normalization.hpp"
#include "layer/rotary_embedding.hpp"
#include "model/causal_lm/model.hpp"
#include "parallel/tp.hpp"
#include "weight/weight_builder.hpp"

namespace zephyr::model::causal_lm {

/** Dense Qwen3 with SiLU, unbiased projections, full attention, and standard rotary embeddings. */
struct Qwen3Config final {
  int64_t vocab_size_{0};
  int64_t hidden_size_{0};
  int64_t intermediate_size_{0};
  int64_t num_hidden_layers_{0};
  int64_t num_attention_heads_{0};
  int64_t num_key_value_heads_{0};
  /** The resolved head width; Load derives it from hidden_size / heads only when absent in JSON. */
  int64_t head_dim_{0};
  int64_t max_position_embeddings_{0};
  double rms_norm_eps_{1e-6};
  double rope_theta_{1000000.0};
  bool tie_word_embeddings_{false};

  /** Read a config.json file and reject features outside the supported model architecture. */
  [[nodiscard]] static auto Load(const std::filesystem::path &config_path) -> Qwen3Config;
};

class Qwen3Model final : public CausalLM {
 public:
  /** Load this rank's projection shards and replicated embedding/head weights. The TP context must outlive the model.
   */
  [[nodiscard]] static auto Load(ttl::ExecutionContext &context, const Qwen3Config &config,
                                 const weight::WeightBuilder &builder, const parallel::TpRankContext &rank)
      -> Qwen3Model;

  Qwen3Model(const Qwen3Model &) = delete;
  auto operator=(const Qwen3Model &) -> Qwen3Model & = delete;
  Qwen3Model(Qwen3Model &&) noexcept = default;
  auto operator=(Qwen3Model &&) noexcept -> Qwen3Model & = default;
  ~Qwen3Model() override = default;

  [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input_ids,
                             const ModelForwardContext &forward_context) const -> ttl::Tensor override;
  [[nodiscard]] auto GetSpec() const noexcept -> const ModelSpec & override { return spec_; }

 private:
  /** Q/K normalization and rotary encoding precede attention; the output projection sums TP rank contributions. */
  struct Attention final {
    Attention(ttl::ExecutionContext &context, const Qwen3Config &config, const weight::WeightBuilder &builder,
              const parallel::TpRankContext &rank);

    [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input,
                               const attention::AttentionMask &mask, const layer::RotaryEmbedding &rotary,
                               const ModelForwardContext &forward_context, size_t layer_index) const -> ttl::Tensor;

    layer::ColumnParallelLayer q_proj_;
    layer::ColumnParallelLayer k_proj_;
    layer::ColumnParallelLayer v_proj_;
    layer::RowParallelLayer o_proj_;
    layer::RmsNorm q_norm_;
    layer::RmsNorm k_norm_;
    int64_t num_heads_;
    int64_t num_kv_heads_;
    int64_t head_dim_;
    attention::SdpaParams sdpa_params_;
    layer::PagedAttention paged_attention_;
  };

  struct DecoderLayer final {
    DecoderLayer(ttl::ExecutionContext &context, const Qwen3Config &config, const weight::WeightBuilder &builder,
                 const parallel::TpRankContext &rank);

    [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input,
                               const attention::AttentionMask &mask, const layer::RotaryEmbedding &rotary,
                               const ModelForwardContext &forward_context, size_t layer_index) const -> ttl::Tensor;

    Attention self_attn_;
    layer::Mlp mlp_;
    layer::RmsNorm input_layernorm_;
    layer::RmsNorm post_attention_layernorm_;
  };

  Qwen3Model(ModelSpec spec, layer::Embedding embedding, layer::RotaryEmbedding rotary,
             std::vector<DecoderLayer> layers, layer::RmsNorm norm, layer::Linear lm_head);

  ModelSpec spec_;
  layer::Embedding embed_tokens_;
  /** All decoder layers on this rank use the same rotary tables. */
  layer::RotaryEmbedding rotary_;
  std::vector<DecoderLayer> layers_;
  layer::RmsNorm norm_;
  layer::Linear lm_head_;
};

}  // namespace zephyr::model::causal_lm
