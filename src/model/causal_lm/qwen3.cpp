#include "model/causal_lm/qwen3.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include <ttl/ops/creation.hpp>
#include <ttl/ops/elementwise.hpp>
#include <ttl/tensor/layout.hpp>

#include "common/exception.hpp"

namespace zephyr::model::causal_lm {

Qwen3Model::Attention::Attention(ttl::ExecutionContext &context, const Qwen3Config &config,
                                 const weight::WeightBuilder &builder, const parallel::TpRankContext &rank)
    : q_proj_(layer::ColumnParallelLayer::Load(context, config.hidden_size_,
                                               config.num_attention_heads_ * config.head_dim_, false, rank,
                                               builder.PushPrefix("q_proj"))),
      k_proj_(layer::ColumnParallelLayer::LoadWithShard(
          context, config.hidden_size_, config.num_key_value_heads_ * config.head_dim_, false, rank,
          layer::ComputeKvShard(config.num_key_value_heads_, config.head_dim_, rank), builder.PushPrefix("k_proj"))),
      v_proj_(layer::ColumnParallelLayer::LoadWithShard(
          context, config.hidden_size_, config.num_key_value_heads_ * config.head_dim_, false, rank,
          layer::ComputeKvShard(config.num_key_value_heads_, config.head_dim_, rank), builder.PushPrefix("v_proj"))),
      o_proj_(layer::RowParallelLayer::Load(context, config.num_attention_heads_ * config.head_dim_,
                                            config.hidden_size_, false, rank, builder.PushPrefix("o_proj"))),
      q_norm_(context, config.head_dim_, config.rms_norm_eps_, builder.PushPrefix("q_norm")),
      k_norm_(context, config.head_dim_, config.rms_norm_eps_, builder.PushPrefix("k_norm")),
      num_heads_(config.num_attention_heads_ / static_cast<int64_t>(rank.WorldSize())),
      num_kv_heads_(std::max(config.num_key_value_heads_ / static_cast<int64_t>(rank.WorldSize()), int64_t{1})),
      head_dim_(config.head_dim_),
      sdpa_params_{
          .n_kv_groups_ = layer::ComputeNumKvGroups(config.num_key_value_heads_, config.num_attention_heads_, rank),
          .softmax_scale_ = 1.0F / std::sqrt(static_cast<float>(config.head_dim_))} {}

auto Qwen3Model::Attention::Forward(ttl::ExecutionContext &context, const ttl::Tensor &input,
                                    const attention::AttentionMask &mask, const layer::RotaryEmbedding &rotary,
                                    const ModelForwardContext &forward_context, size_t layer_index) const
    -> ttl::Tensor {
  const auto batch = input.GetShape().GetDimension(0);
  const auto sequence = input.GetShape().GetDimension(1);
  auto query = ttl::Transpose(
      ttl::Reshape(context, q_proj_.Forward(context, input), ttl::Shape{batch, sequence, num_heads_, head_dim_}), 1, 2);
  auto key = ttl::Transpose(
      ttl::Reshape(context, k_proj_.Forward(context, input), ttl::Shape{batch, sequence, num_kv_heads_, head_dim_}), 1,
      2);
  const auto value = ttl::Transpose(
      ttl::Reshape(context, v_proj_.Forward(context, input), ttl::Shape{batch, sequence, num_kv_heads_, head_dim_}), 1,
      2);
  // Normalize each head before rotation. Both operations preserve the model's activation dtype.
  query = q_norm_.Forward(context, query);
  key = k_norm_.Forward(context, key);
  auto [rotated_query, rotated_key] = rotary.Forward(context, query, key, forward_context.positions_);
  const auto *cache = forward_context.cache_ == nullptr ? nullptr : &forward_context.cache_->GetLayerCache(layer_index);
  auto output =
      paged_attention_.Forward(context, rotated_query, rotated_key, value, mask,
                               cache == nullptr ? std::nullopt : std::optional{cache->key_cache_},
                               cache == nullptr ? std::nullopt : std::optional{cache->value_cache_},
                               forward_context.paged_attention_, sdpa_params_, &forward_context.flash_params_);
  // Prompt attention is head-major. Unmasked cache paths already place token rows in flattening order.
  if (!mask.IsNone()) {
    output = ttl::Transpose(output, 1, 2);
  }
  output = ttl::Reshape(context, output, ttl::Shape{batch, sequence, num_heads_ * head_dim_});
  return o_proj_.Forward(context, output);
}

Qwen3Model::DecoderLayer::DecoderLayer(ttl::ExecutionContext &context, const Qwen3Config &config,
                                       const weight::WeightBuilder &builder, const parallel::TpRankContext &rank)
    : self_attn_(context, config, builder.PushPrefix("self_attn"), rank),
      mlp_(layer::Mlp::Load(context, builder.PushPrefix("mlp"), config.hidden_size_, config.intermediate_size_, rank)),
      input_layernorm_(context, config.hidden_size_, config.rms_norm_eps_, builder.PushPrefix("input_layernorm")),
      post_attention_layernorm_(context, config.hidden_size_, config.rms_norm_eps_,
                                builder.PushPrefix("post_attention_layernorm")) {}

auto Qwen3Model::DecoderLayer::Forward(ttl::ExecutionContext &context, const ttl::Tensor &input,
                                       const attention::AttentionMask &mask, const layer::RotaryEmbedding &rotary,
                                       const ModelForwardContext &forward_context, size_t layer_index) const
    -> ttl::Tensor {
  const auto attention_output =
      self_attn_.Forward(context, input_layernorm_.Forward(context, input), mask, rotary, forward_context, layer_index);
  const auto residual = ttl::Add(context, attention_output, input);
  const auto mlp_output = mlp_.Forward(context, post_attention_layernorm_.Forward(context, residual));
  return ttl::Add(context, residual, mlp_output);
}

Qwen3Model::Qwen3Model(ModelSpec spec, layer::Embedding embedding, layer::RotaryEmbedding rotary,
                       std::vector<DecoderLayer> layers, layer::RmsNorm norm, layer::Linear lm_head)
    : spec_(std::move(spec)),
      embed_tokens_(std::move(embedding)),
      rotary_(std::move(rotary)),
      layers_(std::move(layers)),
      norm_(std::move(norm)),
      lm_head_(std::move(lm_head)) {}

auto Qwen3Model::Load(ttl::ExecutionContext &context, const Qwen3Config &config, const weight::WeightBuilder &builder,
                      const parallel::TpRankContext &rank) -> Qwen3Model {
  // Validate architecture once, before loading weights or constructing layer-local dimensions.
  if (config.vocab_size_ <= 0 || config.hidden_size_ <= 0 || config.intermediate_size_ <= 0 ||
      config.num_hidden_layers_ <= 0 || config.num_attention_heads_ <= 0 || config.num_key_value_heads_ <= 0 ||
      config.head_dim_ <= 0 || config.head_dim_ % 2 != 0 || config.max_position_embeddings_ <= 0 ||
      config.num_attention_heads_ % config.num_key_value_heads_ != 0 ||
      config.head_dim_ > std::numeric_limits<int64_t>::max() / config.num_attention_heads_ ||
      !std::isfinite(static_cast<float>(config.rms_norm_eps_)) || config.rms_norm_eps_ <= 0.0 ||
      !std::isfinite(static_cast<float>(config.rope_theta_)) || config.rope_theta_ <= 0.0) {
    throw ConfigurationException(
        "Qwen3 requires positive dimensions, divisible Q/KV heads, an even head width, "
        "and finite normalization/rotary parameters");
  }
  if (context.GetDevice() != rank.Device()) {
    throw InvalidArgumentException("model execution context must belong to its tensor-parallel rank");
  }
  static_cast<void>(layer::ComputeNumKvGroups(config.num_key_value_heads_, config.num_attention_heads_, rank));
  if (config.intermediate_size_ % static_cast<int64_t>(rank.WorldSize()) != 0) {
    throw ConfigurationException("Qwen3 intermediate size must be divisible by tensor-parallel size");
  }

  const auto model_builder = builder.PushPrefix("model");
  auto embedding = layer::Embedding::Load(context, config.vocab_size_, config.hidden_size_,
                                          model_builder.PushPrefix("embed_tokens"));
  const auto dtype = embedding.GetWeight().GetDType();
  auto rotary = layer::RotaryEmbedding{
      context, static_cast<float>(config.rope_theta_), config.head_dim_, config.max_position_embeddings_, true, dtype};
  std::vector<DecoderLayer> layers;
  layers.reserve(static_cast<size_t>(config.num_hidden_layers_));
  for (int64_t index = 0; index < config.num_hidden_layers_; ++index) {
    layers.emplace_back(context, config, model_builder.PushPrefix("layers").PushPrefix(static_cast<size_t>(index)),
                        rank);
  }
  auto norm = layer::RmsNorm{context, config.hidden_size_, config.rms_norm_eps_, model_builder.PushPrefix("norm")};
  auto lm_head = config.tie_word_embeddings_ ? layer::Linear{embedding.GetWeight(), std::nullopt}
                                             : layer::Linear::Load(context, config.hidden_size_, config.vocab_size_,
                                                                   false, builder.PushPrefix("lm_head"));
  const auto local_kv_heads =
      std::max(config.num_key_value_heads_ / static_cast<int64_t>(rank.WorldSize()), int64_t{1});
  ModelSpec spec{.max_seq_len_ = config.max_position_embeddings_,
                 .vocab_size_ = config.vocab_size_,
                 .device_ = context.GetDevice(),
                 .dtype_ = dtype,
                 .layer_specs_ = std::vector<kv_cache::LayerCacheSpec>(
                     layers.size(), {.num_kv_heads_ = static_cast<size_t>(local_kv_heads),
                                     .key_head_dim_ = static_cast<size_t>(config.head_dim_),
                                     .value_head_dim_ = static_cast<size_t>(config.head_dim_)}),
                 .supports_packed_prefill_ = true};
  return Qwen3Model{std::move(spec),   std::move(embedding), std::move(rotary),
                    std::move(layers), std::move(norm),      std::move(lm_head)};
}

auto Qwen3Model::Forward(ttl::ExecutionContext &context, const ttl::Tensor &input_ids,
                         const ModelForwardContext &forward_context) const -> ttl::Tensor {
  if (input_ids.GetRank() != 2 || input_ids.GetShape().GetDimension(0) <= 0 ||
      input_ids.GetShape().GetDimension(1) <= 0) {
    throw InvalidArgumentException("Qwen3 input IDs must be a nonempty [batch, sequence] tensor");
  }
  if (context.GetDevice() != spec_.device_ ||
      (forward_context.cache_ != nullptr && forward_context.cache_->GetNumLayers() != layers_.size())) {
    throw InvalidArgumentException("forward context must match the model device and layer cache count");
  }
  const auto sequence = input_ids.GetShape().GetDimension(1);
  const auto first_prompt = forward_context.paged_attention_.is_first_prompt_chunk_;
  const auto causal_prompt = first_prompt && (sequence > 1 || forward_context.flash_params_.packed_);
  if (causal_prompt && !forward_context.flash_params_.causal_) {
    throw InvalidArgumentException("Qwen3 prompt attention requires causal flash metadata");
  }
  const auto mask = causal_prompt ? attention::AttentionMask::CausalFlash() : attention::AttentionMask::None();
  auto hidden = embed_tokens_.Forward(context, input_ids);
  for (size_t index = 0; index < layers_.size(); ++index) {
    hidden = layers_[index].Forward(context, hidden, mask, rotary_, forward_context, index);
  }
  hidden = norm_.Forward(context, hidden);
  auto selected = forward_context.SelectLogits(context, hidden);
  if (selected.GetShape().GetDimension(1) == 0) {
    // Intermediate prompt chunks still compute every layer and write KV, but need no vocabulary projection.
    return ttl::Empty(context, ttl::Shape{selected.GetShape().GetDimension(0), 0, spec_.vocab_size_},
                      selected.GetDType());
  }
  return lm_head_.Forward(context, selected);
}

}  // namespace zephyr::model::causal_lm
