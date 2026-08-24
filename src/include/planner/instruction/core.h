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

/** Looks up embedding rows from a vocabulary buffer. */
class Embedding final : public CloneableInstruction<Embedding> {
 public:
  Embedding(BufferView indices, BufferView weight, BufferView output)
      : indices_(std::move(indices)), weight_(std::move(weight)), output_(std::move(output)) {}

  BufferView indices_;
  BufferView weight_;
  BufferView output_;
};

/** Computes one or more projections sharing the same input rows. */
class Linear final : public CloneableInstruction<Linear> {
 public:
  Linear(BufferView input, std::vector<BufferView> weights, std::vector<std::optional<BufferView>> biases,
         std::vector<BufferView> outputs)
      : input_(std::move(input)),
        weights_(std::move(weights)),
        biases_(std::move(biases)),
        outputs_(std::move(outputs)) {}

  BufferView input_;
  std::vector<BufferView> weights_;
  std::vector<std::optional<BufferView>> biases_;
  std::vector<BufferView> outputs_;
};

/** Applies RMS normalization to the last dimension. */
class RmsNorm final : public CloneableInstruction<RmsNorm> {
 public:
  RmsNorm(BufferView input, BufferView weight, BufferView output, float epsilon)
      : input_(std::move(input)), weight_(std::move(weight)), output_(std::move(output)), epsilon_(epsilon) {}

  BufferView input_;
  BufferView weight_;
  BufferView output_;
  float epsilon_;
};

/** Applies rotary position encoding to query and key tensors. */
class RotaryEmbedding final : public CloneableInstruction<RotaryEmbedding> {
 public:
  RotaryEmbedding(BufferView query, BufferView key, BufferView positions, BufferView rotated_query,
                  BufferView rotated_key, float theta, int32_t rotary_dimension, RotaryLayout layout)
      : query_(std::move(query)),
        key_(std::move(key)),
        positions_(std::move(positions)),
        rotated_query_(std::move(rotated_query)),
        rotated_key_(std::move(rotated_key)),
        theta_(theta),
        rotary_dimension_(rotary_dimension),
        layout_(layout) {}

  BufferView query_;
  BufferView key_;
  BufferView positions_;
  BufferView rotated_query_;
  BufferView rotated_key_;
  float theta_;
  int32_t rotary_dimension_;
  RotaryLayout layout_;
};

/** Computes scaled self-attention, optionally backed by a persistent KV entry. */
class SelfAttention final : public CloneableInstruction<SelfAttention> {
 public:
  SelfAttention(BufferView query, BufferView key, BufferView value, BufferView output, int32_t query_head_count,
                int32_t kv_head_count, int32_t head_dimension, AttentionMaskKind mask_kind,
                std::optional<AttentionWindow> window, float scale, std::optional<float> softcap,
                std::optional<kv_layer_id_t> kv_layer_id)
      : query_(std::move(query)),
        key_(std::move(key)),
        value_(std::move(value)),
        output_(std::move(output)),
        query_head_count_(query_head_count),
        kv_head_count_(kv_head_count),
        head_dimension_(head_dimension),
        mask_kind_(mask_kind),
        window_(window),
        scale_(scale),
        softcap_(softcap),
        kv_layer_id_(kv_layer_id) {}

  BufferView query_;
  BufferView key_;
  BufferView value_;
  BufferView output_;
  int32_t query_head_count_;
  int32_t kv_head_count_;
  int32_t head_dimension_;
  AttentionMaskKind mask_kind_;
  std::optional<AttentionWindow> window_;
  float scale_;
  std::optional<float> softcap_;
  std::optional<kv_layer_id_t> kv_layer_id_;
};

}  // namespace zephyr::planner
