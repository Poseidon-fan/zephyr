#pragma once

#include <cstdint>

namespace zephyr {

/** Visibility rule applied by an Attention operation. */
enum class AttentionMaskKind : uint8_t {
  /** Every query may attend to tokens on both sides. */
  BIDIRECTIONAL,

  /** A query may not attend to future tokens. */
  CAUSAL,
};

/** Finite attention visibility measured relative to the query position. */
struct AttentionWindow final {
  /** Maximum number of visible tokens to the left. */
  int64_t left_;

  /** Maximum number of visible tokens to the right. */
  int64_t right_;

  [[nodiscard]] auto operator==(const AttentionWindow &) const -> bool = default;
};

/** Arrangement of paired channels used by rotary position embedding. */
enum class RotaryLayout : uint8_t {
  /** The first and second halves of the head dimension form rotary pairs. */
  SPLIT_HALF,

  /** Adjacent channels form rotary pairs. */
  INTERLEAVED,
};

/** Function that converts router logits into expert scores. */
enum class RoutingScoreFunction : uint8_t {
  /** Normalizes router logits across all experts. */
  SOFTMAX,

  /** Applies an independent sigmoid to each router logit. */
  SIGMOID,
};

/** Function that reduces expert scores into one score per expert group. */
enum class GroupScoreFunction : uint8_t {
  /** Uses the highest score in each group. */
  MAX,

  /** Uses the sum of the two highest scores in each group. */
  TOP2_SUM,
};

/** Optional group-selection stage performed before expert top-k selection. */
struct ExpertGroupRouting final {
  /** Number of disjoint expert groups. */
  int64_t group_count_;

  /** Number of groups retained for each token. */
  int64_t selected_group_count_;

  /** Reduction used to rank groups. */
  GroupScoreFunction score_function_;

  [[nodiscard]] auto operator==(const ExpertGroupRouting &) const -> bool = default;
};

/** Normalization applied to selected expert routing weights. */
enum class RoutingWeightNormalization : uint8_t {
  /** Keeps selected expert weights unchanged. */
  NONE,

  /** Renormalizes selected expert weights to sum to one. */
  SUM,
};

/** Gated activation used by an expert feed-forward network. */
enum class GatedActivation : uint8_t {
  /** SiLU/Swish gated activation. */
  SILU,

  /** GELU gated activation. */
  GELU,
};

}  // namespace zephyr
