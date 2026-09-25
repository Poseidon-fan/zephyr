#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

#include "parallel/tp.hpp"
#include "weight/weight_builder.hpp"

namespace zephyr::layer {

/** Dense projection with weight [out_features, in_features] and optional bias [out_features]. */
class Linear final {
 public:
  /** Tensor handles can share storage with another layer, including a tied embedding. */
  Linear(ttl::Tensor weight, std::optional<ttl::Tensor> bias) : weight_(std::move(weight)), bias_(std::move(bias)) {}

  /** When bias is requested, its checkpoint parameter is required. */
  [[nodiscard]] static auto Load(ttl::ExecutionContext &context, int64_t in_features, int64_t out_features, bool bias,
                                 const weight::WeightBuilder &builder) -> Linear;
  [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor;

 private:
  ttl::Tensor weight_;
  std::optional<ttl::Tensor> bias_;
};

/** Output-dimension partition; Forward returns local features without a collective. */
class ColumnParallelLayer final {
 public:
  /** A requested bias is loaded when present in the checkpoint. */
  [[nodiscard]] static auto Load(ttl::ExecutionContext &context, int64_t in_features, int64_t out_features, bool bias,
                                 const parallel::TpRankContext &rank, const weight::WeightBuilder &builder)
      -> ColumnParallelLayer;
  /** An explicit output-axis shard also supports KV-head replication. */
  [[nodiscard]] static auto LoadWithShard(ttl::ExecutionContext &context, int64_t in_features, int64_t out_features,
                                          bool bias, const parallel::TpRankContext &rank, weight::Shard shard,
                                          const weight::WeightBuilder &builder) -> ColumnParallelLayer;
  [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor;

 private:
  ColumnParallelLayer(Linear projection, std::optional<ttl::Tensor> bias)
      : projection_(std::move(projection)), bias_(std::move(bias)) {}

  Linear projection_;
  std::optional<ttl::Tensor> bias_;
};

/** Input-dimension partition; callers supply local input and receive the sum across ranks. */
class RowParallelLayer final {
 public:
  /** Load an optional checkpoint bias and apply it once, after the reduction. */
  [[nodiscard]] static auto Load(ttl::ExecutionContext &context, int64_t in_features, int64_t out_features, bool bias,
                                 const parallel::TpRankContext &rank, const weight::WeightBuilder &builder)
      -> RowParallelLayer;
  [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor;

 private:
  RowParallelLayer(Linear projection, std::optional<ttl::Tensor> bias, parallel::TpRankContext rank)
      : projection_(std::move(projection)), bias_(std::move(bias)), rank_(rank) {}

  Linear projection_;
  std::optional<ttl::Tensor> bias_;
  /** The owning TP context must outlive this layer and its in-flight collectives. */
  parallel::TpRankContext rank_;
};

/** Partition whole KV heads, or replicate one head across consecutive ranks when TP exceeds the head count. */
[[nodiscard]] auto ComputeKvShard(int64_t num_kv_heads, int64_t head_dim, const parallel::TpRankContext &rank)
    -> weight::Shard;
/** Compute rank-local query heads per KV head, including KV-head replication. */
[[nodiscard]] auto ComputeNumKvGroups(int64_t num_kv_heads, int64_t num_attention_heads,
                                      const parallel::TpRankContext &rank) -> int64_t;

}  // namespace zephyr::layer
