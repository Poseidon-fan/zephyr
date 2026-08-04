#pragma once

#include <cstdint>
#include <source_location>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/**
 * @brief Select entries along one input axis using a rank-one index tensor.
 *
 * `index` contains non-negative INT32 or INT64 indices. The output shape replaces `input.shape[axis]` with the index
 * length and must be non-overlapping dense and disjoint from both inputs.
 */
void IndexSelectOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis, const Tensor &index,
                    std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and select entries along one axis. */
[[nodiscard]] auto IndexSelect(ExecutionContext &context, const Tensor &input, int64_t axis, const Tensor &index,
                               std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Gather one input value for every index at the corresponding non-axis coordinates.
 *
 * `input` and `index` have equal rank; non-axis index extents may not exceed the corresponding input extents. The
 * output shape equals the index shape. Indices are non-negative INT32 or INT64 values.
 */
void GatherOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis, const Tensor &index,
               std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and gather values using `index`. */
[[nodiscard]] auto Gather(ExecutionContext &context, const Tensor &input, int64_t axis, const Tensor &index,
                          std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Gather values along one dimension using a broadcast-compatible index tensor.
 *
 * `input` and `index` have equal rank. Their non-axis dimensions must broadcast, and the result uses that broadcast
 * shape with the index extent along `axis`. Indices are non-negative INT32 or INT64 values.
 */
void TakeAlongDimensionOut(ExecutionContext &context, Tensor &output, const Tensor &input, const Tensor &index,
                           int64_t axis, std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and take values along one dimension. */
[[nodiscard]] auto TakeAlongDimension(ExecutionContext &context, const Tensor &input, const Tensor &index, int64_t axis,
                                      std::source_location location = std::source_location::current()) -> Tensor;

/**
 * Copy input and overwrite elements selected by index along one axis with source values.
 *
 * `index` and `source` have the same rank as `input`. Their extents may not exceed `source`, and non-axis extents may
 * not exceed `input`. Duplicate destination indices have an unspecified winner, matching parallel overwrite-scatter
 * semantics; callers requiring deterministic duplicate handling must reduce or deduplicate first. `output` may
 * exactly alias `input` but must not otherwise overlap any operand.
 */
void ScatterElementsOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis,
                        const Tensor &index, const Tensor &source,
                        std::source_location location = std::source_location::current());
[[nodiscard]] auto ScatterElements(ExecutionContext &context, const Tensor &input, int64_t axis, const Tensor &index,
                                   const Tensor &source,
                                   std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Gather slices along the first axis of a table using an arbitrary-shape index tensor.
 *
 * `table` has rank at least two and the output shape is `indices.shape + table.shape[1:]`. Indices are non-negative
 * INT32 or INT64 values, and the output must be non-overlapping dense and disjoint from both inputs.
 */
void GatherRowsOut(ExecutionContext &context, Tensor &output, const Tensor &table, const Tensor &indices,
                   std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and gather first-axis slices from `table`. */
[[nodiscard]] auto GatherRows(ExecutionContext &context, const Tensor &table, const Tensor &indices,
                              std::source_location location = std::source_location::current()) -> Tensor;

/** @brief Gather embedding rows from rank-two `weight` for arbitrary-shape integer `token_ids`. */
void EmbeddingOut(ExecutionContext &context, Tensor &output, const Tensor &weight, const Tensor &token_ids,
                  std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and gather embedding rows. */
[[nodiscard]] auto Embedding(ExecutionContext &context, const Tensor &weight, const Tensor &token_ids,
                             std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
