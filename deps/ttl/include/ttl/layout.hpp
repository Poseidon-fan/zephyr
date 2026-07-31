#pragma once

#include <cstdint>
#include <optional>
#include <source_location>
#include <span>
#include <vector>

#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

class ExecutionContext;

/** Resolve one optional -1 dimension and validate that a requested reshape preserves the element count. */
[[nodiscard]] auto InferReshape(const Tensor &input, std::span<const int64_t> requested,
                                std::source_location location = std::source_location::current()) -> Shape;

/** Create a metadata-only view, or reject a shape that is incompatible with the input's physical chunks. */
[[nodiscard]] auto View(const Tensor &input, const Shape &shape,
                        std::source_location location = std::source_location::current()) -> Tensor;

/** Infer a requested -1 dimension and create a metadata-only view. */
[[nodiscard]] auto View(const Tensor &input, std::span<const int64_t> requested,
                        std::source_location location = std::source_location::current()) -> Tensor;

/** Reshape as a view when possible; otherwise asynchronously materialize a contiguous tensor on the context stream. */
[[nodiscard]] auto Reshape(ExecutionContext &context, const Tensor &input, const Shape &shape,
                           std::source_location location = std::source_location::current()) -> Tensor;

/** Infer one optional -1 dimension and reshape, materializing only when the source strides require it. */
[[nodiscard]] auto Reshape(ExecutionContext &context, const Tensor &input, std::span<const int64_t> requested,
                           std::source_location location = std::source_location::current()) -> Tensor;

/** Collapse an inclusive axis range into one dimension, materializing only when required by the source strides. */
[[nodiscard]] auto Flatten(ExecutionContext &context, const Tensor &input, int64_t start_axis = 0,
                           int64_t end_axis = -1, std::source_location location = std::source_location::current())
    -> Tensor;

/** Reorder dimensions without moving data. Every input axis must appear exactly once. */
[[nodiscard]] auto Permute(const Tensor &input, std::span<const int64_t> axes,
                           std::source_location location = std::source_location::current()) -> Tensor;

/** Exchange two dimensions without moving data. */
[[nodiscard]] auto Transpose(const Tensor &input, int64_t axis_a, int64_t axis_b,
                             std::source_location location = std::source_location::current()) -> Tensor;

/** Remove all size-one dimensions, or one explicitly selected size-one dimension. */
[[nodiscard]] auto Squeeze(const Tensor &input, std::optional<int64_t> axis = std::nullopt,
                           std::source_location location = std::source_location::current()) -> Tensor;

/** Insert a size-one dimension at an axis in [-rank - 1, rank]. */
[[nodiscard]] auto Unsqueeze(const Tensor &input, int64_t axis,
                             std::source_location location = std::source_location::current()) -> Tensor;

/** Select a consecutive range along one dimension without moving data. */
[[nodiscard]] auto Narrow(const Tensor &input, int64_t axis, int64_t start, int64_t length,
                          std::source_location location = std::source_location::current()) -> Tensor;

/** Partition one dimension into explicitly sized metadata-only Narrow views. */
[[nodiscard]] auto Split(const Tensor &input, std::span<const int64_t> sizes, int64_t axis,
                         std::source_location location = std::source_location::current()) -> std::vector<Tensor>;

/**
 * Partition one dimension into at most the requested number of nonempty metadata-only views.
 *
 * An empty selected dimension produces exactly `chunks` empty views.
 */
[[nodiscard]] auto Chunk(const Tensor &input, int64_t chunks, int64_t axis,
                         std::source_location location = std::source_location::current()) -> std::vector<Tensor>;

/** Select a positive-step Python-style slice along one dimension without moving data. */
[[nodiscard]] auto Slice(const Tensor &input, int64_t axis, std::optional<int64_t> start, std::optional<int64_t> stop,
                         int64_t step = 1, std::source_location location = std::source_location::current()) -> Tensor;

/** Select one index and remove its dimension without moving data. */
[[nodiscard]] auto Select(const Tensor &input, int64_t axis, int64_t index,
                          std::source_location location = std::source_location::current()) -> Tensor;

/** Broadcast size-one dimensions through zero strides. The returned Tensor is read-only as an operator output. */
[[nodiscard]] auto Expand(const Tensor &input, const Shape &shape,
                          std::source_location location = std::source_location::current()) -> Tensor;

/** Infer the common trailing-dimension broadcast shape. An empty input list produces a scalar shape. */
[[nodiscard]] auto BroadcastShapes(std::span<const Shape> shapes,
                                   std::source_location location = std::source_location::current()) -> Shape;

}  // namespace ttl
