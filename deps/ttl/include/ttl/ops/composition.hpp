#pragma once

#include <cstdint>
#include <source_location>
#include <span>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/**
 * @brief Concatenate tensors along an existing axis into a caller-provided output.
 *
 * Inputs must be nonempty, have one dtype and rank, and match in every dimension except `axis`. The output must be
 * non-overlapping dense and disjoint from every input.
 */
void ConcatOut(ExecutionContext &context, Tensor &output, std::span<const Tensor> inputs, int64_t axis,
               std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous tensor containing `inputs` concatenated along `axis`. */
[[nodiscard]] auto Concat(ExecutionContext &context, std::span<const Tensor> inputs, int64_t axis,
                          std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Stack equal-shaped tensors along a newly inserted axis into a caller-provided output.
 *
 * Inputs must be nonempty and have one shape and dtype. The output must be non-overlapping dense and disjoint from
 * every input.
 */
void StackOut(ExecutionContext &context, Tensor &output, std::span<const Tensor> inputs, int64_t axis,
              std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous tensor containing `inputs` stacked along a new `axis`. */
[[nodiscard]] auto Stack(ExecutionContext &context, std::span<const Tensor> inputs, int64_t axis,
                         std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
