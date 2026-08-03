#pragma once

#include <cstdint>
#include <source_location>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

void IndexSelectOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis, const Tensor &index,
                    std::source_location location = std::source_location::current());
[[nodiscard]] auto IndexSelect(ExecutionContext &context, const Tensor &input, int64_t axis, const Tensor &index,
                               std::source_location location = std::source_location::current()) -> Tensor;

void GatherOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis, const Tensor &index,
               std::source_location location = std::source_location::current());
[[nodiscard]] auto Gather(ExecutionContext &context, const Tensor &input, int64_t axis, const Tensor &index,
                          std::source_location location = std::source_location::current()) -> Tensor;

void TakeAlongDimensionOut(ExecutionContext &context, Tensor &output, const Tensor &input, const Tensor &index,
                           int64_t axis, std::source_location location = std::source_location::current());
[[nodiscard]] auto TakeAlongDimension(ExecutionContext &context, const Tensor &input, const Tensor &index, int64_t axis,
                                      std::source_location location = std::source_location::current()) -> Tensor;

void GatherRowsOut(ExecutionContext &context, Tensor &output, const Tensor &table, const Tensor &indices,
                   std::source_location location = std::source_location::current());
[[nodiscard]] auto GatherRows(ExecutionContext &context, const Tensor &table, const Tensor &indices,
                              std::source_location location = std::source_location::current()) -> Tensor;

void EmbeddingOut(ExecutionContext &context, Tensor &output, const Tensor &weight, const Tensor &token_ids,
                  std::source_location location = std::source_location::current());
[[nodiscard]] auto Embedding(ExecutionContext &context, const Tensor &weight, const Tensor &token_ids,
                             std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
