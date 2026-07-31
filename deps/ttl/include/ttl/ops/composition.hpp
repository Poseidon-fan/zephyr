#pragma once

#include <cstdint>
#include <source_location>
#include <span>

#include "ttl/execution_context.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

void ConcatOut(ExecutionContext &context, Tensor &output, std::span<const Tensor> inputs, int64_t axis,
               std::source_location location = std::source_location::current());
[[nodiscard]] auto Concat(ExecutionContext &context, std::span<const Tensor> inputs, int64_t axis,
                          std::source_location location = std::source_location::current()) -> Tensor;

void StackOut(ExecutionContext &context, Tensor &output, std::span<const Tensor> inputs, int64_t axis,
              std::source_location location = std::source_location::current());
[[nodiscard]] auto Stack(ExecutionContext &context, std::span<const Tensor> inputs, int64_t axis,
                         std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
