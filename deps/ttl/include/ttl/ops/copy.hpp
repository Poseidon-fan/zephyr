#pragma once

#include <source_location>

#include "ttl/execution_context.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

/** Copy between equal-shape, equal-dtype tensors on one device. */
void CopyOut(ExecutionContext &context, Tensor &output, const Tensor &input,
             std::source_location location = std::source_location::current());

/** Allocate a canonical contiguous tensor and copy the input's logical values into it. */
[[nodiscard]] auto Clone(ExecutionContext &context, const Tensor &input,
                         std::source_location location = std::source_location::current()) -> Tensor;

/** Copy into a canonical contiguous output tensor. */
void ContiguousOut(ExecutionContext &context, Tensor &output, const Tensor &input,
                   std::source_location location = std::source_location::current());

/** Return the input when already contiguous; otherwise materialize a canonical contiguous copy. */
[[nodiscard]] auto Contiguous(ExecutionContext &context, const Tensor &input,
                              std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
