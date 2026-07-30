#pragma once

#include <source_location>

#include "ttl/dtype.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

/** Convert an input into an equal-shape output using checked TTL dtype conversion semantics. */
void CastOut(ExecutionContext &context, Tensor &output, const Tensor &input,
             std::source_location location = std::source_location::current());

/** Allocate a canonical contiguous output and convert every logical input value. */
[[nodiscard]] auto Cast(ExecutionContext &context, const Tensor &input, DType dtype,
                        std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
