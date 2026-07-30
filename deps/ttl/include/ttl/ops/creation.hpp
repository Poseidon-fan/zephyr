#pragma once

#include <source_location>

#include "ttl/execution_context.hpp"
#include "ttl/scalar.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

/** Fill every logical element of a writable tensor on the context stream. */
void FillOut(ExecutionContext &context, Tensor &output, Scalar value,
             std::source_location location = std::source_location::current());

}  // namespace ttl
