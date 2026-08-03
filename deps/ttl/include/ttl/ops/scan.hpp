#pragma once

#include <cstdint>
#include <source_location>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/** Inclusive prefix sum along one axis. Integer overflow follows TTL's modular integer arithmetic. */
void CumulativeSumOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis,
                      std::source_location location = std::source_location::current());
[[nodiscard]] auto CumulativeSum(ExecutionContext &context, const Tensor &input, int64_t axis,
                                 std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
