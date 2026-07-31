#pragma once

#include <source_location>

#include "ttl/event.hpp"
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

/**
 * Copy a contiguous tensor between different runtime-managed CUDA devices.
 *
 * The destination stream first waits for `source_ready`; the dependency is explicit because allocation lifetime does
 * not imply that the source producer has completed.
 */
void CopyPeerOut(ExecutionContext &destination_context, Tensor &destination, const Tensor &source,
                 const Event &source_ready, std::source_location location = std::source_location::current());

}  // namespace ttl
