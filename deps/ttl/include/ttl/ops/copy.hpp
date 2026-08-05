#pragma once

#include <cstddef>
#include <source_location>
#include <span>

#include "ttl/runtime/event.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/tensor/tensor.hpp"

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
 * @brief Copy a contiguous tensor between different runtime-managed CUDA devices.
 *
 * The destination stream first waits for `source_ready`; the dependency is explicit because allocation lifetime does
 * not imply that the source producer has completed.
 */
void CopyPeerOut(ExecutionContext &destination_context, Tensor &destination, const Tensor &source,
                 const Event &source_ready, std::source_location location = std::source_location::current());

/** Enqueue a full-buffer copy from page-locked host memory into a contiguous tensor. */
void CopyFromPinnedAsync(ExecutionContext &context, Tensor &output, const PinnedBuffer &source,
                         std::source_location location = std::source_location::current());

/** Enqueue a full-buffer copy from a contiguous tensor into page-locked host memory. */
void CopyToPinnedAsync(ExecutionContext &context, PinnedBuffer &output, const Tensor &source,
                       std::source_location location = std::source_location::current());

/** Copy pageable host bytes into a contiguous tensor and block until the transfer and async checks complete. */
void CopyFromHostBlocking(ExecutionContext &context, Tensor &output, std::span<const std::byte> source,
                          std::source_location location = std::source_location::current());

/** Copy a contiguous tensor into pageable host bytes and block until the transfer and async checks complete. */
void CopyToHostBlocking(ExecutionContext &context, std::span<std::byte> output, const Tensor &source,
                        std::source_location location = std::source_location::current());

}  // namespace ttl
