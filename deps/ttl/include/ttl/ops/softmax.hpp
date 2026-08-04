#pragma once

#include <cstdint>
#include <source_location>
#include <vector>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/** @brief Selects the nonempty set of axes normalized together by softmax. */
struct SoftmaxOptions final {
  std::vector<int64_t> axes_;
};

/**
 * @brief Compute softmax over the selected axes of a floating tensor.
 *
 * Input and output shapes and dtypes match. The output must be non-overlapping dense and disjoint from the input.
 */
void SoftmaxOut(ExecutionContext &context, Tensor &output, const Tensor &input, const SoftmaxOptions &options,
                std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and compute softmax. */
[[nodiscard]] auto Softmax(ExecutionContext &context, const Tensor &input, const SoftmaxOptions &options,
                           std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Compute log-softmax over the selected axes of a floating tensor.
 *
 * Input and output shapes and dtypes match. The output must be non-overlapping dense and disjoint from the input.
 */
void LogSoftmaxOut(ExecutionContext &context, Tensor &output, const Tensor &input, const SoftmaxOptions &options,
                   std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and compute log-softmax. */
[[nodiscard]] auto LogSoftmax(ExecutionContext &context, const Tensor &input, const SoftmaxOptions &options,
                              std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
