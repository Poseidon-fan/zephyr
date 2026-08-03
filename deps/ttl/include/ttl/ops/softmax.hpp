#pragma once

#include <cstdint>
#include <source_location>
#include <vector>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

struct SoftmaxOptions final {
  std::vector<int64_t> axes_;
};

void SoftmaxOut(ExecutionContext &context, Tensor &output, const Tensor &input, const SoftmaxOptions &options,
                std::source_location location = std::source_location::current());
[[nodiscard]] auto Softmax(ExecutionContext &context, const Tensor &input, const SoftmaxOptions &options,
                           std::source_location location = std::source_location::current()) -> Tensor;

void LogSoftmaxOut(ExecutionContext &context, Tensor &output, const Tensor &input, const SoftmaxOptions &options,
                   std::source_location location = std::source_location::current());
[[nodiscard]] auto LogSoftmax(ExecutionContext &context, const Tensor &input, const SoftmaxOptions &options,
                              std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
