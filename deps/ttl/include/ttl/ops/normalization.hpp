#pragma once

#include <cstdint>
#include <optional>
#include <source_location>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

struct NormOptions final {
  int64_t normalized_rank_;
  float epsilon_{1.0e-5F};
};

void LayerNormOut(ExecutionContext &context, Tensor &output, const Tensor &input, const std::optional<Tensor> &weight,
                  const std::optional<Tensor> &bias, const NormOptions &options,
                  std::source_location location = std::source_location::current());
[[nodiscard]] auto LayerNorm(ExecutionContext &context, const Tensor &input, const std::optional<Tensor> &weight,
                             const std::optional<Tensor> &bias, const NormOptions &options,
                             std::source_location location = std::source_location::current()) -> Tensor;

void RmsNormOut(ExecutionContext &context, Tensor &output, const Tensor &input, const std::optional<Tensor> &weight,
                const NormOptions &options, std::source_location location = std::source_location::current());
[[nodiscard]] auto RmsNorm(ExecutionContext &context, const Tensor &input, const std::optional<Tensor> &weight,
                           const NormOptions &options, std::source_location location = std::source_location::current())
    -> Tensor;

}  // namespace ttl
