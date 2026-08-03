#pragma once

#include <cstdint>
#include <optional>
#include <source_location>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

enum class CausalAlignment : uint8_t {
  UPPER_LEFT,
  LOWER_RIGHT,
};

struct SdpaOptions final {
  std::optional<float> scale_;
  bool causal_{false};
  CausalAlignment causal_alignment_{CausalAlignment::UPPER_LEFT};
};

void ScaledDotProductAttentionOut(ExecutionContext &context, Tensor &output, const Tensor &query, const Tensor &key,
                                  const Tensor &value, const std::optional<Tensor> &mask = std::nullopt,
                                  const SdpaOptions &options = {},
                                  std::source_location location = std::source_location::current());
[[nodiscard]] auto ScaledDotProductAttention(ExecutionContext &context, const Tensor &query, const Tensor &key,
                                             const Tensor &value, const std::optional<Tensor> &mask = std::nullopt,
                                             const SdpaOptions &options = {},
                                             std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
