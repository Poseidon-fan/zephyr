#pragma once

#include <cstdint>
#include <optional>
#include <source_location>

#include "ttl/execution_context.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

struct MatmulOptions final {
  bool allow_tf32_{true};
};

enum class LinearActivation : uint8_t {
  NONE,
  RELU,
  GELU,
};

struct LinearOptions final {
  LinearActivation activation_{LinearActivation::NONE};
  GeluApproximation gelu_approximation_{GeluApproximation::NONE};
  MatmulOptions matmul_;
};

void MatmulOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
               const MatmulOptions &options = {}, std::source_location location = std::source_location::current());
[[nodiscard]] auto Matmul(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                          const MatmulOptions &options = {},
                          std::source_location location = std::source_location::current()) -> Tensor;

void BatchedMatmulOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                      const MatmulOptions &options = {},
                      std::source_location location = std::source_location::current());
[[nodiscard]] auto BatchedMatmul(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                                 const MatmulOptions &options = {},
                                 std::source_location location = std::source_location::current()) -> Tensor;

void LinearOut(ExecutionContext &context, Tensor &output, const Tensor &input, const Tensor &weight,
               const std::optional<Tensor> &bias = std::nullopt, const LinearOptions &options = {},
               std::source_location location = std::source_location::current());
[[nodiscard]] auto Linear(ExecutionContext &context, const Tensor &input, const Tensor &weight,
                          const std::optional<Tensor> &bias = std::nullopt, const LinearOptions &options = {},
                          std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
