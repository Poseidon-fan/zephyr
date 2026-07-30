#pragma once

#include <cstdint>
#include <optional>
#include <source_location>

#include "ttl/execution_context.hpp"
#include "ttl/scalar.hpp"
#include "ttl/tensor.hpp"

namespace ttl {

enum class GeluApproximation : uint8_t {
  NONE,
  TANH,
};

void AddOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
            std::source_location location = std::source_location::current());
void AddOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
            std::source_location location = std::source_location::current());
[[nodiscard]] auto Add(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                       std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto Add(ExecutionContext &context, const Tensor &input, Scalar scalar,
                       std::source_location location = std::source_location::current()) -> Tensor;

void SubtractOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                 std::source_location location = std::source_location::current());
void SubtractOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                 std::source_location location = std::source_location::current());
[[nodiscard]] auto Subtract(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                            std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto Subtract(ExecutionContext &context, const Tensor &input, Scalar scalar,
                            std::source_location location = std::source_location::current()) -> Tensor;

void MultiplyOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                 std::source_location location = std::source_location::current());
void MultiplyOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                 std::source_location location = std::source_location::current());
[[nodiscard]] auto Multiply(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                            std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto Multiply(ExecutionContext &context, const Tensor &input, Scalar scalar,
                            std::source_location location = std::source_location::current()) -> Tensor;

void DivideOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
               std::source_location location = std::source_location::current());
void DivideOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
               std::source_location location = std::source_location::current());
[[nodiscard]] auto Divide(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                          std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto Divide(ExecutionContext &context, const Tensor &input, Scalar scalar,
                          std::source_location location = std::source_location::current()) -> Tensor;

void MaximumOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                std::source_location location = std::source_location::current());
void MaximumOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                std::source_location location = std::source_location::current());
[[nodiscard]] auto Maximum(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                           std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto Maximum(ExecutionContext &context, const Tensor &input, Scalar scalar,
                           std::source_location location = std::source_location::current()) -> Tensor;

void MinimumOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                std::source_location location = std::source_location::current());
void MinimumOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                std::source_location location = std::source_location::current());
[[nodiscard]] auto Minimum(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                           std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto Minimum(ExecutionContext &context, const Tensor &input, Scalar scalar,
                           std::source_location location = std::source_location::current()) -> Tensor;

void EqualOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
              std::source_location location = std::source_location::current());
void EqualOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
              std::source_location location = std::source_location::current());
[[nodiscard]] auto Equal(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                         std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto Equal(ExecutionContext &context, const Tensor &input, Scalar scalar,
                         std::source_location location = std::source_location::current()) -> Tensor;

void NotEqualOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                 std::source_location location = std::source_location::current());
void NotEqualOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                 std::source_location location = std::source_location::current());
[[nodiscard]] auto NotEqual(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                            std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto NotEqual(ExecutionContext &context, const Tensor &input, Scalar scalar,
                            std::source_location location = std::source_location::current()) -> Tensor;

void LessOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
             std::source_location location = std::source_location::current());
void LessOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
             std::source_location location = std::source_location::current());
[[nodiscard]] auto Less(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                        std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto Less(ExecutionContext &context, const Tensor &input, Scalar scalar,
                        std::source_location location = std::source_location::current()) -> Tensor;

void LessEqualOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                  std::source_location location = std::source_location::current());
void LessEqualOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                  std::source_location location = std::source_location::current());
[[nodiscard]] auto LessEqual(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                             std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto LessEqual(ExecutionContext &context, const Tensor &input, Scalar scalar,
                             std::source_location location = std::source_location::current()) -> Tensor;

void GreaterOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                std::source_location location = std::source_location::current());
void GreaterOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                std::source_location location = std::source_location::current());
[[nodiscard]] auto Greater(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                           std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto Greater(ExecutionContext &context, const Tensor &input, Scalar scalar,
                           std::source_location location = std::source_location::current()) -> Tensor;

void GreaterEqualOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                     std::source_location location = std::source_location::current());
void GreaterEqualOut(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                     std::source_location location = std::source_location::current());
[[nodiscard]] auto GreaterEqual(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                                std::source_location location = std::source_location::current()) -> Tensor;
[[nodiscard]] auto GreaterEqual(ExecutionContext &context, const Tensor &input, Scalar scalar,
                                std::source_location location = std::source_location::current()) -> Tensor;

void NegateOut(ExecutionContext &context, Tensor &output, const Tensor &input,
               std::source_location location = std::source_location::current());
[[nodiscard]] auto Negate(ExecutionContext &context, const Tensor &input,
                          std::source_location location = std::source_location::current()) -> Tensor;

void AbsOut(ExecutionContext &context, Tensor &output, const Tensor &input,
            std::source_location location = std::source_location::current());
[[nodiscard]] auto Abs(ExecutionContext &context, const Tensor &input,
                       std::source_location location = std::source_location::current()) -> Tensor;

void ExpOut(ExecutionContext &context, Tensor &output, const Tensor &input,
            std::source_location location = std::source_location::current());
[[nodiscard]] auto Exp(ExecutionContext &context, const Tensor &input,
                       std::source_location location = std::source_location::current()) -> Tensor;

void LogOut(ExecutionContext &context, Tensor &output, const Tensor &input,
            std::source_location location = std::source_location::current());
[[nodiscard]] auto Log(ExecutionContext &context, const Tensor &input,
                       std::source_location location = std::source_location::current()) -> Tensor;

void SqrtOut(ExecutionContext &context, Tensor &output, const Tensor &input,
             std::source_location location = std::source_location::current());
[[nodiscard]] auto Sqrt(ExecutionContext &context, const Tensor &input,
                        std::source_location location = std::source_location::current()) -> Tensor;

void RsqrtOut(ExecutionContext &context, Tensor &output, const Tensor &input,
              std::source_location location = std::source_location::current());
[[nodiscard]] auto Rsqrt(ExecutionContext &context, const Tensor &input,
                         std::source_location location = std::source_location::current()) -> Tensor;

void SinOut(ExecutionContext &context, Tensor &output, const Tensor &input,
            std::source_location location = std::source_location::current());
[[nodiscard]] auto Sin(ExecutionContext &context, const Tensor &input,
                       std::source_location location = std::source_location::current()) -> Tensor;

void CosOut(ExecutionContext &context, Tensor &output, const Tensor &input,
            std::source_location location = std::source_location::current());
[[nodiscard]] auto Cos(ExecutionContext &context, const Tensor &input,
                       std::source_location location = std::source_location::current()) -> Tensor;

void TanhOut(ExecutionContext &context, Tensor &output, const Tensor &input,
             std::source_location location = std::source_location::current());
[[nodiscard]] auto Tanh(ExecutionContext &context, const Tensor &input,
                        std::source_location location = std::source_location::current()) -> Tensor;

void SigmoidOut(ExecutionContext &context, Tensor &output, const Tensor &input,
                std::source_location location = std::source_location::current());
[[nodiscard]] auto Sigmoid(ExecutionContext &context, const Tensor &input,
                           std::source_location location = std::source_location::current()) -> Tensor;

void ReluOut(ExecutionContext &context, Tensor &output, const Tensor &input,
             std::source_location location = std::source_location::current());
[[nodiscard]] auto Relu(ExecutionContext &context, const Tensor &input,
                        std::source_location location = std::source_location::current()) -> Tensor;

void SiluOut(ExecutionContext &context, Tensor &output, const Tensor &input,
             std::source_location location = std::source_location::current());
[[nodiscard]] auto Silu(ExecutionContext &context, const Tensor &input,
                        std::source_location location = std::source_location::current()) -> Tensor;

void GeluOut(ExecutionContext &context, Tensor &output, const Tensor &input, GeluApproximation approximation,
             std::source_location location = std::source_location::current());
[[nodiscard]] auto Gelu(ExecutionContext &context, const Tensor &input, GeluApproximation approximation,
                        std::source_location location = std::source_location::current()) -> Tensor;

void ClampOut(ExecutionContext &context, Tensor &output, const Tensor &input, std::optional<Scalar> minimum,
              std::optional<Scalar> maximum, std::source_location location = std::source_location::current());
[[nodiscard]] auto Clamp(ExecutionContext &context, const Tensor &input, std::optional<Scalar> minimum,
                         std::optional<Scalar> maximum, std::source_location location = std::source_location::current())
    -> Tensor;

void LogicalAndOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                   std::source_location location = std::source_location::current());
[[nodiscard]] auto LogicalAnd(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                              std::source_location location = std::source_location::current()) -> Tensor;

void LogicalOrOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                  std::source_location location = std::source_location::current());
[[nodiscard]] auto LogicalOr(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                             std::source_location location = std::source_location::current()) -> Tensor;

void LogicalNotOut(ExecutionContext &context, Tensor &output, const Tensor &input,
                   std::source_location location = std::source_location::current());
[[nodiscard]] auto LogicalNot(ExecutionContext &context, const Tensor &input,
                              std::source_location location = std::source_location::current()) -> Tensor;

void WhereOut(ExecutionContext &context, Tensor &output, const Tensor &condition, const Tensor &true_value,
              const Tensor &false_value, std::source_location location = std::source_location::current());
[[nodiscard]] auto Where(ExecutionContext &context, const Tensor &condition, const Tensor &true_value,
                         const Tensor &false_value, std::source_location location = std::source_location::current())
    -> Tensor;

}  // namespace ttl
