#include "ttl/ops/elementwise.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "ttl/common/error.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/elementwise_ops.hpp"
#include "ttl/internal/runtime/execution/device_error.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/layout.hpp"
#include "ttl/tensor/scalar.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

[[noreturn]] void ThrowElementwiseError(std::string_view operation, std::string_view reason,
                                        std::source_location location) {
  std::string message{operation};
  message.append(": ");
  message.append(reason);
  throw InvalidArgumentError(std::move(message), location);
}

[[noreturn]] void ThrowUnsupportedDType(std::string_view operation, DType dtype, std::source_location location) {
  std::string message{operation};
  message.append(" does not support dtype ");
  message.append(GetDTypeInfo(dtype, location).name_);
  throw NotSupportedError(std::move(message), location);
}

[[nodiscard]] auto GetBinaryName(internal::BinaryElementwiseOp operation) noexcept -> std::string_view {
  switch (operation) {
    case internal::BinaryElementwiseOp::ADD:
      return "AddOut";
    case internal::BinaryElementwiseOp::SUBTRACT:
      return "SubtractOut";
    case internal::BinaryElementwiseOp::MULTIPLY:
      return "MultiplyOut";
    case internal::BinaryElementwiseOp::DIVIDE:
      return "DivideOut";
    case internal::BinaryElementwiseOp::MAXIMUM:
      return "MaximumOut";
    case internal::BinaryElementwiseOp::MINIMUM:
      return "MinimumOut";
  }
  return "<invalid binary operation>";
}

[[nodiscard]] auto GetComparisonName(internal::ComparisonElementwiseOp operation) noexcept -> std::string_view {
  switch (operation) {
    case internal::ComparisonElementwiseOp::EQUAL:
      return "EqualOut";
    case internal::ComparisonElementwiseOp::NOT_EQUAL:
      return "NotEqualOut";
    case internal::ComparisonElementwiseOp::LESS:
      return "LessOut";
    case internal::ComparisonElementwiseOp::LESS_EQUAL:
      return "LessEqualOut";
    case internal::ComparisonElementwiseOp::GREATER:
      return "GreaterOut";
    case internal::ComparisonElementwiseOp::GREATER_EQUAL:
      return "GreaterEqualOut";
  }
  return "<invalid comparison operation>";
}

[[nodiscard]] auto GetUnaryName(internal::UnaryElementwiseOp operation) noexcept -> std::string_view {
  switch (operation) {
    case internal::UnaryElementwiseOp::NEGATE:
      return "NegateOut";
    case internal::UnaryElementwiseOp::ABS:
      return "AbsOut";
    case internal::UnaryElementwiseOp::EXP:
      return "ExpOut";
    case internal::UnaryElementwiseOp::LOG:
      return "LogOut";
    case internal::UnaryElementwiseOp::SQRT:
      return "SqrtOut";
    case internal::UnaryElementwiseOp::RSQRT:
      return "RsqrtOut";
    case internal::UnaryElementwiseOp::SIN:
      return "SinOut";
    case internal::UnaryElementwiseOp::COS:
      return "CosOut";
    case internal::UnaryElementwiseOp::TANH:
      return "TanhOut";
    case internal::UnaryElementwiseOp::SIGMOID:
      return "SigmoidOut";
    case internal::UnaryElementwiseOp::RELU:
      return "ReluOut";
    case internal::UnaryElementwiseOp::SILU:
      return "SiluOut";
  }
  return "<invalid unary operation>";
}

[[nodiscard]] auto GetLogicalName(internal::LogicalElementwiseOp operation) noexcept -> std::string_view {
  switch (operation) {
    case internal::LogicalElementwiseOp::AND:
      return "LogicalAndOut";
    case internal::LogicalElementwiseOp::OR:
      return "LogicalOrOut";
    case internal::LogicalElementwiseOp::NOT:
      return "LogicalNotOut";
  }
  return "<invalid logical operation>";
}

void ValidateBinaryDType(DType dtype, std::string_view operation, std::source_location location) {
  if (!IsSignedInteger(dtype, location) && !IsFloating(dtype, location)) {
    ThrowUnsupportedDType(operation, dtype, location);
  }
}

void ValidateComparisonDType(DType dtype, internal::ComparisonElementwiseOp operation, std::string_view name,
                             std::source_location location) {
  if (operation == internal::ComparisonElementwiseOp::EQUAL ||
      operation == internal::ComparisonElementwiseOp::NOT_EQUAL) {
    return;
  }
  if (dtype == DType::UINT8 || IsSignedInteger(dtype, location) || IsFloating(dtype, location)) {
    return;
  }
  ThrowUnsupportedDType(name, dtype, location);
}

void ValidateFloatingDType(DType dtype, std::string_view operation, std::source_location location) {
  if (!IsFloating(dtype, location)) {
    ThrowUnsupportedDType(operation, dtype, location);
  }
}

void ValidateUnaryDType(DType dtype, internal::UnaryElementwiseOp operation, std::string_view name,
                        std::source_location location) {
  if (IsFloating(dtype, location)) {
    return;
  }
  const auto supports_signed_integer = operation == internal::UnaryElementwiseOp::NEGATE ||
                                       operation == internal::UnaryElementwiseOp::ABS ||
                                       operation == internal::UnaryElementwiseOp::RELU;
  if (supports_signed_integer && IsSignedInteger(dtype, location)) {
    return;
  }
  ThrowUnsupportedDType(name, dtype, location);
}

void ValidateEqualDTypes(DType lhs, DType rhs, std::string_view operation, std::source_location location) {
  if (lhs != rhs) {
    ThrowElementwiseError(operation, "input dtypes must match", location);
  }
}

void ValidateOutputDType(DType actual, DType expected, std::string_view operation, std::source_location location) {
  if (actual != expected) {
    ThrowElementwiseError(operation, "output dtype does not match the inferred dtype", location);
  }
}

void ValidateOutputShape(const Shape &actual, const Shape &expected, std::string_view operation,
                         std::source_location location) {
  if (actual != expected) {
    ThrowElementwiseError(operation, "output shape does not match the broadcast result", location);
  }
}

[[nodiscard]] auto InferBinaryShape(const Tensor &lhs, const Tensor &rhs, std::source_location location) -> Shape {
  const std::array shapes{lhs.GetShape(), rhs.GetShape()};
  return BroadcastShapes(std::span<const Shape>{shapes}, location);
}

[[nodiscard]] auto InferWhereShape(const Tensor &condition, const Tensor &true_value, const Tensor &false_value,
                                   std::source_location location) -> Shape {
  const std::array shapes{condition.GetShape(), true_value.GetShape(), false_value.GetShape()};
  return BroadcastShapes(std::span<const Shape>{shapes}, location);
}

[[nodiscard]] auto NormalizeScalar(Scalar scalar, DType dtype, std::source_location location) -> Scalar {
  switch (dtype) {
    case DType::BOOL:
      return Scalar{scalar.Cast<bool>(location)};
    case DType::UINT8:
      return Scalar{static_cast<int64_t>(scalar.Cast<uint8_t>(location))};
    case DType::INT32:
      return Scalar{static_cast<int64_t>(scalar.Cast<int32_t>(location))};
    case DType::INT64:
      return Scalar{scalar.Cast<int64_t>(location)};
    case DType::FLOAT16:
      return Scalar{static_cast<double>(Float16ToFloat(scalar.Cast<Float16>(location)))};
    case DType::BFLOAT16:
      return Scalar{static_cast<double>(BFloat16ToFloat(scalar.Cast<BFloat16>(location)))};
    case DType::FLOAT32:
      return Scalar{static_cast<double>(scalar.Cast<float>(location))};
  }
  throw InvalidArgumentError("invalid dtype", location);
}

void RecordAndLaunchBinary(internal::OpGuard &guard, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                           internal::BinaryElementwiseOp operation, const internal::ElementwiseIterator &iterator,
                           std::source_location location) {
  auto error_context = internal::DeviceErrorLaunchContext{};
  if (operation == internal::BinaryElementwiseOp::DIVIDE && IsIntegral(output.GetDType(), location)) {
    error_context = guard.RegisterDeviceError(output.GetDType(), output.GetDType());
  }
  guard.RecordTensor(output);
  guard.RecordTensor(lhs);
  guard.RecordTensor(rhs);
  internal::LaunchBinaryElementwise(guard.GetNativeStream(), output.GetDType(), operation, iterator, error_context,
                                    location);
  guard.CheckLaunch();
}

void BinaryOutImpl(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                   internal::BinaryElementwiseOp operation, std::source_location location) {
  const auto name = GetBinaryName(operation);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(lhs);
  guard.ValidateTensor(rhs);
  const auto shape = InferBinaryShape(lhs, rhs, location);
  ValidateOutputShape(output.GetShape(), shape, name, location);
  ValidateEqualDTypes(lhs.GetDType(), rhs.GetDType(), name, location);
  ValidateOutputDType(output.GetDType(), lhs.GetDType(), name, location);
  ValidateBinaryDType(output.GetDType(), name, location);

  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(lhs)
                            .AddInput(rhs)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_ONE_BINARY_INPUT)
                            .SetRequireSameDType(true)
                            .Build(name, location);
  if (output.GetNumElements() == 0) {
    return;
  }
  RecordAndLaunchBinary(guard, output, lhs, rhs, operation, iterator, location);
}

void ScalarBinaryOutImpl(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                         internal::BinaryElementwiseOp operation, std::source_location location) {
  const auto name = GetBinaryName(operation);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  ValidateOutputShape(output.GetShape(), input.GetShape(), name, location);
  ValidateOutputDType(output.GetDType(), input.GetDType(), name, location);
  ValidateBinaryDType(output.GetDType(), name, location);
  scalar = NormalizeScalar(scalar, input.GetDType(), location);

  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(input)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_UNARY)
                            .SetRequireSameDType(true)
                            .Build(name, location);
  if (output.GetNumElements() == 0) {
    return;
  }

  auto error_context = internal::DeviceErrorLaunchContext{};
  if (operation == internal::BinaryElementwiseOp::DIVIDE && IsIntegral(output.GetDType(), location)) {
    error_context = guard.RegisterDeviceError(output.GetDType(), output.GetDType());
  }
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  internal::LaunchScalarBinaryElementwise(guard.GetNativeStream(), output.GetDType(), operation, iterator, scalar,
                                          error_context, location);
  guard.CheckLaunch();
}

[[nodiscard]] auto BinaryImpl(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                              internal::BinaryElementwiseOp operation, std::source_location location) -> Tensor {
  const auto name = GetBinaryName(operation);
  Shape shape;
  DType dtype;
  {
    internal::OpGuard guard{context, name, location};
    guard.ValidateTensor(lhs);
    guard.ValidateTensor(rhs);
    shape = InferBinaryShape(lhs, rhs, location);
    ValidateEqualDTypes(lhs.GetDType(), rhs.GetDType(), name, location);
    ValidateBinaryDType(lhs.GetDType(), name, location);
    dtype = lhs.GetDType();
  }
  auto output = Empty(context, shape, dtype, location);
  BinaryOutImpl(context, output, lhs, rhs, operation, location);
  return output;
}

[[nodiscard]] auto ScalarBinaryImpl(ExecutionContext &context, const Tensor &input, Scalar scalar,
                                    internal::BinaryElementwiseOp operation, std::source_location location) -> Tensor {
  const auto name = GetBinaryName(operation);
  {
    internal::OpGuard guard{context, name, location};
    guard.ValidateTensor(input);
    ValidateBinaryDType(input.GetDType(), name, location);
    scalar = NormalizeScalar(scalar, input.GetDType(), location);
  }
  auto output = Empty(context, input.GetShape(), input.GetDType(), location);
  ScalarBinaryOutImpl(context, output, input, scalar, operation, location);
  return output;
}

void ComparisonOutImpl(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                       internal::ComparisonElementwiseOp operation, std::source_location location) {
  const auto name = GetComparisonName(operation);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(lhs);
  guard.ValidateTensor(rhs);
  const auto shape = InferBinaryShape(lhs, rhs, location);
  ValidateOutputShape(output.GetShape(), shape, name, location);
  ValidateEqualDTypes(lhs.GetDType(), rhs.GetDType(), name, location);
  ValidateOutputDType(output.GetDType(), DType::BOOL, name, location);
  ValidateComparisonDType(lhs.GetDType(), operation, name, location);

  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(lhs)
                            .AddInput(rhs)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_ONE_BINARY_INPUT)
                            .Build(name, location);
  if (output.GetNumElements() == 0) {
    return;
  }
  guard.RecordTensor(output);
  guard.RecordTensor(lhs);
  guard.RecordTensor(rhs);
  internal::LaunchComparisonElementwise(guard.GetNativeStream(), lhs.GetDType(), operation, iterator, location);
  guard.CheckLaunch();
}

void ScalarComparisonOutImpl(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,
                             internal::ComparisonElementwiseOp operation, std::source_location location) {
  const auto name = GetComparisonName(operation);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  ValidateOutputShape(output.GetShape(), input.GetShape(), name, location);
  ValidateOutputDType(output.GetDType(), DType::BOOL, name, location);
  ValidateComparisonDType(input.GetDType(), operation, name, location);
  scalar = NormalizeScalar(scalar, input.GetDType(), location);

  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(input)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_UNARY)
                            .Build(name, location);
  if (output.GetNumElements() == 0) {
    return;
  }
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  internal::LaunchScalarComparisonElementwise(guard.GetNativeStream(), input.GetDType(), operation, iterator, scalar,
                                              location);
  guard.CheckLaunch();
}

[[nodiscard]] auto ComparisonImpl(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                                  internal::ComparisonElementwiseOp operation, std::source_location location)
    -> Tensor {
  const auto name = GetComparisonName(operation);
  Shape shape;
  {
    internal::OpGuard guard{context, name, location};
    guard.ValidateTensor(lhs);
    guard.ValidateTensor(rhs);
    shape = InferBinaryShape(lhs, rhs, location);
    ValidateEqualDTypes(lhs.GetDType(), rhs.GetDType(), name, location);
    ValidateComparisonDType(lhs.GetDType(), operation, name, location);
  }
  auto output = Empty(context, shape, DType::BOOL, location);
  ComparisonOutImpl(context, output, lhs, rhs, operation, location);
  return output;
}

[[nodiscard]] auto ScalarComparisonImpl(ExecutionContext &context, const Tensor &input, Scalar scalar,
                                        internal::ComparisonElementwiseOp operation, std::source_location location)
    -> Tensor {
  const auto name = GetComparisonName(operation);
  {
    internal::OpGuard guard{context, name, location};
    guard.ValidateTensor(input);
    ValidateComparisonDType(input.GetDType(), operation, name, location);
    scalar = NormalizeScalar(scalar, input.GetDType(), location);
  }
  auto output = Empty(context, input.GetShape(), DType::BOOL, location);
  ScalarComparisonOutImpl(context, output, input, scalar, operation, location);
  return output;
}

void UnaryOutImpl(ExecutionContext &context, Tensor &output, const Tensor &input,
                  internal::UnaryElementwiseOp operation, std::source_location location) {
  const auto name = GetUnaryName(operation);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  ValidateOutputShape(output.GetShape(), input.GetShape(), name, location);
  ValidateOutputDType(output.GetDType(), input.GetDType(), name, location);
  ValidateUnaryDType(input.GetDType(), operation, name, location);
  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(input)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_UNARY)
                            .SetRequireSameDType(true)
                            .Build(name, location);
  if (output.GetNumElements() == 0) {
    return;
  }
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  internal::LaunchUnaryElementwise(guard.GetNativeStream(), input.GetDType(), operation, iterator, location);
  guard.CheckLaunch();
}

[[nodiscard]] auto UnaryImpl(ExecutionContext &context, const Tensor &input, internal::UnaryElementwiseOp operation,
                             std::source_location location) -> Tensor {
  const auto name = GetUnaryName(operation);
  {
    internal::OpGuard guard{context, name, location};
    guard.ValidateTensor(input);
    ValidateUnaryDType(input.GetDType(), operation, name, location);
  }
  auto output = Empty(context, input.GetShape(), input.GetDType(), location);
  UnaryOutImpl(context, output, input, operation, location);
  return output;
}

void ValidateGeluApproximation(GeluApproximation approximation, std::source_location location) {
  switch (approximation) {
    case GeluApproximation::NONE:
    case GeluApproximation::TANH:
      return;
  }
  throw InvalidArgumentError("invalid Gelu approximation", location);
}

[[nodiscard]] auto NormalizeClampBound(const std::optional<Scalar> &bound, DType dtype, std::source_location location)
    -> std::optional<Scalar> {
  if (!bound.has_value()) {
    return std::nullopt;
  }
  auto normalized = NormalizeScalar(*bound, dtype, location);
  if (std::isnan(normalized.Cast<float>(location))) {
    throw InvalidArgumentError("Clamp bounds must not be NaN", location);
  }
  return normalized;
}

void ValidateClampBounds(const std::optional<Scalar> &minimum, const std::optional<Scalar> &maximum,
                         std::source_location location) {
  if (!minimum.has_value() && !maximum.has_value()) {
    throw InvalidArgumentError("Clamp requires at least one bound", location);
  }
  if (minimum.has_value() && maximum.has_value() && minimum->Cast<float>(location) > maximum->Cast<float>(location)) {
    throw InvalidArgumentError("Clamp minimum must not exceed maximum", location);
  }
}

void LogicalBinaryOutImpl(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                          internal::LogicalElementwiseOp operation, std::source_location location) {
  const auto name = GetLogicalName(operation);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(lhs);
  guard.ValidateTensor(rhs);
  const auto shape = InferBinaryShape(lhs, rhs, location);
  ValidateOutputShape(output.GetShape(), shape, name, location);
  ValidateOutputDType(output.GetDType(), DType::BOOL, name, location);
  if (lhs.GetDType() != DType::BOOL || rhs.GetDType() != DType::BOOL) {
    ThrowUnsupportedDType(name, lhs.GetDType() != DType::BOOL ? lhs.GetDType() : rhs.GetDType(), location);
  }
  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(lhs)
                            .AddInput(rhs)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_ONE_BINARY_INPUT)
                            .SetRequireSameDType(true)
                            .Build(name, location);
  if (output.GetNumElements() == 0) {
    return;
  }
  guard.RecordTensor(output);
  guard.RecordTensor(lhs);
  guard.RecordTensor(rhs);
  internal::LaunchLogicalElementwise(guard.GetNativeStream(), operation, iterator, location);
  guard.CheckLaunch();
}

[[nodiscard]] auto LogicalBinaryImpl(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs,
                                     internal::LogicalElementwiseOp operation, std::source_location location)
    -> Tensor {
  const auto name = GetLogicalName(operation);
  Shape shape;
  {
    internal::OpGuard guard{context, name, location};
    guard.ValidateTensor(lhs);
    guard.ValidateTensor(rhs);
    shape = InferBinaryShape(lhs, rhs, location);
    if (lhs.GetDType() != DType::BOOL || rhs.GetDType() != DType::BOOL) {
      ThrowUnsupportedDType(name, lhs.GetDType() != DType::BOOL ? lhs.GetDType() : rhs.GetDType(), location);
    }
  }
  auto output = Empty(context, shape, DType::BOOL, location);
  LogicalBinaryOutImpl(context, output, lhs, rhs, operation, location);
  return output;
}

}  // namespace

#define TTL_DEFINE_BINARY(name, operation)                                                                            \
  void name##Out(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,                     \
                 std::source_location location) {                                                                     \
    BinaryOutImpl(context, output, lhs, rhs, internal::BinaryElementwiseOp::operation, location);                     \
  }                                                                                                                   \
  void name##Out(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,                       \
                 std::source_location location) {                                                                     \
    ScalarBinaryOutImpl(context, output, input, scalar, internal::BinaryElementwiseOp::operation, location);          \
  }                                                                                                                   \
  auto name(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs, std::source_location location)           \
      -> Tensor {                                                                                                     \
    return BinaryImpl(context, lhs, rhs, internal::BinaryElementwiseOp::operation, location);                         \
  }                                                                                                                   \
  auto name(ExecutionContext &context, const Tensor &input, Scalar scalar, std::source_location location) -> Tensor { \
    return ScalarBinaryImpl(context, input, scalar, internal::BinaryElementwiseOp::operation, location);              \
  }

TTL_DEFINE_BINARY(Add, ADD)
TTL_DEFINE_BINARY(Subtract, SUBTRACT)
TTL_DEFINE_BINARY(Multiply, MULTIPLY)
TTL_DEFINE_BINARY(Divide, DIVIDE)
TTL_DEFINE_BINARY(Maximum, MAXIMUM)
TTL_DEFINE_BINARY(Minimum, MINIMUM)

#undef TTL_DEFINE_BINARY

#define TTL_DEFINE_COMPARISON(name, operation)                                                                        \
  void name##Out(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,                     \
                 std::source_location location) {                                                                     \
    ComparisonOutImpl(context, output, lhs, rhs, internal::ComparisonElementwiseOp::operation, location);             \
  }                                                                                                                   \
  void name##Out(ExecutionContext &context, Tensor &output, const Tensor &input, Scalar scalar,                       \
                 std::source_location location) {                                                                     \
    ScalarComparisonOutImpl(context, output, input, scalar, internal::ComparisonElementwiseOp::operation, location);  \
  }                                                                                                                   \
  auto name(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs, std::source_location location)           \
      -> Tensor {                                                                                                     \
    return ComparisonImpl(context, lhs, rhs, internal::ComparisonElementwiseOp::operation, location);                 \
  }                                                                                                                   \
  auto name(ExecutionContext &context, const Tensor &input, Scalar scalar, std::source_location location) -> Tensor { \
    return ScalarComparisonImpl(context, input, scalar, internal::ComparisonElementwiseOp::operation, location);      \
  }

TTL_DEFINE_COMPARISON(Equal, EQUAL)
TTL_DEFINE_COMPARISON(NotEqual, NOT_EQUAL)
TTL_DEFINE_COMPARISON(Less, LESS)
TTL_DEFINE_COMPARISON(LessEqual, LESS_EQUAL)
TTL_DEFINE_COMPARISON(Greater, GREATER)
TTL_DEFINE_COMPARISON(GreaterEqual, GREATER_EQUAL)

#undef TTL_DEFINE_COMPARISON

#define TTL_DEFINE_UNARY(name, operation)                                                                         \
  void name##Out(ExecutionContext &context, Tensor &output, const Tensor &input, std::source_location location) { \
    UnaryOutImpl(context, output, input, internal::UnaryElementwiseOp::operation, location);                      \
  }                                                                                                               \
  auto name(ExecutionContext &context, const Tensor &input, std::source_location location) -> Tensor {            \
    return UnaryImpl(context, input, internal::UnaryElementwiseOp::operation, location);                          \
  }

TTL_DEFINE_UNARY(Negate, NEGATE)
TTL_DEFINE_UNARY(Abs, ABS)
TTL_DEFINE_UNARY(Exp, EXP)
TTL_DEFINE_UNARY(Log, LOG)
TTL_DEFINE_UNARY(Sqrt, SQRT)
TTL_DEFINE_UNARY(Rsqrt, RSQRT)
TTL_DEFINE_UNARY(Sin, SIN)
TTL_DEFINE_UNARY(Cos, COS)
TTL_DEFINE_UNARY(Tanh, TANH)
TTL_DEFINE_UNARY(Sigmoid, SIGMOID)
TTL_DEFINE_UNARY(Relu, RELU)
TTL_DEFINE_UNARY(Silu, SILU)

#undef TTL_DEFINE_UNARY

void GeluOut(ExecutionContext &context, Tensor &output, const Tensor &input, GeluApproximation approximation,
             std::source_location location) {
  internal::OpGuard guard{context, "GeluOut", location, internal::CapturePolicy::SAFE};
  ValidateGeluApproximation(approximation, location);
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  ValidateOutputShape(output.GetShape(), input.GetShape(), "GeluOut", location);
  ValidateOutputDType(output.GetDType(), input.GetDType(), "GeluOut", location);
  ValidateFloatingDType(input.GetDType(), "GeluOut", location);
  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(input)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_UNARY)
                            .SetRequireSameDType(true)
                            .Build("GeluOut", location);
  if (output.GetNumElements() == 0) {
    return;
  }
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  internal::LaunchGeluElementwise(guard.GetNativeStream(), input.GetDType(), approximation, iterator, location);
  guard.CheckLaunch();
}

auto Gelu(ExecutionContext &context, const Tensor &input, GeluApproximation approximation,
          std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, "Gelu", location};
    ValidateGeluApproximation(approximation, location);
    guard.ValidateTensor(input);
    ValidateFloatingDType(input.GetDType(), "Gelu", location);
  }
  auto output = Empty(context, input.GetShape(), input.GetDType(), location);
  GeluOut(context, output, input, approximation, location);
  return output;
}

void ClampOut(ExecutionContext &context, Tensor &output, const Tensor &input, std::optional<Scalar> minimum,
              std::optional<Scalar> maximum, std::source_location location) {
  internal::OpGuard guard{context, "ClampOut", location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  ValidateOutputShape(output.GetShape(), input.GetShape(), "ClampOut", location);
  ValidateOutputDType(output.GetDType(), input.GetDType(), "ClampOut", location);
  ValidateFloatingDType(input.GetDType(), "ClampOut", location);
  minimum = NormalizeClampBound(minimum, input.GetDType(), location);
  maximum = NormalizeClampBound(maximum, input.GetDType(), location);
  ValidateClampBounds(minimum, maximum, location);
  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(input)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_UNARY)
                            .SetRequireSameDType(true)
                            .Build("ClampOut", location);
  if (output.GetNumElements() == 0) {
    return;
  }
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  internal::LaunchClampElementwise(guard.GetNativeStream(), input.GetDType(), iterator, minimum, maximum, location);
  guard.CheckLaunch();
}

auto Clamp(ExecutionContext &context, const Tensor &input, std::optional<Scalar> minimum, std::optional<Scalar> maximum,
           std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, "Clamp", location};
    guard.ValidateTensor(input);
    ValidateFloatingDType(input.GetDType(), "Clamp", location);
    minimum = NormalizeClampBound(minimum, input.GetDType(), location);
    maximum = NormalizeClampBound(maximum, input.GetDType(), location);
    ValidateClampBounds(minimum, maximum, location);
  }
  auto output = Empty(context, input.GetShape(), input.GetDType(), location);
  ClampOut(context, output, input, minimum, maximum, location);
  return output;
}

void LogicalAndOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                   std::source_location location) {
  LogicalBinaryOutImpl(context, output, lhs, rhs, internal::LogicalElementwiseOp::AND, location);
}

auto LogicalAnd(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs, std::source_location location)
    -> Tensor {
  return LogicalBinaryImpl(context, lhs, rhs, internal::LogicalElementwiseOp::AND, location);
}

void LogicalOrOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                  std::source_location location) {
  LogicalBinaryOutImpl(context, output, lhs, rhs, internal::LogicalElementwiseOp::OR, location);
}

auto LogicalOr(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs, std::source_location location)
    -> Tensor {
  return LogicalBinaryImpl(context, lhs, rhs, internal::LogicalElementwiseOp::OR, location);
}

void LogicalNotOut(ExecutionContext &context, Tensor &output, const Tensor &input, std::source_location location) {
  internal::OpGuard guard{context, "LogicalNotOut", location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  ValidateOutputShape(output.GetShape(), input.GetShape(), "LogicalNotOut", location);
  ValidateOutputDType(output.GetDType(), DType::BOOL, "LogicalNotOut", location);
  if (input.GetDType() != DType::BOOL) {
    ThrowUnsupportedDType("LogicalNotOut", input.GetDType(), location);
  }
  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(input)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_UNARY)
                            .SetRequireSameDType(true)
                            .Build("LogicalNotOut", location);
  if (output.GetNumElements() == 0) {
    return;
  }
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  internal::LaunchLogicalElementwise(guard.GetNativeStream(), internal::LogicalElementwiseOp::NOT, iterator, location);
  guard.CheckLaunch();
}

auto LogicalNot(ExecutionContext &context, const Tensor &input, std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, "LogicalNot", location};
    guard.ValidateTensor(input);
    if (input.GetDType() != DType::BOOL) {
      ThrowUnsupportedDType("LogicalNot", input.GetDType(), location);
    }
  }
  auto output = Empty(context, input.GetShape(), DType::BOOL, location);
  LogicalNotOut(context, output, input, location);
  return output;
}

void WhereOut(ExecutionContext &context, Tensor &output, const Tensor &condition, const Tensor &true_value,
              const Tensor &false_value, std::source_location location) {
  internal::OpGuard guard{context, "WhereOut", location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(condition);
  guard.ValidateTensor(true_value);
  guard.ValidateTensor(false_value);
  const auto shape = InferWhereShape(condition, true_value, false_value, location);
  ValidateOutputShape(output.GetShape(), shape, "WhereOut", location);
  if (condition.GetDType() != DType::BOOL) {
    ThrowUnsupportedDType("WhereOut condition", condition.GetDType(), location);
  }
  ValidateEqualDTypes(true_value.GetDType(), false_value.GetDType(), "WhereOut", location);
  ValidateOutputDType(output.GetDType(), true_value.GetDType(), "WhereOut", location);
  const auto iterator = internal::ElementwiseIterator::Builder{}
                            .AddOutput(output)
                            .AddInput(condition)
                            .AddInput(true_value)
                            .AddInput(false_value)
                            .SetAliasPolicy(internal::AliasPolicy::EXACT_ONE_WHERE_VALUE)
                            .Build("WhereOut", location);
  if (output.GetNumElements() == 0) {
    return;
  }
  guard.RecordTensor(output);
  guard.RecordTensor(condition);
  guard.RecordTensor(true_value);
  guard.RecordTensor(false_value);
  internal::LaunchWhereElementwise(guard.GetNativeStream(), output.GetDType(), iterator, location);
  guard.CheckLaunch();
}

auto Where(ExecutionContext &context, const Tensor &condition, const Tensor &true_value, const Tensor &false_value,
           std::source_location location) -> Tensor {
  Shape shape;
  DType dtype;
  {
    internal::OpGuard guard{context, "Where", location};
    guard.ValidateTensor(condition);
    guard.ValidateTensor(true_value);
    guard.ValidateTensor(false_value);
    shape = InferWhereShape(condition, true_value, false_value, location);
    if (condition.GetDType() != DType::BOOL) {
      ThrowUnsupportedDType("Where condition", condition.GetDType(), location);
    }
    ValidateEqualDTypes(true_value.GetDType(), false_value.GetDType(), "Where", location);
    dtype = true_value.GetDType();
  }
  auto output = Empty(context, shape, dtype, location);
  WhereOut(context, output, condition, true_value, false_value, location);
  return output;
}

}  // namespace ttl
