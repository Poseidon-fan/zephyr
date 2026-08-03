#include "ttl/ops/softmax.hpp"

#include <array>
#include <cstddef>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ttl/common/error.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/rowwise.hpp"
#include "ttl/internal/ops/softmax.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

[[nodiscard]] auto GetName(internal::SoftmaxOp operation) noexcept -> std::string_view {
  return operation == internal::SoftmaxOp::SOFTMAX ? "SoftmaxOut" : "LogSoftmaxOut";
}

auto ValidateAndResolveAxes(const Tensor &input, const SoftmaxOptions &options, std::source_location location)
    -> std::vector<size_t> {
  if (options.axes_.empty()) {
    throw InvalidArgumentError("softmax axes must not be empty", location);
  }
  return NormalizeAxes(options.axes_, input.GetRank(), location);
}

void ValidateDType(DType dtype, std::string_view operation, std::source_location location) {
  if (!IsFloating(dtype, location)) {
    std::string message{operation};
    message.append(" supports only floating dtypes");
    throw NotSupportedError(std::move(message), location);
  }
}

void SoftmaxOutImpl(ExecutionContext &context, Tensor &output, const Tensor &input, const SoftmaxOptions &options,
                    internal::SoftmaxOp operation, std::source_location location) {
  const auto name = GetName(operation);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  const auto axes = ValidateAndResolveAxes(input, options, location);
  ValidateDType(input.GetDType(), name, location);
  if (output.GetShape() != input.GetShape() || output.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("softmax output must match the input shape and dtype", location);
  }
  internal::ValidateWritableOutput(output, name, location);
  const std::array<const Tensor *, 1> inputs{&input};
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, output, inputs, name, location);
  if (output.GetNumElements() == 0) {
    return;
  }

  const auto &properties = internal::ContextAccess::GetDeviceProperties(context, location);
  const auto plan = internal::BuildRowwisePlan(output, input, axes, sizeof(float) * 2, properties, location);
  auto scratch_scope = guard.MakeScratchScope();
  const auto scratch = scratch_scope.AllocateBytes(plan.GetScratchBytes());
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  internal::LaunchSoftmax(guard.GetNativeStream(), input.GetDType(), operation, plan, scratch.GetData(), location);
  guard.CheckLaunch();
}

[[nodiscard]] auto SoftmaxImpl(ExecutionContext &context, const Tensor &input, const SoftmaxOptions &options,
                               internal::SoftmaxOp operation, std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, GetName(operation), location};
    guard.ValidateTensor(input);
    ValidateAndResolveAxes(input, options, location);
    ValidateDType(input.GetDType(), GetName(operation), location);
  }
  auto output = Empty(context, input.GetShape(), input.GetDType(), location);
  SoftmaxOutImpl(context, output, input, options, operation, location);
  return output;
}

}  // namespace

void SoftmaxOut(ExecutionContext &context, Tensor &output, const Tensor &input, const SoftmaxOptions &options,
                std::source_location location) {
  SoftmaxOutImpl(context, output, input, options, internal::SoftmaxOp::SOFTMAX, location);
}

auto Softmax(ExecutionContext &context, const Tensor &input, const SoftmaxOptions &options,
             std::source_location location) -> Tensor {
  return SoftmaxImpl(context, input, options, internal::SoftmaxOp::SOFTMAX, location);
}

void LogSoftmaxOut(ExecutionContext &context, Tensor &output, const Tensor &input, const SoftmaxOptions &options,
                   std::source_location location) {
  SoftmaxOutImpl(context, output, input, options, internal::SoftmaxOp::LOG_SOFTMAX, location);
}

auto LogSoftmax(ExecutionContext &context, const Tensor &input, const SoftmaxOptions &options,
                std::source_location location) -> Tensor {
  return SoftmaxImpl(context, input, options, internal::SoftmaxOp::LOG_SOFTMAX, location);
}

}  // namespace ttl
