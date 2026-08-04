#include "ttl/ops/reduction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/reduction.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/internal/runtime/runtime.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

[[noreturn]] void ThrowReductionError(std::string_view operation, std::string_view reason,
                                      std::source_location location) {
  std::string message{operation};
  message.append(": ");
  message.append(reason);
  throw InvalidArgumentError(std::move(message), location);
}

[[noreturn]] void ThrowUnsupportedReductionDType(std::string_view operation, DType dtype,
                                                 std::source_location location) {
  std::string message{operation};
  message.append(" does not support dtype ");
  message.append(GetDTypeInfo(dtype, location).name_);
  throw NotSupportedError(std::move(message), location);
}

void ValidateInputDType(internal::ReductionOp operation, DType dtype, std::source_location location) {
  const auto name = internal::GetReductionName(operation);
  switch (operation) {
    case internal::ReductionOp::SUM:
      if (dtype == DType::INT32 || dtype == DType::INT64 || IsFloating(dtype, location)) {
        return;
      }
      break;
    case internal::ReductionOp::MEAN:
      if (IsFloating(dtype, location)) {
        return;
      }
      break;
    case internal::ReductionOp::MINIMUM:
    case internal::ReductionOp::MAXIMUM:
    case internal::ReductionOp::ARG_MIN:
    case internal::ReductionOp::ARG_MAX:
      if (dtype == DType::UINT8 || dtype == DType::INT32 || dtype == DType::INT64 || IsFloating(dtype, location)) {
        return;
      }
      break;
    case internal::ReductionOp::ANY:
    case internal::ReductionOp::ALL:
      if (dtype == DType::BOOL) {
        return;
      }
      break;
  }
  ThrowUnsupportedReductionDType(name, dtype, location);
}

[[nodiscard]] auto GetOutputDType(internal::ReductionOp operation, DType input_dtype) noexcept -> DType {
  switch (operation) {
    case internal::ReductionOp::ARG_MIN:
    case internal::ReductionOp::ARG_MAX:
      return DType::INT64;
    case internal::ReductionOp::ANY:
    case internal::ReductionOp::ALL:
      return DType::BOOL;
    case internal::ReductionOp::SUM:
    case internal::ReductionOp::MEAN:
    case internal::ReductionOp::MINIMUM:
    case internal::ReductionOp::MAXIMUM:
      return input_dtype;
  }
  return input_dtype;
}

[[nodiscard]] auto GetReductionCount(const Shape &shape, std::span<const size_t> axes, std::source_location location)
    -> uint64_t {
  auto count = uint64_t{1};
  for (const auto axis : axes) {
    count = internal::CheckedMultiply(count, static_cast<uint64_t>(shape.GetDimension(axis, location)),
                                      "reduction group element count", location);
  }
  return count;
}

void ValidateOutput(Tensor &output, const Tensor &input, internal::ReductionOp operation, const Shape &expected_shape,
                    std::source_location location) {
  const auto name = internal::GetReductionName(operation);
  if (output.GetShape() != expected_shape) {
    ThrowReductionError(name, "output shape does not match the inferred reduction shape", location);
  }
  if (output.GetDType() != GetOutputDType(operation, input.GetDType())) {
    ThrowReductionError(name, "output dtype does not match the inferred reduction dtype", location);
  }
  internal::ValidateWritableOutput(output, name, location);
  const std::array<const Tensor *, 1> inputs{&input};
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, output, inputs, name, location);
}

void LaunchReduction(ExecutionContext &context, internal::OpGuard &guard, Tensor &output, const Tensor &input,
                     internal::ReductionOp operation, std::span<const size_t> axes, std::source_location location) {
  const auto &properties = internal::ContextAccess::GetDeviceContext(context, location)->GetProperties();
  const auto plan = internal::BuildReductionPlan(output, input, axes, operation, properties, location);
  auto scratch_scope = guard.MakeScratchScope();
  const auto scratch = scratch_scope.AllocateBytes(plan.GetScratchBytes());

  guard.RecordTensor(output);
  guard.RecordTensor(input);
  switch (operation) {
    case internal::ReductionOp::SUM:
    case internal::ReductionOp::MEAN:
      internal::LaunchSummationReduction(guard.GetNativeStream(), input.GetDType(), operation, plan, scratch.GetData(),
                                         location);
      break;
    case internal::ReductionOp::MINIMUM:
    case internal::ReductionOp::MAXIMUM:
    case internal::ReductionOp::ARG_MIN:
    case internal::ReductionOp::ARG_MAX:
      internal::LaunchExtremaReduction(guard.GetNativeStream(), input.GetDType(), operation, plan, scratch.GetData(),
                                       location);
      break;
    case internal::ReductionOp::ANY:
    case internal::ReductionOp::ALL:
      internal::LaunchLogicalReduction(guard.GetNativeStream(), operation, plan, scratch.GetData(), location);
      break;
  }
  guard.CheckLaunch();
}

void ReductionOutImpl(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options,
                      internal::ReductionOp operation, std::source_location location) {
  const auto name = internal::GetReductionName(operation);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  ValidateInputDType(operation, input.GetDType(), location);

  const auto axes = internal::ResolveReductionAxes(options.axes_, input.GetRank(), location);
  const auto expected_shape = internal::InferReductionShape(input.GetShape(), axes, options.keep_dimensions_, location);
  ValidateOutput(output, input, operation, expected_shape, location);

  if (output.GetNumElements() == 0) {
    return;
  }
  const auto reduction_count = GetReductionCount(input.GetShape(), axes, location);
  if (reduction_count == 0 &&
      (operation == internal::ReductionOp::MINIMUM || operation == internal::ReductionOp::MAXIMUM ||
       operation == internal::ReductionOp::ARG_MIN || operation == internal::ReductionOp::ARG_MAX)) {
    ThrowReductionError(name, "cannot reduce an empty group without an identity", location);
  }
  LaunchReduction(context, guard, output, input, operation, axes, location);
}

[[nodiscard]] auto ReductionImpl(ExecutionContext &context, const Tensor &input, const ReductionOptions &options,
                                 internal::ReductionOp operation, std::source_location location) -> Tensor {
  Shape output_shape;
  DType output_dtype;
  {
    internal::OpGuard guard{context, internal::GetReductionName(operation), location};
    guard.ValidateTensor(input);
    ValidateInputDType(operation, input.GetDType(), location);
    const auto axes = internal::ResolveReductionAxes(options.axes_, input.GetRank(), location);
    output_shape = internal::InferReductionShape(input.GetShape(), axes, options.keep_dimensions_, location);
    output_dtype = GetOutputDType(operation, input.GetDType());
    if (output_shape.GetNumElements() != 0 && GetReductionCount(input.GetShape(), axes, location) == 0 &&
        (operation == internal::ReductionOp::MINIMUM || operation == internal::ReductionOp::MAXIMUM ||
         operation == internal::ReductionOp::ARG_MIN || operation == internal::ReductionOp::ARG_MAX)) {
      ThrowReductionError(internal::GetReductionName(operation), "cannot reduce an empty group without an identity",
                          location);
    }
  }

  auto output = Empty(context, output_shape, output_dtype, location);
  ReductionOutImpl(context, output, input, options, operation, location);
  return output;
}

[[nodiscard]] auto MakeArgOptions(int64_t axis, bool keep_dimension) -> ReductionOptions {
  return ReductionOptions{
      .axes_ = {axis},
      .keep_dimensions_ = keep_dimension,
  };
}

}  // namespace

#define TTL_DEFINE_REDUCTION(name, operation)                                                                     \
  void name##Out(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options, \
                 std::source_location location) {                                                                 \
    ReductionOutImpl(context, output, input, options, internal::ReductionOp::operation, location);                \
  }                                                                                                               \
  auto name(ExecutionContext &context, const Tensor &input, const ReductionOptions &options,                      \
            std::source_location location) -> Tensor {                                                            \
    return ReductionImpl(context, input, options, internal::ReductionOp::operation, location);                    \
  }

TTL_DEFINE_REDUCTION(Sum, SUM)
TTL_DEFINE_REDUCTION(Mean, MEAN)
TTL_DEFINE_REDUCTION(Minimum, MINIMUM)
TTL_DEFINE_REDUCTION(Maximum, MAXIMUM)
TTL_DEFINE_REDUCTION(Any, ANY)
TTL_DEFINE_REDUCTION(All, ALL)

#undef TTL_DEFINE_REDUCTION

void ArgMinOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis, bool keep_dimension,
               std::source_location location) {
  ReductionOutImpl(context, output, input, MakeArgOptions(axis, keep_dimension), internal::ReductionOp::ARG_MIN,
                   location);
}

auto ArgMin(ExecutionContext &context, const Tensor &input, int64_t axis, bool keep_dimension,
            std::source_location location) -> Tensor {
  return ReductionImpl(context, input, MakeArgOptions(axis, keep_dimension), internal::ReductionOp::ARG_MIN, location);
}

void ArgMaxOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis, bool keep_dimension,
               std::source_location location) {
  ReductionOutImpl(context, output, input, MakeArgOptions(axis, keep_dimension), internal::ReductionOp::ARG_MAX,
                   location);
}

auto ArgMax(ExecutionContext &context, const Tensor &input, int64_t axis, bool keep_dimension,
            std::source_location location) -> Tensor {
  return ReductionImpl(context, input, MakeArgOptions(axis, keep_dimension), internal::ReductionOp::ARG_MAX, location);
}

}  // namespace ttl
