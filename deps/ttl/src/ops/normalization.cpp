#include "ttl/ops/normalization.hpp"

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
#include <vector>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/normalization.hpp"
#include "ttl/internal/ops/rowwise.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

[[nodiscard]] auto GetName(internal::NormalizationOp operation) noexcept -> std::string_view {
  return operation == internal::NormalizationOp::LAYER_NORM ? "LayerNormOut" : "RmsNormOut";
}

constexpr size_t NORMALIZATION_ACCUMULATOR_BYTES = 16;

struct NormalizationShapeInfo final {
  std::vector<size_t> axes_;
  Shape normalized_shape_;
  uint64_t group_count_;
  uint64_t reduction_count_;
};

[[nodiscard]] auto ValidateOptionsAndInferShape(const Tensor &input, const NormOptions &options,
                                                std::source_location location) -> NormalizationShapeInfo {
  if (options.normalized_rank_ <= 0 ||
      static_cast<uint64_t>(options.normalized_rank_) > static_cast<uint64_t>(input.GetRank())) {
    throw InvalidArgumentError("normalized_rank must be in [1, input rank]", location);
  }
  if (!std::isfinite(options.epsilon_) || options.epsilon_ < 0.0F) {
    throw InvalidArgumentError("normalization epsilon must be finite and non-negative", location);
  }

  const auto normalized_rank = static_cast<size_t>(options.normalized_rank_);
  const auto first_axis = input.GetRank() - normalized_rank;
  std::vector<size_t> axes(normalized_rank);
  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  auto group_count = uint64_t{1};
  auto reduction_count = uint64_t{1};
  for (size_t axis = 0; axis < input.GetRank(); ++axis) {
    const auto extent = static_cast<uint64_t>(input.GetShape().GetDimension(axis, location));
    if (axis < first_axis) {
      group_count = internal::CheckedMultiply(group_count, extent, "normalization group count", location);
    } else {
      axes[axis - first_axis] = axis;
      dimensions[axis - first_axis] = static_cast<int64_t>(extent);
      reduction_count =
          internal::CheckedMultiply(reduction_count, extent, "normalization group element count", location);
    }
  }
  return {
      .axes_ = std::move(axes),
      .normalized_shape_ = Shape{std::span<const int64_t>{dimensions.data(), normalized_rank}, location},
      .group_count_ = group_count,
      .reduction_count_ = reduction_count,
  };
}

void ValidateFloatingDType(DType dtype, std::string_view operation, std::source_location location) {
  if (!IsFloating(dtype, location)) {
    std::string message{operation};
    message.append(" supports only floating dtypes");
    throw NotSupportedError(std::move(message), location);
  }
}

void ValidateAuxiliary(const std::optional<Tensor> &tensor, const Tensor &input, const Shape &expected_shape,
                       std::string_view role, internal::OpGuard &guard, std::source_location location) {
  if (!tensor.has_value()) {
    return;
  }
  guard.ValidateTensor(*tensor);
  if (tensor->GetShape() != expected_shape || tensor->GetDType() != input.GetDType()) {
    std::string message{"normalization "};
    message.append(role);
    message.append(" must match the normalized shape and input dtype");
    throw InvalidArgumentError(std::move(message), location);
  }
}

[[nodiscard]] auto BuildAuxiliaryParameters(const std::optional<Tensor> &weight, const std::optional<Tensor> &bias,
                                            DType dtype, std::source_location location)
    -> internal::NormalizationAuxiliaryParameters64 {
  auto parameters = internal::NormalizationAuxiliaryParameters64{};
  const auto element_size = GetDTypeSize(dtype, location);
  if (weight.has_value()) {
    parameters.weight_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(*weight, location));
    for (size_t axis = 0; axis < weight->GetRank(); ++axis) {
      parameters.weight_strides_bytes_[axis] =
          internal::CheckedBytes(weight->GetStrides().GetStride(axis, location), element_size, location);
    }
  }
  if (bias.has_value()) {
    parameters.bias_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(*bias, location));
    for (size_t axis = 0; axis < bias->GetRank(); ++axis) {
      parameters.bias_strides_bytes_[axis] =
          internal::CheckedBytes(bias->GetStrides().GetStride(axis, location), element_size, location);
    }
  }
  return parameters;
}

void NormalizationOutImpl(ExecutionContext &context, Tensor &output, const Tensor &input,
                          const std::optional<Tensor> &weight, const std::optional<Tensor> &bias,
                          const NormOptions &options, internal::NormalizationOp operation,
                          std::source_location location) {
  const auto name = GetName(operation);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  ValidateFloatingDType(input.GetDType(), name, location);
  const auto shape_info = ValidateOptionsAndInferShape(input, options, location);
  ValidateAuxiliary(weight, input, shape_info.normalized_shape_, "weight", guard, location);
  ValidateAuxiliary(bias, input, shape_info.normalized_shape_, "bias", guard, location);
  if (operation == internal::NormalizationOp::RMS_NORM && bias.has_value()) {
    throw InternalError("RmsNorm does not accept bias", location);
  }
  if (output.GetShape() != input.GetShape() || output.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("normalization output must match the input shape and dtype", location);
  }
  internal::ValidateWritableOutput(output, name, location);
  std::array<const Tensor *, 3> inputs{&input, weight.has_value() ? &*weight : nullptr,
                                       bias.has_value() ? &*bias : nullptr};
  const auto input_count = size_t{1} + (weight.has_value() ? 1 : 0) + (bias.has_value() ? 1 : 0);
  if (bias.has_value() && !weight.has_value()) {
    inputs[1] = &*bias;
  }
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, output,
                          std::span<const Tensor *const>{inputs.data(), input_count}, name, location);

  if (shape_info.group_count_ != 0 && shape_info.reduction_count_ == 0) {
    throw InvalidArgumentError("normalization group must not be empty", location);
  }
  if (output.GetNumElements() == 0) {
    return;
  }

  const auto &properties = internal::ContextAccess::GetDeviceProperties(context, location);
  const auto plan = internal::BuildRowwisePlan(output, input, shape_info.axes_, NORMALIZATION_ACCUMULATOR_BYTES,
                                               properties, location);
  auto auxiliary = BuildAuxiliaryParameters(weight, bias, input.GetDType(), location);
  auxiliary.epsilon_ = options.epsilon_;
  auto scratch_scope = guard.MakeScratchScope();
  const auto scratch = scratch_scope.AllocateBytes(plan.GetScratchBytes());
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  if (weight.has_value()) {
    guard.RecordTensor(*weight);
  }
  if (bias.has_value()) {
    guard.RecordTensor(*bias);
  }
  internal::LaunchNormalization(guard.GetNativeStream(), input.GetDType(), operation, plan, auxiliary,
                                scratch.GetData(), location);
  guard.CheckLaunch();
}

[[nodiscard]] auto NormalizationImpl(ExecutionContext &context, const Tensor &input,
                                     const std::optional<Tensor> &weight, const std::optional<Tensor> &bias,
                                     const NormOptions &options, internal::NormalizationOp operation,
                                     std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, GetName(operation), location};
    guard.ValidateTensor(input);
    ValidateFloatingDType(input.GetDType(), GetName(operation), location);
    const auto shape_info = ValidateOptionsAndInferShape(input, options, location);
    ValidateAuxiliary(weight, input, shape_info.normalized_shape_, "weight", guard, location);
    ValidateAuxiliary(bias, input, shape_info.normalized_shape_, "bias", guard, location);
    if (shape_info.group_count_ != 0 && shape_info.reduction_count_ == 0) {
      throw InvalidArgumentError("normalization group must not be empty", location);
    }
  }
  auto output = Empty(context, input.GetShape(), input.GetDType(), location);
  NormalizationOutImpl(context, output, input, weight, bias, options, operation, location);
  return output;
}

}  // namespace

void LayerNormOut(ExecutionContext &context, Tensor &output, const Tensor &input, const std::optional<Tensor> &weight,
                  const std::optional<Tensor> &bias, const NormOptions &options, std::source_location location) {
  NormalizationOutImpl(context, output, input, weight, bias, options, internal::NormalizationOp::LAYER_NORM, location);
}

auto LayerNorm(ExecutionContext &context, const Tensor &input, const std::optional<Tensor> &weight,
               const std::optional<Tensor> &bias, const NormOptions &options, std::source_location location) -> Tensor {
  return NormalizationImpl(context, input, weight, bias, options, internal::NormalizationOp::LAYER_NORM, location);
}

void RmsNormOut(ExecutionContext &context, Tensor &output, const Tensor &input, const std::optional<Tensor> &weight,
                const NormOptions &options, std::source_location location) {
  NormalizationOutImpl(context, output, input, weight, std::nullopt, options, internal::NormalizationOp::RMS_NORM,
                       location);
}

auto RmsNorm(ExecutionContext &context, const Tensor &input, const std::optional<Tensor> &weight,
             const NormOptions &options, std::source_location location) -> Tensor {
  return NormalizationImpl(context, input, weight, std::nullopt, options, internal::NormalizationOp::RMS_NORM,
                           location);
}

}  // namespace ttl
