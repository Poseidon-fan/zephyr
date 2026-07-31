#include "ttl/ops/composition.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <string_view>
#include <vector>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/composition.hpp"
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/layout.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

enum class CompositionKind : uint8_t {
  CONCAT,
  STACK,
};

[[nodiscard]] auto GetName(CompositionKind kind) noexcept -> std::string_view {
  return kind == CompositionKind::CONCAT ? "ConcatOut" : "StackOut";
}

struct CompositionShapeInfo final {
  Shape shape_;
  size_t axis_;
};

[[nodiscard]] auto InferCompositionShape(std::span<const Tensor> inputs, int64_t raw_axis, CompositionKind kind,
                                         std::source_location location) -> CompositionShapeInfo {
  if (inputs.empty()) {
    throw InvalidArgumentError("Concat and Stack require at least one input", location);
  }
  const auto rank = inputs.front().GetRank();
  const auto axis = NormalizeAxis(raw_axis, kind == CompositionKind::CONCAT ? rank : rank + 1, location);
  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  if (kind == CompositionKind::STACK && rank == TTL_MAX_RANK) {
    throw InvalidArgumentError("Stack output rank exceeds TTL_MAX_RANK", location);
  }

  for (size_t input_index = 1; input_index < inputs.size(); ++input_index) {
    if (inputs[input_index].GetRank() != rank || inputs[input_index].GetDType() != inputs.front().GetDType()) {
      throw InvalidArgumentError("Concat and Stack inputs must have equal rank and dtype", location);
    }
  }

  if (kind == CompositionKind::STACK) {
    for (size_t input_index = 1; input_index < inputs.size(); ++input_index) {
      if (inputs[input_index].GetShape() != inputs.front().GetShape()) {
        throw InvalidArgumentError("Stack inputs must have identical shapes", location);
      }
    }
    auto input_axis = size_t{0};
    for (size_t output_axis = 0; output_axis < rank + 1; ++output_axis) {
      if (output_axis == axis) {
        dimensions[output_axis] = internal::CheckedNarrow<int64_t>(inputs.size(), "Stack input count", location);
      } else {
        dimensions[output_axis] = inputs.front().GetShape().GetDimension(input_axis++, location);
      }
    }
    return {
        .shape_ = Shape{std::span<const int64_t>{dimensions.data(), rank + 1}, location},
        .axis_ = axis,
    };
  }

  auto concatenated_extent = int64_t{0};
  for (const auto &input : inputs) {
    for (size_t current = 0; current < rank; ++current) {
      const auto extent = input.GetShape().GetDimension(current, location);
      if (current != axis && extent != inputs.front().GetShape().GetDimension(current, location)) {
        throw InvalidArgumentError("Concat input shapes differ outside the concatenation axis", location);
      }
    }
    concatenated_extent = internal::CheckedAdd(concatenated_extent, input.GetShape().GetDimension(axis, location),
                                               "Concat output extent", location);
  }
  for (size_t current = 0; current < rank; ++current) {
    dimensions[current] =
        current == axis ? concatenated_extent : inputs.front().GetShape().GetDimension(current, location);
  }
  return {
      .shape_ = Shape{std::span<const int64_t>{dimensions.data(), rank}, location},
      .axis_ = axis,
  };
}

void ValidateComposition(ExecutionContext &context, Tensor *output, std::span<const Tensor> inputs, int64_t raw_axis,
                         CompositionKind kind, CompositionShapeInfo &shape_info, std::source_location location) {
  const auto name = GetName(kind);
  internal::OpGuard guard{context, name, location};
  for (const auto &input : inputs) {
    guard.ValidateTensor(input);
  }
  shape_info = InferCompositionShape(inputs, raw_axis, kind, location);
  if (output == nullptr) {
    return;
  }
  guard.ValidateTensor(*output);
  if (output->GetShape() != shape_info.shape_ || output->GetDType() != inputs.front().GetDType()) {
    throw InvalidArgumentError("composition output shape or dtype does not match inference", location);
  }
  internal::ValidateWritableOutput(*output, name, location);
  std::vector<const Tensor *> input_pointers;
  input_pointers.reserve(inputs.size());
  for (const auto &input : inputs) {
    input_pointers.push_back(&input);
  }
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, *output, input_pointers, name, location);
}

[[nodiscard]] auto BuildCompositionParameters(Tensor &output, const Tensor &input, size_t output_axis,
                                              int64_t output_axis_offset, CompositionKind kind,
                                              std::source_location location) -> internal::CompositionParameters64 {
  const auto element_size = GetDTypeSize(input.GetDType(), location);
  const auto output_axis_stride =
      internal::CheckedBytes(output.GetStrides().GetStride(output_axis, location), element_size, location);
  const auto output_byte_offset = internal::CheckedMultiply(
      output_axis_offset,
      internal::CheckedNarrow<int64_t>(output_axis_stride, "composition output axis stride", location),
      "composition output byte offset", location);
  auto *output_data = static_cast<std::byte *>(internal::TensorAccess::GetMutableData(output, location));
  auto parameters = internal::CompositionParameters64{
      .output_ = output_data + output_byte_offset,
      .input_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(input, location)),
      .num_elements_ = static_cast<uint64_t>(input.GetNumElements()),
      .rank_ = static_cast<uint8_t>(input.GetRank()),
  };
  for (size_t input_axis = 0; input_axis < input.GetRank(); ++input_axis) {
    const auto mapped_output_axis =
        kind == CompositionKind::STACK && input_axis >= output_axis ? input_axis + 1 : input_axis;
    parameters.shape_[input_axis] = static_cast<uint64_t>(input.GetShape().GetDimension(input_axis, location));
    parameters.output_strides_bytes_[input_axis] =
        internal::CheckedBytes(output.GetStrides().GetStride(mapped_output_axis, location), element_size, location);
    parameters.input_strides_bytes_[input_axis] =
        internal::CheckedBytes(input.GetStrides().GetStride(input_axis, location), element_size, location);
  }
  return parameters;
}

void CompositionOutImpl(ExecutionContext &context, Tensor &output, std::span<const Tensor> inputs, int64_t raw_axis,
                        CompositionKind kind, std::source_location location) {
  const auto name = GetName(kind);
  internal::OpGuard guard{context, name, location};
  auto shape_info = CompositionShapeInfo{};
  guard.ValidateTensor(output);
  for (const auto &input : inputs) {
    guard.ValidateTensor(input);
  }
  shape_info = InferCompositionShape(inputs, raw_axis, kind, location);
  if (output.GetShape() != shape_info.shape_ || output.GetDType() != inputs.front().GetDType()) {
    throw InvalidArgumentError("composition output shape or dtype does not match inference", location);
  }
  internal::ValidateWritableOutput(output, name, location);
  std::vector<const Tensor *> input_pointers;
  input_pointers.reserve(inputs.size());
  for (const auto &input : inputs) {
    input_pointers.push_back(&input);
  }
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, output, input_pointers, name, location);
  if (output.GetNumElements() == 0) {
    return;
  }

  auto offset = int64_t{0};
  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    const auto &input = inputs[input_index];
    if (input.GetNumElements() != 0) {
      const auto output_axis_offset = kind == CompositionKind::STACK
                                          ? internal::CheckedNarrow<int64_t>(input_index, "Stack input index", location)
                                          : offset;
      const auto parameters =
          BuildCompositionParameters(output, input, shape_info.axis_, output_axis_offset, kind, location);
      const auto index_width = internal::GetCompositionIndexWidth(parameters);
      guard.RecordTensor(input);
      internal::LaunchCompositionCopy(guard.GetNativeStream(), output.GetDType(), index_width, parameters, location);
    }
    if (kind == CompositionKind::CONCAT) {
      offset = internal::CheckedAdd(offset, input.GetShape().GetDimension(shape_info.axis_, location),
                                    "Concat copy offset", location);
    }
  }
  guard.RecordTensor(output);
  guard.CheckLaunch();
}

[[nodiscard]] auto CompositionImpl(ExecutionContext &context, std::span<const Tensor> inputs, int64_t raw_axis,
                                   CompositionKind kind, std::source_location location) -> Tensor {
  auto shape_info = CompositionShapeInfo{};
  ValidateComposition(context, nullptr, inputs, raw_axis, kind, shape_info, location);
  auto output = Empty(context, shape_info.shape_, inputs.front().GetDType(), location);
  CompositionOutImpl(context, output, inputs, raw_axis, kind, location);
  return output;
}

}  // namespace

void ConcatOut(ExecutionContext &context, Tensor &output, std::span<const Tensor> inputs, int64_t axis,
               std::source_location location) {
  CompositionOutImpl(context, output, inputs, axis, CompositionKind::CONCAT, location);
}

auto Concat(ExecutionContext &context, std::span<const Tensor> inputs, int64_t axis, std::source_location location)
    -> Tensor {
  return CompositionImpl(context, inputs, axis, CompositionKind::CONCAT, location);
}

void StackOut(ExecutionContext &context, Tensor &output, std::span<const Tensor> inputs, int64_t axis,
              std::source_location location) {
  CompositionOutImpl(context, output, inputs, axis, CompositionKind::STACK, location);
}

auto Stack(ExecutionContext &context, std::span<const Tensor> inputs, int64_t axis, std::source_location location)
    -> Tensor {
  return CompositionImpl(context, inputs, axis, CompositionKind::STACK, location);
}

}  // namespace ttl
