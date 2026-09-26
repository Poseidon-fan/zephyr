#include "ttl/ops/scan.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <source_location>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/common/index_width.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/scan.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

constexpr uint64_t DEVICE_SCAN_MINIMUM_AXIS_SIZE = 4096;

[[nodiscard]] auto AddOffset(uint64_t current, uint64_t extent, uint64_t stride, uint64_t maximum) noexcept
    -> uint64_t {
  if (extent == 0 || stride == 0) {
    return current;
  }
  const auto coordinate = extent - 1;
  if (coordinate > (maximum - current) / stride) {
    return maximum + 1;
  }
  return current + (coordinate * stride);
}

[[nodiscard]] auto CanUse32BitIndexing(const internal::CumulativeSumParameters64 &parameters) noexcept -> bool {
  constexpr auto maximum = uint64_t{std::numeric_limits<uint32_t>::max()};
  if (parameters.slice_count_ > maximum || parameters.axis_size_ > maximum) {
    return false;
  }
  auto output_offset = uint64_t{0};
  auto input_offset = uint64_t{0};
  for (size_t axis = 0; axis < parameters.rank_; ++axis) {
    if (parameters.shape_[axis] > maximum || parameters.output_strides_bytes_[axis] > maximum ||
        parameters.input_strides_bytes_[axis] > maximum) {
      return false;
    }
    output_offset = AddOffset(output_offset, parameters.shape_[axis], parameters.output_strides_bytes_[axis], maximum);
    input_offset = AddOffset(input_offset, parameters.shape_[axis], parameters.input_strides_bytes_[axis], maximum);
    if (output_offset > maximum || input_offset > maximum) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] auto BuildParameters(Tensor &output, const Tensor &input, size_t axis, std::source_location location)
    -> internal::CumulativeSumParameters64 {
  const auto axis_size = input.GetShape().GetDimension(axis, location);
  auto parameters = internal::CumulativeSumParameters64{
      .output_ = static_cast<std::byte *>(internal::TensorAccess::GetMutableData(output, location)),
      .input_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(input, location)),
      .slice_count_ = static_cast<uint64_t>(axis_size == 0 ? 0 : input.GetNumElements() / axis_size),
      .axis_size_ = static_cast<uint64_t>(axis_size),
      .rank_ = static_cast<uint8_t>(input.GetRank()),
      .axis_ = static_cast<uint8_t>(axis),
  };
  const auto element_size = GetDTypeInfo(input.GetDType(), location).size_bytes_;
  for (size_t current = 0; current < input.GetRank(); ++current) {
    parameters.shape_[current] = static_cast<uint64_t>(input.GetShape().GetDimension(current, location));
    parameters.output_strides_bytes_[current] =
        internal::CheckedBytes(output.GetStrides().GetStride(current, location), element_size, location);
    parameters.input_strides_bytes_[current] =
        internal::CheckedBytes(input.GetStrides().GetStride(current, location), element_size, location);
  }
  return parameters;
}

void ValidateOutput(Tensor &output, const Tensor &input, std::source_location location) {
  if (output.GetShape() != input.GetShape() || output.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("CumulativeSumOut output shape and dtype must match the input", location);
  }
  internal::ValidateWritableOutput(output, "CumulativeSumOut", location);
  const auto alias = ClassifyAlias(output, input, location);
  if (alias != AliasKind::DISJOINT && alias != AliasKind::EXACT) {
    throw InvalidArgumentError("CumulativeSumOut supports only disjoint or exact input/output aliasing", location);
  }
}

}  // namespace

void CumulativeSumOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t raw_axis,
                      std::source_location location) {
  internal::OpGuard guard{context, "CumulativeSumOut", location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  if (input.GetDType() == DType::BOOL) {
    throw NotSupportedError("CumulativeSumOut does not support BOOL input", location);
  }
  const auto axis = NormalizeAxis(raw_axis, input.GetRank(), location);
  ValidateOutput(output, input, location);
  if (input.GetNumElements() == 0) {
    return;
  }

  const auto parameters = BuildParameters(output, input, axis, location);
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  // Long contiguous rows need multiple blocks to use the device. Other layouts retain the bounded-scratch block scan.
  if (input.GetDType() == DType::FLOAT32 && axis + 1 == input.GetRank() && input.IsContiguous() &&
      output.IsContiguous() && parameters.axis_size_ >= DEVICE_SCAN_MINIMUM_AXIS_SIZE &&
      input.GetNumElements() <= std::numeric_limits<int32_t>::max()) {
    const auto workspace_bytes = internal::GetCumulativeSumWorkspaceBytes(
        static_cast<int32_t>(input.GetNumElements()), static_cast<int32_t>(parameters.axis_size_), location);
    auto scratch = guard.MakeScratchScope();
    auto *workspace = scratch.AllocateBytes(workspace_bytes, 256, location).GetData();
    internal::LaunchCumulativeSumContiguous(guard.GetNativeStream(), parameters, workspace, workspace_bytes, location);
  } else {
    const auto index_width =
        CanUse32BitIndexing(parameters) ? internal::IndexWidth::UINT32 : internal::IndexWidth::UINT64;
    internal::LaunchCumulativeSum(guard.GetNativeStream(), input.GetDType(), index_width, parameters, location);
  }
  guard.CheckLaunch();
}

auto CumulativeSum(ExecutionContext &context, const Tensor &input, int64_t axis, std::source_location location)
    -> Tensor {
  {
    internal::OpGuard guard{context, "CumulativeSum", location};
    guard.ValidateTensor(input);
    if (input.GetDType() == DType::BOOL) {
      throw NotSupportedError("CumulativeSum does not support BOOL input", location);
    }
    static_cast<void>(NormalizeAxis(axis, input.GetRank(), location));
  }
  auto output = Empty(context, input.GetShape(), input.GetDType(), location);
  CumulativeSumOut(context, output, input, axis, location);
  return output;
}

}  // namespace ttl
