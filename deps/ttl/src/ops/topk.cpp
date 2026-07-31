#include "ttl/ops/topk.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <source_location>
#include <span>
#include <utility>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/scratch_arena.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/internal/topk.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

struct TopKShapeInfo final {
  Shape shape_;
  size_t axis_;
  int64_t axis_size_;
  int64_t slice_count_;
};

[[nodiscard]] auto InferTopKShape(const Tensor &input, const TopKOptions &options, std::source_location location)
    -> TopKShapeInfo {
  if (input.GetRank() == 0) {
    throw InvalidArgumentError("TopK requires an input with rank at least one", location);
  }
  const auto axis = NormalizeAxis(options.axis_, input.GetRank(), location);
  const auto axis_size = input.GetShape().GetDimension(axis, location);
  if (options.k_ < 0 || options.k_ > axis_size) {
    throw InvalidArgumentError("TopK k must be in [0, input.shape[axis]]", location);
  }
  auto dimensions = std::array<int64_t, TTL_MAX_RANK>{};
  std::ranges::copy(input.GetShape().GetDimensions(), dimensions.begin());
  dimensions[axis] = options.k_;
  return {
      .shape_ = Shape{std::span<const int64_t>{dimensions.data(), input.GetRank()}, location},
      .axis_ = axis,
      .axis_size_ = axis_size,
      .slice_count_ = axis_size == 0 ? int64_t{0} : input.GetNumElements() / axis_size,
  };
}

void ValidateTopKOutputs(Tensor &values, Tensor &indices, const Tensor &input, const TopKShapeInfo &shape_info,
                         std::source_location location) {
  if (values.GetShape() != shape_info.shape_ || values.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("TopK values output shape or dtype does not match inference", location);
  }
  if (indices.GetShape() != shape_info.shape_ || indices.GetDType() != DType::INT64) {
    throw InvalidArgumentError("TopK indices output must have inferred shape and INT64 dtype", location);
  }
  internal::ValidateWritableOutput(values, "TopKOut", location);
  internal::ValidateWritableOutput(indices, "TopKOut", location);
  const std::array<const Tensor *, 1> input_pointer{&input};
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, values, input_pointer, "TopKOut", location);
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, indices, input_pointer, "TopKOut", location);
  if (ClassifyAlias(values, indices, location) != AliasKind::DISJOINT) {
    throw InvalidArgumentError("TopK values and indices outputs must not overlap", location);
  }
}

[[nodiscard]] auto BuildTopKParameters(Tensor &values, Tensor &indices, const Tensor &input,
                                       const TopKShapeInfo &shape_info, const TopKOptions &options,
                                       std::source_location location) -> internal::TopKParameters {
  const auto value_size = GetDTypeSize(values.GetDType(), location);
  auto parameters = internal::TopKParameters{
      .values_ = static_cast<std::byte *>(internal::TensorAccess::GetMutableData(values, location)),
      .indices_ = internal::TensorAccess::GetMutableData<int64_t>(indices, location),
      .input_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(input, location)),
      .slice_count_ = static_cast<uint64_t>(shape_info.slice_count_),
      .axis_size_ = static_cast<uint64_t>(shape_info.axis_size_),
      .k_ = static_cast<uint64_t>(options.k_),
      .rank_ = static_cast<uint8_t>(input.GetRank()),
      .axis_ = static_cast<uint8_t>(shape_info.axis_),
      .largest_ = options.largest_,
  };
  for (size_t axis = 0; axis < input.GetRank(); ++axis) {
    parameters.shape_[axis] = static_cast<uint64_t>(input.GetShape().GetDimension(axis, location));
    parameters.value_strides_bytes_[axis] =
        internal::CheckedBytes(values.GetStrides().GetStride(axis, location), value_size, location);
    parameters.index_strides_elements_[axis] = static_cast<uint64_t>(indices.GetStrides().GetStride(axis, location));
    parameters.input_strides_bytes_[axis] =
        internal::CheckedBytes(input.GetStrides().GetStride(axis, location), value_size, location);
  }
  return parameters;
}

}  // namespace

void TopKOut(ExecutionContext &context, Tensor &values, Tensor &indices, const Tensor &input,
             const TopKOptions &options, std::source_location location) {
  internal::OpGuard guard{context, "TopKOut", location};
  guard.ValidateTensor(values);
  guard.ValidateTensor(indices);
  guard.ValidateTensor(input);
  if (input.GetDType() == DType::BOOL) {
    throw NotSupportedError("TopK does not support BOOL input", location);
  }
  const auto shape_info = InferTopKShape(input, options, location);
  ValidateTopKOutputs(values, indices, input, shape_info, location);
  if (values.GetNumElements() == 0) {
    return;
  }

  const auto parameters = BuildTopKParameters(values, indices, input, shape_info, options, location);
  guard.RecordTensor(values);
  guard.RecordTensor(indices);
  guard.RecordTensor(input);
  if (shape_info.axis_size_ <= 1024) {
    internal::LaunchTopKSmall(guard.GetNativeStream(), input.GetDType(), parameters, location);
  } else if (input.GetNumElements() <= std::numeric_limits<int32_t>::max() &&
             shape_info.slice_count_ <= std::numeric_limits<int32_t>::max() &&
             shape_info.axis_size_ <= std::numeric_limits<int32_t>::max()) {
    const auto num_items = internal::CheckedNarrow<int32_t>(input.GetNumElements(), "TopK item count", location);
    const auto num_segments = internal::CheckedNarrow<int32_t>(shape_info.slice_count_, "TopK segment count", location);
    const auto axis_size = internal::CheckedNarrow<int32_t>(shape_info.axis_size_, "TopK axis size", location);
    const auto workspace_bytes =
        internal::GetTopKSortWorkspaceBytes(num_items, num_segments, axis_size, options.largest_, location);
    const auto item_bytes = internal::CheckedBytes(input.GetNumElements(), sizeof(uint64_t), location);
    auto scratch = guard.MakeScratchScope();
    auto *keys_input =
        static_cast<uint64_t *>(scratch.AllocateBytes(item_bytes, alignof(uint64_t), location).GetData());
    auto *keys_output =
        static_cast<uint64_t *>(scratch.AllocateBytes(item_bytes, alignof(uint64_t), location).GetData());
    auto *indices_input =
        static_cast<int64_t *>(scratch.AllocateBytes(item_bytes, alignof(int64_t), location).GetData());
    auto *indices_output =
        static_cast<int64_t *>(scratch.AllocateBytes(item_bytes, alignof(int64_t), location).GetData());
    auto *workspace = scratch.AllocateBytes(workspace_bytes, 256, location).GetData();
    internal::LaunchTopKSort(guard.GetNativeStream(), input.GetDType(), parameters, keys_input, keys_output,
                             indices_input, indices_output, workspace, workspace_bytes, location);
  } else {
    internal::LaunchTopKSerial(guard.GetNativeStream(), input.GetDType(), parameters, location);
  }
  guard.CheckLaunch();
}

auto TopK(ExecutionContext &context, const Tensor &input, const TopKOptions &options, std::source_location location)
    -> std::pair<Tensor, Tensor> {
  Shape shape;
  {
    internal::OpGuard guard{context, "TopK", location};
    guard.ValidateTensor(input);
    if (input.GetDType() == DType::BOOL) {
      throw NotSupportedError("TopK does not support BOOL input", location);
    }
    shape = InferTopKShape(input, options, location).shape_;
  }
  auto values = Empty(context, shape, input.GetDType(), location);
  auto indices = Empty(context, shape, DType::INT64, location);
  TopKOut(context, values, indices, input, options, location);
  return {std::move(values), std::move(indices)};
}

}  // namespace ttl
