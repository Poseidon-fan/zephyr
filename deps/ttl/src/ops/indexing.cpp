#include "ttl/ops/indexing.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/common/index_width.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/indexing.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

enum class IndexingKind : uint8_t {
  INDEX_SELECT,
  GATHER,
  TAKE_ALONG_DIMENSION,
};

[[nodiscard]] auto GetName(IndexingKind kind) noexcept -> std::string_view {
  switch (kind) {
    case IndexingKind::INDEX_SELECT:
      return "IndexSelectOut";
    case IndexingKind::GATHER:
      return "GatherOut";
    case IndexingKind::TAKE_ALONG_DIMENSION:
      return "TakeAlongDimensionOut";
  }
  return "<invalid indexing operation>";
}

void ValidateIndexDType(DType dtype, std::string_view operation, std::source_location location) {
  if (dtype != DType::INT32 && dtype != DType::INT64) {
    std::string message{operation};
    message.append(" requires INT32 or INT64 indices");
    throw NotSupportedError(std::move(message), location);
  }
}

[[nodiscard]] auto InferIndexingShape(const Tensor &input, const Tensor &index, size_t axis, IndexingKind kind,
                                      std::source_location location) -> Shape {
  if (kind == IndexingKind::INDEX_SELECT) {
    if (index.GetRank() != 1) {
      throw InvalidArgumentError("IndexSelect index must have rank 1", location);
    }
    std::array<int64_t, TTL_MAX_RANK> dimensions{};
    for (size_t current = 0; current < input.GetRank(); ++current) {
      dimensions[current] = current == axis ? index.GetNumElements() : input.GetShape().GetDimension(current, location);
    }
    return Shape{std::span<const int64_t>{dimensions.data(), input.GetRank()}, location};
  }

  if (index.GetRank() != input.GetRank()) {
    throw InvalidArgumentError("Gather and TakeAlongDimension require input and index with equal rank", location);
  }
  if (kind == IndexingKind::GATHER) {
    for (size_t current = 0; current < input.GetRank(); ++current) {
      if (current != axis &&
          index.GetShape().GetDimension(current, location) > input.GetShape().GetDimension(current, location)) {
        throw InvalidArgumentError("Gather index shape exceeds the input on a non-indexed axis", location);
      }
    }
    return index.GetShape();
  }

  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  for (size_t current = 0; current < input.GetRank(); ++current) {
    const auto input_extent = input.GetShape().GetDimension(current, location);
    const auto index_extent = index.GetShape().GetDimension(current, location);
    if (current == axis) {
      dimensions[current] = index_extent;
    } else if (input_extent == index_extent || input_extent == 1 || index_extent == 1) {
      dimensions[current] = std::max(input_extent, index_extent);
    } else {
      throw InvalidArgumentError("TakeAlongDimension non-axis dimensions are not broadcastable", location);
    }
  }
  return Shape{std::span<const int64_t>{dimensions.data(), input.GetRank()}, location};
}

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

[[nodiscard]] auto CanUse32BitIndexing(const internal::IndexingParameters64 &parameters) noexcept -> bool {
  constexpr auto maximum = uint64_t{std::numeric_limits<uint32_t>::max()};
  if (parameters.num_elements_ > maximum) {
    return false;
  }
  auto output_offset = uint64_t{0};
  auto input_offset = uint64_t{0};
  auto index_offset = uint64_t{0};
  for (size_t axis = 0; axis < parameters.rank_; ++axis) {
    if (parameters.shape_[axis] > maximum || parameters.output_strides_bytes_[axis] > maximum ||
        parameters.input_strides_bytes_[axis] > maximum || parameters.index_strides_bytes_[axis] > maximum) {
      return false;
    }
    output_offset = AddOffset(output_offset, parameters.shape_[axis], parameters.output_strides_bytes_[axis], maximum);
    index_offset = AddOffset(index_offset, parameters.shape_[axis], parameters.index_strides_bytes_[axis], maximum);
    if (axis != parameters.axis_) {
      input_offset = AddOffset(input_offset, parameters.shape_[axis], parameters.input_strides_bytes_[axis], maximum);
    }
    if (output_offset > maximum || input_offset > maximum || index_offset > maximum) {
      return false;
    }
  }
  input_offset = AddOffset(input_offset, static_cast<uint64_t>(parameters.axis_bound_),
                           parameters.input_strides_bytes_[parameters.axis_], maximum);
  return input_offset <= maximum;
}

[[nodiscard]] auto BuildIndexingParameters(Tensor &output, const Tensor &input, const Tensor &index, size_t axis,
                                           IndexingKind kind, std::source_location location)
    -> internal::IndexingParameters64 {
  auto parameters = internal::IndexingParameters64{
      .output_ = static_cast<std::byte *>(internal::TensorAccess::GetMutableData(output, location)),
      .input_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(input, location)),
      .indices_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(index, location)),
      .num_elements_ = static_cast<uint64_t>(output.GetNumElements()),
      .axis_bound_ = input.GetShape().GetDimension(axis, location),
      .rank_ = static_cast<uint8_t>(output.GetRank()),
      .axis_ = static_cast<uint8_t>(axis),
  };
  const auto output_element_size = GetDTypeSize(output.GetDType(), location);
  const auto input_element_size = GetDTypeSize(input.GetDType(), location);
  const auto index_element_size = GetDTypeSize(index.GetDType(), location);
  for (size_t current = 0; current < output.GetRank(); ++current) {
    const auto output_extent = output.GetShape().GetDimension(current, location);
    parameters.shape_[current] = static_cast<uint64_t>(output_extent);
    parameters.output_strides_bytes_[current] =
        internal::CheckedBytes(output.GetStrides().GetStride(current, location), output_element_size, location);
    parameters.input_strides_bytes_[current] =
        internal::CheckedBytes(input.GetStrides().GetStride(current, location), input_element_size, location);

    if (kind == IndexingKind::INDEX_SELECT) {
      parameters.index_strides_bytes_[current] =
          current == axis
              ? internal::CheckedBytes(index.GetStrides().GetStride(0, location), index_element_size, location)
              : 0;
    } else {
      const auto index_extent = index.GetShape().GetDimension(current, location);
      parameters.index_strides_bytes_[current] =
          kind == IndexingKind::TAKE_ALONG_DIMENSION && current != axis && index_extent == 1 && output_extent != 1
              ? 0
              : internal::CheckedBytes(index.GetStrides().GetStride(current, location), index_element_size, location);
      const auto input_extent = input.GetShape().GetDimension(current, location);
      if (kind == IndexingKind::TAKE_ALONG_DIMENSION && current != axis && input_extent == 1 && output_extent != 1) {
        parameters.input_strides_bytes_[current] = 0;
      }
    }
  }
  return parameters;
}

void ValidateIndexingOutput(Tensor &output, const Tensor &input, const Tensor &index, const Shape &shape,
                            std::string_view operation, std::source_location location) {
  if (output.GetShape() != shape || output.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("indexing output shape or dtype does not match inference", location);
  }
  internal::ValidateWritableOutput(output, operation, location);
  const std::array<const Tensor *, 2> inputs{&input, &index};
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, output, inputs, operation, location);
}

void IndexingOutImpl(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t raw_axis,
                     const Tensor &index, IndexingKind kind, std::source_location location) {
  const auto name = GetName(kind);
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  guard.ValidateTensor(index);
  ValidateIndexDType(index.GetDType(), name, location);
  const auto axis = NormalizeAxis(raw_axis, input.GetRank(), location);
  const auto shape = InferIndexingShape(input, index, axis, kind, location);
  ValidateIndexingOutput(output, input, index, shape, name, location);
  if (output.GetNumElements() == 0) {
    return;
  }

  const auto parameters = BuildIndexingParameters(output, input, index, axis, kind, location);
  const auto index_width =
      CanUse32BitIndexing(parameters) ? internal::IndexWidth::UINT32 : internal::IndexWidth::UINT64;
  const auto error_context = guard.RegisterDeviceError(index.GetDType(), input.GetDType());
  guard.RecordTensor(output);
  guard.RecordTensor(input);
  guard.RecordTensor(index);
  internal::LaunchIndexing(guard.GetNativeStream(), input.GetDType(), index.GetDType(), index_width, parameters,
                           error_context, location);
  guard.CheckLaunch();
}

[[nodiscard]] auto IndexingImpl(ExecutionContext &context, const Tensor &input, int64_t raw_axis, const Tensor &index,
                                IndexingKind kind, std::source_location location) -> Tensor {
  Shape shape;
  {
    internal::OpGuard guard{context, GetName(kind), location};
    guard.ValidateTensor(input);
    guard.ValidateTensor(index);
    ValidateIndexDType(index.GetDType(), GetName(kind), location);
    const auto axis = NormalizeAxis(raw_axis, input.GetRank(), location);
    shape = InferIndexingShape(input, index, axis, kind, location);
  }
  auto output = Empty(context, shape, input.GetDType(), location);
  IndexingOutImpl(context, output, input, raw_axis, index, kind, location);
  return output;
}

[[nodiscard]] auto InferGatherRowsShape(const Tensor &table, const Tensor &indices, std::source_location location)
    -> Shape {
  if (table.GetRank() < 2) {
    throw InvalidArgumentError("GatherRows table must have rank at least 2", location);
  }
  const auto output_rank = indices.GetRank() + table.GetRank() - 1;
  if (output_rank > TTL_MAX_RANK) {
    throw InvalidArgumentError("GatherRows output rank exceeds TTL_MAX_RANK", location);
  }
  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  auto output_axis = size_t{0};
  for (const auto extent : indices.GetShape().GetDimensions()) {
    dimensions[output_axis++] = extent;
  }
  for (size_t axis = 1; axis < table.GetRank(); ++axis) {
    dimensions[output_axis++] = table.GetShape().GetDimension(axis, location);
  }
  return Shape{std::span<const int64_t>{dimensions.data(), output_rank}, location};
}

[[nodiscard]] auto CanUse32BitIndexing(const internal::GatherRowsParameters64 &parameters) noexcept -> bool {
  constexpr auto maximum = uint64_t{std::numeric_limits<uint32_t>::max()};
  if (parameters.num_elements_ > maximum || parameters.table_row_stride_bytes_ > maximum) {
    return false;
  }
  auto output_offset = uint64_t{0};
  auto index_offset = uint64_t{0};
  auto table_offset =
      AddOffset(0, static_cast<uint64_t>(parameters.row_count_), parameters.table_row_stride_bytes_, maximum);
  for (size_t axis = 0; axis < parameters.output_rank_; ++axis) {
    if (parameters.output_shape_[axis] > maximum || parameters.output_strides_bytes_[axis] > maximum) {
      return false;
    }
    output_offset =
        AddOffset(output_offset, parameters.output_shape_[axis], parameters.output_strides_bytes_[axis], maximum);
    if (axis < parameters.index_rank_) {
      if (parameters.index_strides_bytes_[axis] > maximum) {
        return false;
      }
      index_offset =
          AddOffset(index_offset, parameters.output_shape_[axis], parameters.index_strides_bytes_[axis], maximum);
    } else {
      const auto tail_axis = axis - parameters.index_rank_;
      if (parameters.table_tail_strides_bytes_[tail_axis] > maximum) {
        return false;
      }
      table_offset = AddOffset(table_offset, parameters.output_shape_[axis],
                               parameters.table_tail_strides_bytes_[tail_axis], maximum);
    }
  }
  return output_offset <= maximum && index_offset <= maximum && table_offset <= maximum;
}

[[nodiscard]] auto BuildGatherRowsParameters(Tensor &output, const Tensor &table, const Tensor &indices,
                                             std::source_location location) -> internal::GatherRowsParameters64 {
  auto parameters = internal::GatherRowsParameters64{
      .output_ = static_cast<std::byte *>(internal::TensorAccess::GetMutableData(output, location)),
      .table_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(table, location)),
      .indices_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(indices, location)),
      .table_row_stride_bytes_ = internal::CheckedBytes(table.GetStrides().GetStride(0, location),
                                                        GetDTypeSize(table.GetDType(), location), location),
      .num_elements_ = static_cast<uint64_t>(output.GetNumElements()),
      .row_count_ = table.GetShape().GetDimension(0, location),
      .output_rank_ = static_cast<uint8_t>(output.GetRank()),
      .index_rank_ = static_cast<uint8_t>(indices.GetRank()),
  };
  const auto output_element_size = GetDTypeSize(output.GetDType(), location);
  const auto table_element_size = GetDTypeSize(table.GetDType(), location);
  const auto index_element_size = GetDTypeSize(indices.GetDType(), location);
  for (size_t axis = 0; axis < output.GetRank(); ++axis) {
    parameters.output_shape_[axis] = static_cast<uint64_t>(output.GetShape().GetDimension(axis, location));
    parameters.output_strides_bytes_[axis] =
        internal::CheckedBytes(output.GetStrides().GetStride(axis, location), output_element_size, location);
  }
  for (size_t axis = 0; axis < indices.GetRank(); ++axis) {
    parameters.index_strides_bytes_[axis] =
        internal::CheckedBytes(indices.GetStrides().GetStride(axis, location), index_element_size, location);
  }
  for (size_t axis = 1; axis < table.GetRank(); ++axis) {
    parameters.table_tail_strides_bytes_[axis - 1] =
        internal::CheckedBytes(table.GetStrides().GetStride(axis, location), table_element_size, location);
  }
  return parameters;
}

void GatherRowsOutImpl(ExecutionContext &context, Tensor &output, const Tensor &table, const Tensor &indices,
                       std::string_view operation, std::source_location location) {
  internal::OpGuard guard{context, operation, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(table);
  guard.ValidateTensor(indices);
  ValidateIndexDType(indices.GetDType(), operation, location);
  const auto shape = InferGatherRowsShape(table, indices, location);
  ValidateIndexingOutput(output, table, indices, shape, operation, location);
  if (output.GetNumElements() == 0) {
    return;
  }

  const auto parameters = BuildGatherRowsParameters(output, table, indices, location);
  const auto index_width =
      CanUse32BitIndexing(parameters) ? internal::IndexWidth::UINT32 : internal::IndexWidth::UINT64;
  const auto error_context = guard.RegisterDeviceError(indices.GetDType(), table.GetDType());
  guard.RecordTensor(output);
  guard.RecordTensor(table);
  guard.RecordTensor(indices);
  internal::LaunchGatherRows(guard.GetNativeStream(), table.GetDType(), indices.GetDType(), index_width, parameters,
                             error_context, location);
  guard.CheckLaunch();
}

[[nodiscard]] auto GatherRowsImpl(ExecutionContext &context, const Tensor &table, const Tensor &indices,
                                  std::string_view operation, std::source_location location) -> Tensor {
  Shape shape;
  {
    internal::OpGuard guard{context, operation, location};
    guard.ValidateTensor(table);
    guard.ValidateTensor(indices);
    ValidateIndexDType(indices.GetDType(), operation, location);
    shape = InferGatherRowsShape(table, indices, location);
  }
  auto output = Empty(context, shape, table.GetDType(), location);
  GatherRowsOutImpl(context, output, table, indices, operation, location);
  return output;
}

void ValidateScatterElementsInputs(const Tensor &input, const Tensor &index, const Tensor &source, size_t axis,
                                   std::string_view operation, std::source_location location) {
  ValidateIndexDType(index.GetDType(), operation, location);
  if (source.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("ScatterElementsOut source and input dtypes must match", location);
  }
  if (index.GetRank() != input.GetRank() || source.GetRank() != input.GetRank()) {
    throw InvalidArgumentError("ScatterElementsOut input, index, and source must have equal rank", location);
  }
  for (size_t current = 0; current < input.GetRank(); ++current) {
    const auto index_extent = index.GetShape().GetDimension(current, location);
    if (index_extent > source.GetShape().GetDimension(current, location)) {
      throw InvalidArgumentError("ScatterElementsOut index shape exceeds source shape", location);
    }
    if (current != axis && index_extent > input.GetShape().GetDimension(current, location)) {
      throw InvalidArgumentError("ScatterElementsOut index shape exceeds input on a non-indexed axis", location);
    }
  }
}

void ValidateScatterElementsOutput(Tensor &output, const Tensor &input, const Tensor &index, const Tensor &source,
                                   std::source_location location) {
  if (output.GetShape() != input.GetShape() || output.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("ScatterElementsOut output shape and dtype must match the input", location);
  }
  internal::ValidateWritableOutput(output, "ScatterElementsOut", location);
  const auto input_alias = ClassifyAlias(output, input, location);
  if (input_alias != AliasKind::DISJOINT && input_alias != AliasKind::EXACT) {
    throw InvalidArgumentError("ScatterElementsOut supports only disjoint or exact input/output aliasing", location);
  }
  if (ClassifyAlias(output, index, location) != AliasKind::DISJOINT ||
      ClassifyAlias(output, source, location) != AliasKind::DISJOINT) {
    throw InvalidArgumentError("ScatterElementsOut output must not overlap index or source", location);
  }
}

[[nodiscard]] auto BuildScatterElementsParameters(Tensor &output, const Tensor &index, const Tensor &source,
                                                  size_t axis, std::source_location location)
    -> internal::ScatterElementsParameters64 {
  auto parameters = internal::ScatterElementsParameters64{
      .output_ = static_cast<std::byte *>(internal::TensorAccess::GetMutableData(output, location)),
      .source_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(source, location)),
      .indices_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(index, location)),
      .num_elements_ = static_cast<uint64_t>(index.GetNumElements()),
      .axis_bound_ = output.GetShape().GetDimension(axis, location),
      .rank_ = static_cast<uint8_t>(output.GetRank()),
      .axis_ = static_cast<uint8_t>(axis),
  };
  const auto value_size = GetDTypeSize(output.GetDType(), location);
  const auto index_size = GetDTypeSize(index.GetDType(), location);
  for (size_t current = 0; current < output.GetRank(); ++current) {
    parameters.shape_[current] = static_cast<uint64_t>(index.GetShape().GetDimension(current, location));
    parameters.output_strides_bytes_[current] =
        internal::CheckedBytes(output.GetStrides().GetStride(current, location), value_size, location);
    parameters.source_strides_bytes_[current] =
        internal::CheckedBytes(source.GetStrides().GetStride(current, location), value_size, location);
    parameters.index_strides_bytes_[current] =
        internal::CheckedBytes(index.GetStrides().GetStride(current, location), index_size, location);
  }
  return parameters;
}

[[nodiscard]] auto CanUse32BitIndexing(const internal::ScatterElementsParameters64 &parameters) noexcept -> bool {
  constexpr auto maximum = uint64_t{std::numeric_limits<uint32_t>::max()};
  if (parameters.num_elements_ > maximum) {
    return false;
  }
  auto output_offset = uint64_t{0};
  auto source_offset = uint64_t{0};
  auto index_offset = uint64_t{0};
  for (size_t axis = 0; axis < parameters.rank_; ++axis) {
    if (parameters.shape_[axis] > maximum || parameters.output_strides_bytes_[axis] > maximum ||
        parameters.source_strides_bytes_[axis] > maximum || parameters.index_strides_bytes_[axis] > maximum) {
      return false;
    }
    source_offset = AddOffset(source_offset, parameters.shape_[axis], parameters.source_strides_bytes_[axis], maximum);
    index_offset = AddOffset(index_offset, parameters.shape_[axis], parameters.index_strides_bytes_[axis], maximum);
    if (axis != parameters.axis_) {
      output_offset =
          AddOffset(output_offset, parameters.shape_[axis], parameters.output_strides_bytes_[axis], maximum);
    }
    if (output_offset > maximum || source_offset > maximum || index_offset > maximum) {
      return false;
    }
  }
  output_offset = AddOffset(output_offset, static_cast<uint64_t>(parameters.axis_bound_),
                            parameters.output_strides_bytes_[parameters.axis_], maximum);
  return output_offset <= maximum;
}

}  // namespace

void IndexSelectOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis, const Tensor &index,
                    std::source_location location) {
  IndexingOutImpl(context, output, input, axis, index, IndexingKind::INDEX_SELECT, location);
}

auto IndexSelect(ExecutionContext &context, const Tensor &input, int64_t axis, const Tensor &index,
                 std::source_location location) -> Tensor {
  return IndexingImpl(context, input, axis, index, IndexingKind::INDEX_SELECT, location);
}

void GatherOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis, const Tensor &index,
               std::source_location location) {
  IndexingOutImpl(context, output, input, axis, index, IndexingKind::GATHER, location);
}

auto Gather(ExecutionContext &context, const Tensor &input, int64_t axis, const Tensor &index,
            std::source_location location) -> Tensor {
  return IndexingImpl(context, input, axis, index, IndexingKind::GATHER, location);
}

void TakeAlongDimensionOut(ExecutionContext &context, Tensor &output, const Tensor &input, const Tensor &index,
                           int64_t axis, std::source_location location) {
  IndexingOutImpl(context, output, input, axis, index, IndexingKind::TAKE_ALONG_DIMENSION, location);
}

auto TakeAlongDimension(ExecutionContext &context, const Tensor &input, const Tensor &index, int64_t axis,
                        std::source_location location) -> Tensor {
  return IndexingImpl(context, input, axis, index, IndexingKind::TAKE_ALONG_DIMENSION, location);
}

void ScatterElementsOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t raw_axis,
                        const Tensor &index, const Tensor &source, std::source_location location) {
  size_t axis = 0;
  AliasKind input_alias = AliasKind::DISJOINT;
  {
    internal::OpGuard guard{context, "ScatterElementsOut", location, internal::CapturePolicy::SAFE};
    guard.ValidateTensor(output);
    guard.ValidateTensor(input);
    guard.ValidateTensor(index);
    guard.ValidateTensor(source);
    axis = NormalizeAxis(raw_axis, input.GetRank(), location);
    ValidateScatterElementsInputs(input, index, source, axis, "ScatterElementsOut", location);
    ValidateScatterElementsOutput(output, input, index, source, location);
    input_alias = ClassifyAlias(output, input, location);
  }
  if (input_alias == AliasKind::DISJOINT) {
    CopyOut(context, output, input, location);
  }
  if (index.GetNumElements() == 0) {
    return;
  }

  internal::OpGuard guard{context, "ScatterElementsOut", location, internal::CapturePolicy::SAFE};
  const auto parameters = BuildScatterElementsParameters(output, index, source, axis, location);
  const auto index_width =
      CanUse32BitIndexing(parameters) ? internal::IndexWidth::UINT32 : internal::IndexWidth::UINT64;
  const auto error_context = guard.RegisterDeviceError(index.GetDType(), output.GetDType());
  guard.RecordTensor(output);
  guard.RecordTensor(index);
  guard.RecordTensor(source);
  internal::LaunchScatterElements(guard.GetNativeStream(), output.GetDType(), index.GetDType(), index_width, parameters,
                                  error_context, location);
  guard.CheckLaunch();
}

auto ScatterElements(ExecutionContext &context, const Tensor &input, int64_t axis, const Tensor &index,
                     const Tensor &source, std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, "ScatterElements", location};
    guard.ValidateTensor(input);
    guard.ValidateTensor(index);
    guard.ValidateTensor(source);
    const auto normalized_axis = NormalizeAxis(axis, input.GetRank(), location);
    ValidateScatterElementsInputs(input, index, source, normalized_axis, "ScatterElements", location);
  }
  auto output = Empty(context, input.GetShape(), input.GetDType(), location);
  ScatterElementsOut(context, output, input, axis, index, source, location);
  return output;
}

void GatherRowsOut(ExecutionContext &context, Tensor &output, const Tensor &table, const Tensor &indices,
                   std::source_location location) {
  GatherRowsOutImpl(context, output, table, indices, "GatherRowsOut", location);
}

auto GatherRows(ExecutionContext &context, const Tensor &table, const Tensor &indices, std::source_location location)
    -> Tensor {
  return GatherRowsImpl(context, table, indices, "GatherRowsOut", location);
}

void EmbeddingOut(ExecutionContext &context, Tensor &output, const Tensor &weight, const Tensor &token_ids,
                  std::source_location location) {
  GatherRowsOutImpl(context, output, weight, token_ids, "EmbeddingOut", location);
}

auto Embedding(ExecutionContext &context, const Tensor &weight, const Tensor &token_ids, std::source_location location)
    -> Tensor {
  return GatherRowsImpl(context, weight, token_ids, "EmbeddingOut", location);
}

}  // namespace ttl
