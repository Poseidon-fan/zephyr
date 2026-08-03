#include "ttl/tensor/layout.hpp"

#include <algorithm>
#include <array>
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
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/tensor/layout.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::internal {

auto ComputeViewStrides(const Shape &old_shape, const Strides &old_strides, const Shape &new_shape,
                        std::source_location location) -> std::optional<Strides> {
  if (old_shape.GetRank() != old_strides.GetRank()) {
    throw InvalidArgumentError("view source shape and strides must have the same rank", location);
  }
  if (old_shape.GetNumElements() != new_shape.GetNumElements()) {
    return std::nullopt;
  }
  if (old_shape.IsEmpty()) {
    return GetContiguousStrides(new_shape, location);
  }

  std::array<int64_t, TTL_MAX_RANK> new_strides{};
  if (old_shape.IsScalar()) {
    std::ranges::fill(new_strides, int64_t{1});
    return Strides{std::span<const int64_t>{new_strides.data(), new_shape.GetRank()}, location};
  }

  const auto old_dimensions = old_shape.GetDimensions();
  const auto old_stride_values = old_strides.GetValues();
  const auto new_dimensions = new_shape.GetDimensions();
  auto view_axis = static_cast<int64_t>(new_shape.GetRank()) - 1;
  auto chunk_base_stride = old_stride_values.back();
  auto tensor_elements = int64_t{1};
  auto view_elements = int64_t{1};

  for (auto tensor_axis = static_cast<int64_t>(old_shape.GetRank()) - 1; tensor_axis >= 0; --tensor_axis) {
    const auto axis = static_cast<size_t>(tensor_axis);
    tensor_elements =
        CheckedMultiply(tensor_elements, old_dimensions[axis], "view source chunk element count", location);

    auto chunk_ends = tensor_axis == 0;
    if (!chunk_ends && old_dimensions[axis - 1] != 1) {
      const auto expected_stride =
          CheckedMultiply(tensor_elements, chunk_base_stride, "view source chunk stride", location);
      chunk_ends = old_stride_values[axis - 1] != expected_stride;
    }
    if (!chunk_ends) {
      continue;
    }

    while (view_axis >= 0) {
      const auto new_axis = static_cast<size_t>(view_axis);
      if (view_elements >= tensor_elements && new_dimensions[new_axis] != 1) {
        break;
      }
      new_strides[new_axis] = CheckedMultiply(view_elements, chunk_base_stride, "view destination stride", location);
      view_elements =
          CheckedMultiply(view_elements, new_dimensions[new_axis], "view destination chunk element count", location);
      --view_axis;
    }
    if (view_elements != tensor_elements) {
      return std::nullopt;
    }

    if (tensor_axis > 0) {
      chunk_base_stride = old_stride_values[axis - 1];
      tensor_elements = 1;
      view_elements = 1;
    }
  }

  if (view_axis != -1) {
    return std::nullopt;
  }
  return Strides{std::span<const int64_t>{new_strides.data(), new_shape.GetRank()}, location};
}

}  // namespace ttl::internal

namespace ttl {
namespace {

[[nodiscard]] auto MakeView(const Tensor &input, Shape shape, Strides strides, int64_t storage_offset,
                            std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  if (impl.GetShape() == shape && impl.GetStrides() == strides && impl.GetStorageOffset() == storage_offset) {
    return input;
  }
  return internal::TensorFactory::Create(impl.GetStorage(), impl.GetDType(), shape, strides, storage_offset, location);
}

[[noreturn]] void ThrowViewIncompatible(const Tensor &input, const Shape &shape, std::source_location location) {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  std::string message{"cannot view shape "};
  message.append(impl.GetShape().ToString());
  message.append(" with strides ");
  message.append(impl.GetStrides().ToString());
  message.append(" as ");
  message.append(shape.ToString());
  message.append(" without copying");
  throw InvalidArgumentError(std::move(message), location);
}

[[nodiscard]] auto NormalizeInsertedAxis(int64_t axis, size_t rank, std::source_location location) -> size_t {
  if (rank >= TTL_MAX_RANK) {
    throw InvalidArgumentError("cannot unsqueeze a tensor at TTL_MAX_RANK", location);
  }
  const auto result_rank = static_cast<int64_t>(rank + 1);
  if (axis < -result_rank || axis >= result_rank) {
    std::string message{"unsqueeze axis out of range: "};
    message.append(std::to_string(axis));
    message.append(" for result rank ");
    message.append(std::to_string(result_rank));
    throw InvalidArgumentError(std::move(message), location);
  }
  return static_cast<size_t>(axis < 0 ? axis + result_rank : axis);
}

[[nodiscard]] auto NormalizeSliceBound(int64_t bound, int64_t dimension, std::string_view description,
                                       std::source_location location) -> int64_t {
  if (bound < 0) {
    bound = internal::CheckedAdd(bound, dimension, description, location);
  }
  return std::clamp(bound, int64_t{0}, dimension);
}

[[nodiscard]] auto NormalizeIndex(int64_t index, int64_t dimension, std::string_view description,
                                  std::source_location location) -> int64_t {
  if (index < 0) {
    index = internal::CheckedAdd(index, dimension, description, location);
  }
  if (index < 0 || index >= dimension) {
    std::string message{description};
    message.append(" out of range: ");
    message.append(std::to_string(index));
    message.append(" for dimension ");
    message.append(std::to_string(dimension));
    throw InvalidArgumentError(std::move(message), location);
  }
  return index;
}

}  // namespace

auto InferReshape(const Tensor &input, std::span<const int64_t> requested, std::source_location location) -> Shape {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  if (requested.size() > TTL_MAX_RANK) {
    throw InvalidArgumentError("reshape rank exceeds TTL_MAX_RANK", location);
  }

  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  auto known_elements = int64_t{1};
  std::optional<size_t> inferred_axis;
  for (size_t axis = 0; axis < requested.size(); ++axis) {
    const auto dimension = requested[axis];
    if (dimension == -1) {
      if (inferred_axis.has_value()) {
        throw InvalidArgumentError("reshape may contain at most one inferred dimension", location);
      }
      inferred_axis = axis;
      continue;
    }
    if (dimension < 0) {
      throw InvalidArgumentError("reshape dimensions must be non-negative or -1", location);
    }
    dimensions[axis] = dimension;
    known_elements = internal::CheckedMultiply(known_elements, dimension, "reshape known element count", location);
  }

  const auto input_elements = impl.GetNumElements();
  if (inferred_axis.has_value()) {
    if (input_elements == 0) {
      throw InvalidArgumentError("cannot infer a reshape dimension for an empty tensor", location);
    }
    if (known_elements == 0 || input_elements % known_elements != 0) {
      throw InvalidArgumentError("reshape inferred dimension is not integral", location);
    }
    dimensions[*inferred_axis] = input_elements / known_elements;
  }

  const auto shape = Shape{std::span<const int64_t>{dimensions.data(), requested.size()}, location};
  if (shape.GetNumElements() != input_elements) {
    throw InvalidArgumentError("reshape must preserve the tensor element count", location);
  }
  return shape;
}

auto View(const Tensor &input, const Shape &shape, std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto strides = internal::ComputeViewStrides(impl.GetShape(), impl.GetStrides(), shape, location);
  if (!strides.has_value()) {
    ThrowViewIncompatible(input, shape, location);
  }
  return MakeView(input, shape, *strides, impl.GetStorageOffset(), location);
}

auto View(const Tensor &input, std::span<const int64_t> requested, std::source_location location) -> Tensor {
  return View(input, InferReshape(input, requested, location), location);
}

auto Reshape(ExecutionContext &context, const Tensor &input, const Shape &shape, std::source_location location)
    -> Tensor {
  {
    internal::OpGuard guard{context, "Reshape", location};
    guard.ValidateTensor(input);
  }
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  if (impl.GetNumElements() != shape.GetNumElements()) {
    throw InvalidArgumentError("reshape must preserve the tensor element count", location);
  }

  const auto strides = internal::ComputeViewStrides(impl.GetShape(), impl.GetStrides(), shape, location);
  if (strides.has_value()) {
    return MakeView(input, shape, *strides, impl.GetStorageOffset(), location);
  }
  return View(Contiguous(context, input, location), shape, location);
}

auto Reshape(ExecutionContext &context, const Tensor &input, std::span<const int64_t> requested,
             std::source_location location) -> Tensor {
  return Reshape(context, input, InferReshape(input, requested, location), location);
}

auto Flatten(ExecutionContext &context, const Tensor &input, int64_t start_axis, int64_t end_axis,
             std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, "Flatten", location};
    guard.ValidateTensor(input);
  }
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto rank = impl.GetShape().GetRank();
  if (rank == 0) {
    if ((start_axis != 0 && start_axis != -1) || (end_axis != 0 && end_axis != -1)) {
      throw InvalidArgumentError("flatten axis out of range for a scalar tensor", location);
    }
    return Reshape(context, input, Shape{1}, location);
  }

  const auto normalized_start = NormalizeAxis(start_axis, rank, location);
  const auto normalized_end = NormalizeAxis(end_axis, rank, location);
  if (normalized_start > normalized_end) {
    throw InvalidArgumentError("flatten start axis must not follow end axis", location);
  }

  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  size_t output_axis = 0;
  for (size_t axis = 0; axis < normalized_start; ++axis) {
    dimensions[output_axis] = impl.GetShape().GetDimensions()[axis];
    output_axis++;
  }
  auto flattened_dimension = int64_t{1};
  for (size_t axis = normalized_start; axis <= normalized_end; ++axis) {
    flattened_dimension = internal::CheckedMultiply(flattened_dimension, impl.GetShape().GetDimensions()[axis],
                                                    "flattened dimension", location);
  }
  dimensions[output_axis] = flattened_dimension;
  output_axis++;
  for (size_t axis = normalized_end + 1; axis < rank; ++axis) {
    dimensions[output_axis] = impl.GetShape().GetDimensions()[axis];
    output_axis++;
  }
  return Reshape(context, input, Shape{std::span<const int64_t>{dimensions.data(), output_axis}, location}, location);
}

auto Permute(const Tensor &input, std::span<const int64_t> axes, std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto rank = impl.GetShape().GetRank();
  if (axes.size() != rank) {
    throw InvalidArgumentError("permutation axis count must equal tensor rank", location);
  }

  std::array<bool, TTL_MAX_RANK> seen{};
  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  std::array<int64_t, TTL_MAX_RANK> strides{};
  auto identity = true;
  for (size_t output_axis = 0; output_axis < rank; ++output_axis) {
    const auto input_axis = NormalizeAxis(axes[output_axis], rank, location);
    if (seen[input_axis]) {
      throw InvalidArgumentError("permutation contains a duplicate axis", location);
    }
    seen[input_axis] = true;
    identity = identity && input_axis == output_axis;
    dimensions[output_axis] = impl.GetShape().GetDimensions()[input_axis];
    strides[output_axis] = impl.GetStrides().GetValues()[input_axis];
  }
  if (identity) {
    return input;
  }
  return MakeView(input, Shape{std::span<const int64_t>{dimensions.data(), rank}, location},
                  Strides{std::span<const int64_t>{strides.data(), rank}, location}, impl.GetStorageOffset(), location);
}

auto Transpose(const Tensor &input, int64_t axis_a, int64_t axis_b, std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto rank = impl.GetShape().GetRank();
  const auto normalized_a = NormalizeAxis(axis_a, rank, location);
  const auto normalized_b = NormalizeAxis(axis_b, rank, location);
  if (normalized_a == normalized_b) {
    return input;
  }

  std::array<int64_t, TTL_MAX_RANK> axes{};
  for (size_t axis = 0; axis < rank; ++axis) {
    axes[axis] = static_cast<int64_t>(axis);
  }
  std::swap(axes[normalized_a], axes[normalized_b]);
  return Permute(input, std::span<const int64_t>{axes.data(), rank}, location);
}

auto Squeeze(const Tensor &input, std::optional<int64_t> axis, std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto old_rank = impl.GetShape().GetRank();
  std::optional<size_t> removed_axis;
  if (axis.has_value()) {
    removed_axis = NormalizeAxis(*axis, old_rank, location);
    if (impl.GetShape().GetDimensions()[*removed_axis] != 1) {
      throw InvalidArgumentError("squeeze axis must have dimension one", location);
    }
  }

  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  std::array<int64_t, TTL_MAX_RANK> strides{};
  size_t new_rank = 0;
  for (size_t old_axis = 0; old_axis < old_rank; ++old_axis) {
    const auto remove =
        removed_axis.has_value() ? old_axis == *removed_axis : impl.GetShape().GetDimensions()[old_axis] == 1;
    if (remove) {
      continue;
    }
    dimensions[new_rank] = impl.GetShape().GetDimensions()[old_axis];
    strides[new_rank] = impl.GetStrides().GetValues()[old_axis];
    ++new_rank;
  }
  if (new_rank == old_rank) {
    return input;
  }
  return MakeView(input, Shape{std::span<const int64_t>{dimensions.data(), new_rank}, location},
                  Strides{std::span<const int64_t>{strides.data(), new_rank}, location}, impl.GetStorageOffset(),
                  location);
}

auto Unsqueeze(const Tensor &input, int64_t axis, std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto old_rank = impl.GetShape().GetRank();
  const auto inserted_axis = NormalizeInsertedAxis(axis, old_rank, location);
  const auto new_rank = old_rank + 1;
  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  std::array<int64_t, TTL_MAX_RANK> strides{};

  for (size_t new_axis = 0; new_axis < new_rank; ++new_axis) {
    if (new_axis == inserted_axis) {
      dimensions[new_axis] = 1;
      if (new_axis == old_rank) {
        strides[new_axis] = 1;
      } else {
        const auto dimension = std::max(impl.GetShape().GetDimensions()[new_axis], int64_t{1});
        strides[new_axis] =
            internal::CheckedMultiply(impl.GetStrides().GetValues()[new_axis], dimension, "unsqueeze stride", location);
      }
      continue;
    }
    const auto old_axis = new_axis < inserted_axis ? new_axis : new_axis - 1;
    dimensions[new_axis] = impl.GetShape().GetDimensions()[old_axis];
    strides[new_axis] = impl.GetStrides().GetValues()[old_axis];
  }
  return MakeView(input, Shape{std::span<const int64_t>{dimensions.data(), new_rank}, location},
                  Strides{std::span<const int64_t>{strides.data(), new_rank}, location}, impl.GetStorageOffset(),
                  location);
}

auto Narrow(const Tensor &input, int64_t axis, int64_t start, int64_t length, std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto normalized_axis = NormalizeAxis(axis, impl.GetShape().GetRank(), location);
  const auto dimension = impl.GetShape().GetDimensions()[normalized_axis];
  if (start < 0) {
    start = internal::CheckedAdd(start, dimension, "narrow start", location);
  }
  if (length < 0) {
    throw InvalidArgumentError("narrow length must be non-negative", location);
  }
  if (start < 0 || start > dimension) {
    throw InvalidArgumentError("narrow start is out of range", location);
  }
  const auto end = internal::CheckedAdd(start, length, "narrow end", location);
  if (end > dimension) {
    throw InvalidArgumentError("narrow range exceeds the selected dimension", location);
  }
  if (start == 0 && length == dimension) {
    return input;
  }

  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  std::ranges::copy(impl.GetShape().GetDimensions(), dimensions.begin());
  dimensions[normalized_axis] = length;
  const auto offset_delta = internal::CheckedMultiply(start, impl.GetStrides().GetValues()[normalized_axis],
                                                      "narrow storage offset", location);
  const auto storage_offset =
      internal::CheckedAdd(impl.GetStorageOffset(), offset_delta, "narrow storage offset", location);
  return MakeView(input, Shape{std::span<const int64_t>{dimensions.data(), impl.GetShape().GetRank()}, location},
                  impl.GetStrides(), storage_offset, location);
}

auto Split(const Tensor &input, std::span<const int64_t> sizes, int64_t axis, std::source_location location)
    -> std::vector<Tensor> {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto normalized_axis = NormalizeAxis(axis, impl.GetShape().GetRank(), location);
  const auto dimension = impl.GetShape().GetDimensions()[normalized_axis];

  auto total = int64_t{0};
  for (const auto size : sizes) {
    if (size < 0) {
      throw InvalidArgumentError("split sizes must be non-negative", location);
    }
    total = internal::CheckedAdd(total, size, "split size sum", location);
  }
  if (total != dimension) {
    throw InvalidArgumentError("split sizes must sum to the selected dimension", location);
  }

  std::vector<Tensor> outputs;
  outputs.reserve(sizes.size());
  auto start = int64_t{0};
  for (const auto size : sizes) {
    outputs.emplace_back(Narrow(input, static_cast<int64_t>(normalized_axis), start, size, location));
    start = internal::CheckedAdd(start, size, "split offset", location);
  }
  return outputs;
}

auto Chunk(const Tensor &input, int64_t chunks, int64_t axis, std::source_location location) -> std::vector<Tensor> {
  if (chunks <= 0) {
    throw InvalidArgumentError("chunk count must be positive", location);
  }
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto normalized_axis = NormalizeAxis(axis, impl.GetShape().GetRank(), location);
  const auto dimension = impl.GetShape().GetDimensions()[normalized_axis];

  std::vector<Tensor> outputs;
  if (dimension == 0) {
    const auto output_count = internal::CheckedNarrow<size_t>(chunks, "chunk output count", location);
    outputs.reserve(output_count);
    for (auto index = int64_t{0}; index < chunks; ++index) {
      outputs.emplace_back(Narrow(input, static_cast<int64_t>(normalized_axis), 0, 0, location));
    }
    return outputs;
  }

  const auto chunk_size = internal::CeilDivide(dimension, chunks, "chunk size", location);
  const auto output_count = internal::CeilDivide(dimension, chunk_size, "chunk output count", location);
  outputs.reserve(internal::CheckedNarrow<size_t>(output_count, "chunk output count", location));
  auto start = int64_t{0};
  while (start < dimension) {
    const auto length = std::min(chunk_size, dimension - start);
    outputs.emplace_back(Narrow(input, static_cast<int64_t>(normalized_axis), start, length, location));
    start = internal::CheckedAdd(start, length, "chunk offset", location);
  }
  return outputs;
}

auto Slice(const Tensor &input, int64_t axis, std::optional<int64_t> start, std::optional<int64_t> stop, int64_t step,
           std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  if (step <= 0) {
    throw InvalidArgumentError("slice step must be positive", location);
  }
  const auto normalized_axis = NormalizeAxis(axis, impl.GetShape().GetRank(), location);
  const auto dimension = impl.GetShape().GetDimensions()[normalized_axis];
  const auto normalized_start =
      start.has_value() ? NormalizeSliceBound(*start, dimension, "slice start", location) : int64_t{0};
  const auto normalized_stop =
      stop.has_value() ? NormalizeSliceBound(*stop, dimension, "slice stop", location) : dimension;
  const auto distance =
      std::max(internal::CheckedSubtract(normalized_stop, normalized_start, "slice range", location), int64_t{0});
  const auto length = internal::CeilDivide(distance, step, "slice length", location);
  if (normalized_start == 0 && normalized_stop == dimension && step == 1) {
    return input;
  }

  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  std::array<int64_t, TTL_MAX_RANK> strides{};
  std::ranges::copy(impl.GetShape().GetDimensions(), dimensions.begin());
  std::ranges::copy(impl.GetStrides().GetValues(), strides.begin());
  dimensions[normalized_axis] = length;
  const auto old_stride = strides[normalized_axis];
  strides[normalized_axis] = internal::CheckedMultiply(old_stride, step, "slice stride", location);
  const auto offset_delta = internal::CheckedMultiply(normalized_start, old_stride, "slice storage offset", location);
  const auto storage_offset =
      internal::CheckedAdd(impl.GetStorageOffset(), offset_delta, "slice storage offset", location);
  const auto rank = impl.GetShape().GetRank();
  return MakeView(input, Shape{std::span<const int64_t>{dimensions.data(), rank}, location},
                  Strides{std::span<const int64_t>{strides.data(), rank}, location}, storage_offset, location);
}

auto Select(const Tensor &input, int64_t axis, int64_t index, std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto old_rank = impl.GetShape().GetRank();
  const auto normalized_axis = NormalizeAxis(axis, old_rank, location);
  const auto dimension = impl.GetShape().GetDimensions()[normalized_axis];
  if (dimension == 0) {
    throw InvalidArgumentError("cannot select from an empty dimension", location);
  }
  index = NormalizeIndex(index, dimension, "select index", location);

  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  std::array<int64_t, TTL_MAX_RANK> strides{};
  size_t new_axis = 0;
  for (size_t old_axis = 0; old_axis < old_rank; ++old_axis) {
    if (old_axis == normalized_axis) {
      continue;
    }
    dimensions[new_axis] = impl.GetShape().GetDimensions()[old_axis];
    strides[new_axis] = impl.GetStrides().GetValues()[old_axis];
    ++new_axis;
  }
  const auto offset_delta = internal::CheckedMultiply(index, impl.GetStrides().GetValues()[normalized_axis],
                                                      "select storage offset", location);
  const auto storage_offset =
      internal::CheckedAdd(impl.GetStorageOffset(), offset_delta, "select storage offset", location);
  return MakeView(input, Shape{std::span<const int64_t>{dimensions.data(), old_rank - 1}, location},
                  Strides{std::span<const int64_t>{strides.data(), old_rank - 1}, location}, storage_offset, location);
}

auto Expand(const Tensor &input, const Shape &shape, std::source_location location) -> Tensor {
  const auto &impl = internal::TensorAccess::GetImpl(input, location);
  const auto input_rank = impl.GetShape().GetRank();
  const auto output_rank = shape.GetRank();
  if (output_rank < input_rank) {
    throw InvalidArgumentError("expanded rank must not be smaller than input rank", location);
  }

  std::array<int64_t, TTL_MAX_RANK> strides{};
  const auto leading_axes = output_rank - input_rank;
  for (size_t output_axis = 0; output_axis < output_rank; ++output_axis) {
    if (output_axis < leading_axes) {
      strides[output_axis] = 0;
      continue;
    }

    const auto input_axis = output_axis - leading_axes;
    const auto input_dimension = impl.GetShape().GetDimensions()[input_axis];
    const auto output_dimension = shape.GetDimensions()[output_axis];
    if (input_dimension == output_dimension) {
      strides[output_axis] = impl.GetStrides().GetValues()[input_axis];
    } else if (input_dimension == 1) {
      strides[output_axis] = 0;
    } else {
      throw InvalidArgumentError("input shape cannot be expanded to the requested shape", location);
    }
  }
  return MakeView(input, shape, Strides{std::span<const int64_t>{strides.data(), output_rank}, location},
                  impl.GetStorageOffset(), location);
}

auto BroadcastShapes(std::span<const Shape> shapes, std::source_location location) -> Shape {
  size_t result_rank = 0;
  for (const auto &shape : shapes) {
    result_rank = std::max(result_rank, shape.GetRank());
  }

  std::array<int64_t, TTL_MAX_RANK> dimensions{};
  std::ranges::fill(dimensions, int64_t{1});
  for (const auto &shape : shapes) {
    const auto leading_axes = result_rank - shape.GetRank();
    for (size_t input_axis = 0; input_axis < shape.GetRank(); ++input_axis) {
      const auto result_axis = leading_axes + input_axis;
      const auto current = dimensions[result_axis];
      const auto candidate = shape.GetDimensions()[input_axis];
      if (current == candidate || candidate == 1) {
        continue;
      }
      if (current == 1) {
        dimensions[result_axis] = candidate;
        continue;
      }
      throw InvalidArgumentError("tensor shapes are not broadcast-compatible", location);
    }
  }
  return Shape{std::span<const int64_t>{dimensions.data(), result_rank}, location};
}

}  // namespace ttl
