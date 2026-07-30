#include "ttl/internal/elementwise_iterator.hpp"

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

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/layout.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl::internal {
namespace {

[[noreturn]] void ThrowIteratorError(std::string_view operation, std::string_view reason,
                                     std::source_location location) {
  std::string message{operation};
  message.append(": ");
  message.append(reason);
  throw InvalidArgumentError(std::move(message), location);
}

[[nodiscard]] auto CanMultiply(uint64_t lhs, uint64_t rhs) noexcept -> bool {
  return rhs == 0 || lhs <= std::numeric_limits<uint64_t>::max() / rhs;
}

[[nodiscard]] auto CanCoalesce(
    size_t outer_axis, size_t inner_axis, uint8_t operand_count, const std::array<uint64_t, TTL_MAX_RANK> &shape,
    const std::array<std::array<uint64_t, TTL_MAX_RANK>, TTL_MAX_ITERATOR_OPERANDS> &strides) noexcept -> bool {
  const auto outer_extent = shape[outer_axis];
  const auto inner_extent = shape[inner_axis];
  if (outer_extent == 1 || inner_extent == 1) {
    return true;
  }

  for (size_t operand = 0; operand < operand_count; ++operand) {
    const auto outer_stride = strides[operand][outer_axis];
    const auto inner_stride = strides[operand][inner_axis];
    if (outer_stride == 0 && inner_stride == 0) {
      continue;
    }
    if (!CanMultiply(inner_stride, inner_extent) || outer_stride != inner_stride * inner_extent) {
      return false;
    }
  }
  return true;
}

void ReorderDimensions(uint8_t rank, uint8_t operand_count, std::array<uint64_t, TTL_MAX_RANK> &shape,
                       std::array<std::array<uint64_t, TTL_MAX_RANK>, TTL_MAX_ITERATOR_OPERANDS> &strides) {
  if (rank <= 1) {
    return;
  }

  std::array<size_t, TTL_MAX_RANK> axes{};
  size_t axis_count = 0;
  for (size_t axis = 0; axis < rank; ++axis) {
    if (shape[axis] == 1) {
      axes[axis_count] = axis;
      ++axis_count;
    }
  }

  std::array<size_t, TTL_MAX_RANK> nontrivial_axes{};
  size_t nontrivial_count = 0;
  for (size_t axis = 0; axis < rank; ++axis) {
    if (shape[axis] != 1) {
      nontrivial_axes[nontrivial_count] = axis;
      ++nontrivial_count;
    }
  }
  std::stable_sort(nontrivial_axes.begin(), nontrivial_axes.begin() + static_cast<ptrdiff_t>(nontrivial_count),
                   [&strides](size_t lhs, size_t rhs) { return strides[0][lhs] > strides[0][rhs]; });
  for (size_t index = 0; index < nontrivial_count; ++index) {
    axes[axis_count] = nontrivial_axes[index];
    ++axis_count;
  }

  const auto old_shape = shape;
  const auto old_strides = strides;
  for (size_t new_axis = 0; new_axis < rank; ++new_axis) {
    const auto old_axis = axes[new_axis];
    shape[new_axis] = old_shape[old_axis];
    for (size_t operand = 0; operand < operand_count; ++operand) {
      strides[operand][new_axis] = old_strides[operand][old_axis];
    }
  }
}

[[nodiscard]] auto CoalesceDimensions(
    uint8_t rank, uint8_t operand_count, std::array<uint64_t, TTL_MAX_RANK> &shape,
    std::array<std::array<uint64_t, TTL_MAX_RANK>, TTL_MAX_ITERATOR_OPERANDS> &strides, std::source_location location)
    -> uint8_t {
  if (rank <= 1) {
    return rank;
  }

  size_t previous_axis = 0;
  for (size_t axis = 1; axis < rank; ++axis) {
    if (CanCoalesce(previous_axis, axis, operand_count, shape, strides)) {
      if (shape[previous_axis] == 1 || shape[axis] != 1) {
        for (size_t operand = 0; operand < operand_count; ++operand) {
          strides[operand][previous_axis] = strides[operand][axis];
        }
      }
      shape[previous_axis] =
          CheckedMultiply(shape[previous_axis], shape[axis], "coalesced iterator dimension", location);
      continue;
    }

    ++previous_axis;
    if (previous_axis == axis) {
      continue;
    }
    shape[previous_axis] = shape[axis];
    for (size_t operand = 0; operand < operand_count; ++operand) {
      strides[operand][previous_axis] = strides[operand][axis];
    }
  }
  return static_cast<uint8_t>(previous_axis + 1);
}

[[nodiscard]] auto IsBroadcastOperand(
    size_t operand, uint8_t rank,
    const std::array<std::array<uint64_t, TTL_MAX_RANK>, TTL_MAX_ITERATOR_OPERANDS> &strides) noexcept -> bool {
  for (size_t axis = 0; axis < rank; ++axis) {
    if (strides[operand][axis] != 0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] auto IsUnitOrBroadcastStride(uint64_t stride, size_t element_size, bool is_output) noexcept -> bool {
  return stride == element_size || (!is_output && stride == 0);
}

[[nodiscard]] auto SelectPath(
    uint8_t rank, uint8_t operand_count, const std::array<size_t, TTL_MAX_ITERATOR_OPERANDS> &element_sizes,
    const std::array<std::array<uint64_t, TTL_MAX_RANK>, TTL_MAX_ITERATOR_OPERANDS> &strides) noexcept -> IteratorPath {
  auto contiguous = rank == 1;
  if (contiguous) {
    for (size_t operand = 0; operand < operand_count; ++operand) {
      contiguous = contiguous && IsUnitOrBroadcastStride(strides[operand][0], element_sizes[operand], operand == 0);
    }
  }
  if (contiguous) {
    for (size_t operand = 1; operand < operand_count; ++operand) {
      if (IsBroadcastOperand(operand, rank, strides)) {
        return IteratorPath::CONTIGUOUS_WITH_SCALAR_INPUTS;
      }
    }
    return IteratorPath::CONTIGUOUS;
  }

  const auto inner_axis = static_cast<size_t>(rank - 1);
  auto contiguous_inner = true;
  for (size_t operand = 0; operand < operand_count; ++operand) {
    contiguous_inner =
        contiguous_inner && IsUnitOrBroadcastStride(strides[operand][inner_axis], element_sizes[operand], operand == 0);
  }
  return contiguous_inner ? IteratorPath::SINGLE_INNER_STRIDE : IteratorPath::GENERIC_STRIDED;
}

[[nodiscard]] auto SelectVectorWidth(
    int64_t num_elements, uint8_t rank, uint8_t operand_count, IteratorPath path,
    const std::array<std::byte *, TTL_MAX_ITERATOR_OPERANDS> &pointers,
    const std::array<size_t, TTL_MAX_ITERATOR_OPERANDS> &element_sizes,
    const std::array<std::array<uint64_t, TTL_MAX_RANK>, TTL_MAX_ITERATOR_OPERANDS> &strides) noexcept -> uint8_t {
  if (num_elements <= 0 || (path != IteratorPath::CONTIGUOUS && path != IteratorPath::CONTIGUOUS_WITH_SCALAR_INPUTS)) {
    return 1;
  }

  auto maximum_width = uint64_t{16};
  for (size_t operand = 0; operand < operand_count; ++operand) {
    if (operand != 0 && IsBroadcastOperand(operand, rank, strides)) {
      continue;
    }
    maximum_width = std::min(maximum_width, uint64_t{16} / element_sizes[operand]);
  }
  maximum_width = std::min(maximum_width, static_cast<uint64_t>(num_elements));

  uint64_t candidate = 1;
  while (candidate <= maximum_width / 2) {
    candidate *= 2;
  }
  for (; candidate > 1; candidate /= 2) {
    auto aligned = true;
    for (size_t operand = 0; operand < operand_count; ++operand) {
      if (operand != 0 && IsBroadcastOperand(operand, rank, strides)) {
        continue;
      }
      const auto alignment = candidate * element_sizes[operand];
      const auto address = reinterpret_cast<uintptr_t>(pointers[operand]);
      aligned = aligned && address % alignment == 0;
    }
    if (aligned) {
      return static_cast<uint8_t>(candidate);
    }
  }
  return 1;
}

[[nodiscard]] auto CanUse32BitIndexing(
    uint64_t num_elements, uint8_t rank, uint8_t operand_count, const std::array<uint64_t, TTL_MAX_RANK> &shape,
    const std::array<std::array<uint64_t, TTL_MAX_RANK>, TTL_MAX_ITERATOR_OPERANDS> &strides) noexcept -> bool {
  constexpr auto maximum = uint64_t{std::numeric_limits<uint32_t>::max()};
  if (num_elements > maximum) {
    return false;
  }

  auto empty = false;
  for (size_t axis = 0; axis < rank; ++axis) {
    if (shape[axis] > maximum) {
      return false;
    }
    empty = empty || shape[axis] == 0;
  }

  for (size_t operand = 0; operand < operand_count; ++operand) {
    auto maximum_offset = uint64_t{0};
    for (size_t axis = 0; axis < rank; ++axis) {
      const auto stride = strides[operand][axis];
      if (stride > maximum) {
        return false;
      }
      if (empty) {
        continue;
      }
      const auto extent = shape[axis] - 1;
      if (!CanMultiply(extent, stride)) {
        return false;
      }
      const auto dimension_offset = extent * stride;
      if (maximum_offset > std::numeric_limits<uint64_t>::max() - dimension_offset) {
        return false;
      }
      maximum_offset += dimension_offset;
    }
    if (maximum_offset > maximum) {
      return false;
    }
  }
  return true;
}

}  // namespace

void ValidateWritableOutput(const Tensor &output, std::string_view operation, std::source_location location) {
  const auto &impl = TensorAccess::GetImpl(output, location);
  if (!impl.HasFlag(TensorFlag::NON_OVERLAPPING_DENSE)) {
    ThrowIteratorError(operation, "output must be non-overlapping and dense", location);
  }
  if (impl.HasFlag(TensorFlag::HAS_ZERO_STRIDE)) {
    ThrowIteratorError(operation, "broadcast output is read-only", location);
  }
}

void ValidateAlias(AliasPolicy policy, const Tensor &output, std::span<const Tensor *const> inputs,
                   std::string_view operation, std::source_location location) {
  switch (policy) {
    case AliasPolicy::EXACT_UNARY:
    case AliasPolicy::COPY:
      if (inputs.size() != 1) {
        ThrowIteratorError(operation, "selected alias policy requires exactly one input", location);
      }
      break;
    case AliasPolicy::EXACT_ONE_BINARY_INPUT:
      if (inputs.size() != 2) {
        ThrowIteratorError(operation, "selected alias policy requires exactly two inputs", location);
      }
      break;
    case AliasPolicy::NO_ALIAS:
      break;
  }

  size_t exact_alias_count = 0;
  for (const auto *input : inputs) {
    if (input == nullptr) {
      ThrowIteratorError(operation, "input Tensor pointer must not be null", location);
    }
    const auto alias = ClassifyAlias(output, *input, location);
    if (alias == AliasKind::MAY_OVERLAP) {
      ThrowIteratorError(operation, "output may partially overlap an input", location);
    }
    if (alias != AliasKind::EXACT) {
      continue;
    }

    ++exact_alias_count;
    switch (policy) {
      case AliasPolicy::NO_ALIAS:
        ThrowIteratorError(operation, "output must not alias an input", location);
      case AliasPolicy::EXACT_UNARY:
      case AliasPolicy::COPY:
        break;
      case AliasPolicy::EXACT_ONE_BINARY_INPUT:
        if (exact_alias_count > 1) {
          ThrowIteratorError(operation, "output may exactly alias at most one input", location);
        }
        break;
    }
  }
}

auto ElementwiseIterator::Builder::AddOutput(Tensor &tensor) -> Builder & {
  if (output_.has_value()) {
    throw InvalidArgumentError("elementwise iterator accepts exactly one output");
  }
  output_.emplace(tensor);
  return *this;
}

auto ElementwiseIterator::Builder::AddInput(const Tensor &tensor) -> Builder & {
  if (input_count_ == inputs_.size()) {
    throw InvalidArgumentError("elementwise iterator accepts at most three inputs");
  }
  inputs_[input_count_].emplace(tensor);
  ++input_count_;
  return *this;
}

auto ElementwiseIterator::Builder::SetAliasPolicy(AliasPolicy policy) noexcept -> Builder & {
  alias_policy_ = policy;
  return *this;
}

auto ElementwiseIterator::Builder::SetRequireSameDType(bool value) noexcept -> Builder & {
  require_same_dtype_ = value;
  return *this;
}

auto ElementwiseIterator::Builder::Build(std::string_view operation, std::source_location location)
    -> ElementwiseIterator {
  if (!output_.has_value()) {
    ThrowIteratorError(operation, "one output is required", location);
  }
  ValidateWritableOutput(*output_, operation, location);

  std::array<Shape, TTL_MAX_ITERATOR_INPUTS> input_shapes{};
  std::array<const Tensor *, TTL_MAX_ITERATOR_INPUTS> input_pointers{};
  for (size_t input = 0; input < input_count_; ++input) {
    input_shapes[input] = TensorAccess::GetImpl(*inputs_[input], location).GetShape();
    input_pointers[input] = &*inputs_[input];
  }
  const auto result_shape = input_count_ == 0
                                ? TensorAccess::GetImpl(*output_, location).GetShape()
                                : BroadcastShapes(std::span<const Shape>{input_shapes.data(), input_count_}, location);
  if (output_->GetShape() != result_shape) {
    ThrowIteratorError(operation, "output shape does not match the broadcast result", location);
  }

  const auto output_device = output_->GetDevice();
  const auto output_dtype = output_->GetDType();
  for (size_t input = 0; input < input_count_; ++input) {
    if (inputs_[input]->GetDevice() != output_device) {
      ThrowIteratorError(operation, "all operands must be on the same device", location);
    }
    if (require_same_dtype_ && inputs_[input]->GetDType() != output_dtype) {
      ThrowIteratorError(operation, "all operands must have the same dtype", location);
    }
  }
  ValidateAlias(alias_policy_, *output_, std::span<const Tensor *const>{input_pointers.data(), input_count_}, operation,
                location);

  ElementwiseIterator iterator;
  iterator.shape_ = result_shape;
  iterator.num_elements_ = result_shape.GetNumElements();
  iterator.operand_count_ = static_cast<uint8_t>(input_count_ + 1);
  iterator.rank_ = static_cast<uint8_t>(result_shape.GetRank());
  iterator.operands_[0] = output_;
  for (size_t input = 0; input < input_count_; ++input) {
    iterator.operands_[input + 1] = inputs_[input];
  }

  std::array<size_t, TTL_MAX_ITERATOR_OPERANDS> element_sizes{};
  element_sizes[0] = GetDTypeSize(output_dtype, location);
  iterator.pointers_[0] = static_cast<std::byte *>(TensorAccess::GetMutableData(*iterator.operands_[0], location));
  for (size_t input = 0; input < input_count_; ++input) {
    const auto operand = input + 1;
    element_sizes[operand] = GetDTypeSize(iterator.operands_[operand]->GetDType(), location);
    iterator.pointers_[operand] = const_cast<std::byte *>(
        static_cast<const std::byte *>(TensorAccess::GetData(*iterator.operands_[operand], location)));
  }

  const auto &output = *iterator.operands_[0];
  for (size_t axis = 0; axis < result_shape.GetRank(); ++axis) {
    iterator.iteration_shape_[axis] =
        CheckedNarrow<uint64_t>(result_shape.GetDimensions()[axis], "iterator dimension", location);
    const auto output_stride =
        CheckedNarrow<uint64_t>(output.GetStrides().GetValues()[axis], "iterator output stride", location);
    iterator.strides_bytes_[0][axis] = CheckedMultiply(output_stride, static_cast<uint64_t>(element_sizes[0]),
                                                       "iterator output byte stride", location);
  }

  for (size_t input = 0; input < input_count_; ++input) {
    const auto operand = input + 1;
    const auto &input_tensor = *iterator.operands_[operand];
    const auto &input_shape = input_tensor.GetShape();
    const auto &input_strides = input_tensor.GetStrides();
    const auto leading_axes = result_shape.GetRank() - input_shape.GetRank();
    for (size_t result_axis = 0; result_axis < result_shape.GetRank(); ++result_axis) {
      if (result_axis < leading_axes) {
        iterator.strides_bytes_[operand][result_axis] = 0;
        continue;
      }

      const auto input_axis = result_axis - leading_axes;
      const auto input_dimension = input_shape.GetDimensions()[input_axis];
      const auto result_dimension = result_shape.GetDimensions()[result_axis];
      if (input_dimension == 1 && result_dimension != 1) {
        iterator.strides_bytes_[operand][result_axis] = 0;
        continue;
      }
      const auto input_stride =
          CheckedNarrow<uint64_t>(input_strides.GetValues()[input_axis], "iterator input stride", location);
      iterator.strides_bytes_[operand][result_axis] = CheckedMultiply(
          input_stride, static_cast<uint64_t>(element_sizes[operand]), "iterator input byte stride", location);
    }
  }

  if (iterator.rank_ == 0) {
    iterator.rank_ = 1;
    iterator.iteration_shape_[0] = 1;
    for (size_t operand = 0; operand < iterator.operand_count_; ++operand) {
      iterator.strides_bytes_[operand][0] = element_sizes[operand];
    }
  } else {
    ReorderDimensions(iterator.rank_, iterator.operand_count_, iterator.iteration_shape_, iterator.strides_bytes_);
    iterator.rank_ = CoalesceDimensions(iterator.rank_, iterator.operand_count_, iterator.iteration_shape_,
                                        iterator.strides_bytes_, location);
  }

  iterator.path_ = SelectPath(iterator.rank_, iterator.operand_count_, element_sizes, iterator.strides_bytes_);
  iterator.vector_width_elements_ =
      SelectVectorWidth(iterator.num_elements_, iterator.rank_, iterator.operand_count_, iterator.path_,
                        iterator.pointers_, element_sizes, iterator.strides_bytes_);
  iterator.index_width_ =
      CanUse32BitIndexing(static_cast<uint64_t>(iterator.num_elements_), iterator.rank_, iterator.operand_count_,
                          iterator.iteration_shape_, iterator.strides_bytes_)
          ? IndexWidth::UINT32
          : IndexWidth::UINT64;
  return iterator;
}

auto ElementwiseIterator::GetShape() const noexcept -> const Shape & { return shape_; }

auto ElementwiseIterator::GetNumElements() const noexcept -> int64_t { return num_elements_; }

auto ElementwiseIterator::GetRank() const noexcept -> size_t { return rank_; }

auto ElementwiseIterator::GetOperandCount() const noexcept -> size_t { return operand_count_; }

auto ElementwiseIterator::GetIndexWidth() const noexcept -> IndexWidth { return index_width_; }

auto ElementwiseIterator::GetVectorWidthElements() const noexcept -> uint8_t { return vector_width_elements_; }

auto ElementwiseIterator::GetPath() const noexcept -> IteratorPath { return path_; }

auto ElementwiseIterator::MakeParameters32(std::source_location location) const -> ElementwiseParameters32 {
  if (index_width_ != IndexWidth::UINT32) {
    throw InternalError("cannot create 32-bit parameters for a 64-bit elementwise iterator", location);
  }

  ElementwiseParameters32 parameters;
  parameters.num_elements_ = static_cast<uint32_t>(num_elements_);
  parameters.rank_ = rank_;
  parameters.operand_count_ = operand_count_;
  for (size_t operand = 0; operand < operand_count_; ++operand) {
    parameters.pointers_[operand] = pointers_[operand];
  }
  for (size_t axis = 0; axis < rank_; ++axis) {
    parameters.shape_[axis] = static_cast<uint32_t>(iteration_shape_[axis]);
    for (size_t operand = 0; operand < operand_count_; ++operand) {
      parameters.strides_bytes_[operand][axis] = static_cast<uint32_t>(strides_bytes_[operand][axis]);
    }
  }
  return parameters;
}

auto ElementwiseIterator::MakeParameters64() const noexcept -> ElementwiseParameters64 {
  ElementwiseParameters64 parameters;
  parameters.num_elements_ = static_cast<uint64_t>(num_elements_);
  parameters.rank_ = rank_;
  parameters.operand_count_ = operand_count_;
  for (size_t operand = 0; operand < operand_count_; ++operand) {
    parameters.pointers_[operand] = pointers_[operand];
    for (size_t axis = 0; axis < rank_; ++axis) {
      parameters.strides_bytes_[operand][axis] = strides_bytes_[operand][axis];
    }
  }
  for (size_t axis = 0; axis < rank_; ++axis) {
    parameters.shape_[axis] = iteration_shape_[axis];
  }
  return parameters;
}

}  // namespace ttl::internal
