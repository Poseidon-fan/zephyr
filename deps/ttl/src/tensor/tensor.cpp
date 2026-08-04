#include "ttl/tensor/tensor.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <source_location>
#include <string>
#include <utility>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/memory/device_allocator.hpp"
#include "ttl/internal/tensor/storage.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {
namespace {

struct TensorByteRange final {
  size_t begin_;
  size_t end_;
};

[[nodiscard]] auto ComputeMaximumElementOffset(const Shape &shape, const Strides &strides, int64_t storage_offset,
                                               std::source_location location) -> int64_t {
  auto maximum_offset = storage_offset;
  for (size_t axis = 0; axis < shape.GetRank(); axis++) {
    const auto dimension_extent =
        CheckedMultiply(shape.GetDimension(axis) - 1, strides.GetStride(axis), "tensor dimension extent", location);
    maximum_offset = CheckedAdd(maximum_offset, dimension_extent, "tensor maximum storage offset", location);
  }
  return maximum_offset;
}

[[nodiscard]] auto ComputeTensorByteRange(const Shape &shape, const Strides &strides, int64_t storage_offset,
                                          size_t element_size, std::source_location location) -> TensorByteRange {
  const auto begin = CheckedElementOffsetToBytes(storage_offset, element_size, location);
  if (shape.IsEmpty()) {
    return {.begin_ = begin, .end_ = begin};
  }

  const auto maximum_offset = ComputeMaximumElementOffset(shape, strides, storage_offset, location);
  const auto end_offset = CheckedAdd(maximum_offset, int64_t{1}, "tensor storage end offset", location);
  return {
      .begin_ = begin,
      .end_ = CheckedElementOffsetToBytes(end_offset, element_size, location),
  };
}

void ValidateTensorStorage(const Storage &storage, const TensorByteRange &byte_range, size_t alignment,
                           std::source_location location) {
  const auto capacity_bytes = storage.GetCapacityBytes();
  const auto *base_pointer = storage.GetBasePointer();
  if ((capacity_bytes == 0) != (base_pointer == nullptr)) {
    throw InternalError("tensor Storage has inconsistent pointer and capacity", location);
  }
  if (byte_range.end_ > capacity_bytes) {
    throw InvalidArgumentError("tensor view exceeds Storage capacity", location);
  }
  if (base_pointer == nullptr) {
    return;
  }

  const auto base_address = reinterpret_cast<uintptr_t>(base_pointer);
  if (byte_range.end_ > std::numeric_limits<uintptr_t>::max() - base_address) {
    throw OverflowError("tensor reachable address range overflow", location);
  }
  if ((base_address + byte_range.begin_) % alignment != 0) {
    throw InvalidArgumentError("tensor data pointer does not satisfy dtype alignment", location);
  }
}

[[nodiscard]] auto HasZeroStride(const Shape &shape, const Strides &strides) noexcept -> bool {
  for (size_t axis = 0; axis < shape.GetRank(); axis++) {
    if (shape.GetDimensions()[axis] > 1 && strides.GetValues()[axis] == 0) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] auto ComputeIsNonOverlappingDense(const Shape &shape, const Strides &strides,
                                                std::source_location location = std::source_location::current())
    -> bool {
  if (shape.GetNumElements() <= 1) {
    return true;
  }

  std::array<size_t, TTL_MAX_RANK> axes{};
  size_t axis_count = 0;
  for (size_t axis = 0; axis < shape.GetRank(); axis++) {
    if (shape.GetDimensions()[axis] > 1) {
      axes[axis_count] = axis;
      axis_count++;
    }
  }

  std::sort(axes.begin(), axes.begin() + static_cast<ptrdiff_t>(axis_count),
            [&strides](size_t lhs, size_t rhs) { return strides.GetValues()[lhs] < strides.GetValues()[rhs]; });

  int64_t expected_stride = 1;
  for (size_t index = 0; index < axis_count; index++) {
    const auto axis = axes[index];
    if (strides.GetValues()[axis] != expected_stride) {
      return false;
    }
    expected_stride = CheckedMultiply(expected_stride, shape.GetDimensions()[axis], "dense stride extent", location);
  }
  return true;
}

[[nodiscard]] auto ComputeTensorFlags(const Shape &shape, const Strides &strides, std::source_location location)
    -> TensorFlags {
  TensorFlags flags;
  if (IsContiguous(shape, strides, location)) {
    flags.Set(TensorFlag::CONTIGUOUS);
  }
  if (HasZeroStride(shape, strides)) {
    flags.Set(TensorFlag::HAS_ZERO_STRIDE);
  }
  if (ComputeIsNonOverlappingDense(shape, strides, location)) {
    flags.Set(TensorFlag::NON_OVERLAPPING_DENSE);
  }
  return flags;
}

[[nodiscard]] auto FormatDTypeMismatch(DType expected, DType actual, std::source_location location) -> std::string {
  std::string message{"tensor data requested as "};
  message.append(GetDTypeName(expected, location));
  message.append(" but tensor dtype is ");
  message.append(GetDTypeName(actual, location));
  return message;
}

}  // namespace

auto IsNonOverlappingDenseLayout(const Shape &shape, const Strides &strides) -> bool {
  return shape.GetRank() == strides.GetRank() && ComputeIsNonOverlappingDense(shape, strides);
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage, DType dtype, Shape shape, Strides strides,
                       int64_t storage_offset, TensorFlags flags) noexcept
    : storage_(std::move(storage)),
      shape_(shape),
      strides_(strides),
      storage_offset_(storage_offset),
      dtype_(dtype),
      flags_(flags) {}

auto TensorImpl::GetStorage() const noexcept -> const std::shared_ptr<Storage> & { return storage_; }

auto TensorImpl::GetDType() const noexcept -> DType { return dtype_; }

auto TensorImpl::GetShape() const noexcept -> const Shape & { return shape_; }

auto TensorImpl::GetStrides() const noexcept -> const Strides & { return strides_; }

auto TensorImpl::GetStorageOffset() const noexcept -> int64_t { return storage_offset_; }

auto TensorImpl::GetNumElements() const noexcept -> int64_t { return shape_.GetNumElements(); }

auto TensorImpl::HasFlag(TensorFlag flag) const noexcept -> bool { return flags_.Has(flag); }

auto TensorFactory::Create(std::shared_ptr<Storage> storage, DType dtype, Shape shape, Strides strides,
                           int64_t storage_offset, std::source_location location) -> Tensor {
  if (storage == nullptr) {
    throw InvalidArgumentError("tensor Storage must not be null", location);
  }
  if (shape.GetRank() != strides.GetRank()) {
    throw InvalidArgumentError("tensor shape and strides must have the same rank", location);
  }
  if (storage_offset < 0) {
    throw InvalidArgumentError("tensor storage offset must be non-negative", location);
  }

  const auto dtype_info = GetDTypeInfo(dtype, location);
  const auto byte_range = ComputeTensorByteRange(shape, strides, storage_offset, dtype_info.size_bytes_, location);
  ValidateTensorStorage(*storage, byte_range, dtype_info.alignment_bytes_, location);
  const auto flags = ComputeTensorFlags(shape, strides, location);

  auto impl = std::shared_ptr<const TensorImpl>{
      new TensorImpl(std::move(storage), dtype, shape, strides, storage_offset, flags)};
  return Tensor{std::move(impl)};
}

auto TensorAccess::GetImpl(const Tensor &tensor, std::source_location location) -> const TensorImpl & {
  if (tensor.impl_ == nullptr) {
    throw InvalidArgumentError("tensor is in a moved-from state", location);
  }
  return *tensor.impl_;
}

auto TensorAccess::GetStorage(const Tensor &tensor, std::source_location location) -> const std::shared_ptr<Storage> & {
  return GetImpl(tensor, location).GetStorage();
}

auto TensorAccess::GetData(const Tensor &tensor, std::source_location location) -> const void * {
  const auto &impl = GetImpl(tensor, location);
  return tensor.GetDataPointer(impl.GetDType(), location);
}

auto TensorAccess::GetMutableData(Tensor &tensor, std::source_location location) -> void * {
  return const_cast<void *>(GetData(tensor, location));
}

void TensorAccess::RecordUsage(const Tensor &tensor, const Stream &stream, std::source_location location) {
  GetStorage(tensor, location)->RecordUsage(stream);
}

}  // namespace ttl::internal

namespace ttl {

Tensor::Tensor(std::shared_ptr<const internal::TensorImpl> impl) noexcept : impl_(std::move(impl)) {}

auto Tensor::GetImpl(std::source_location location) const -> const internal::TensorImpl & {
  return internal::TensorAccess::GetImpl(*this, location);
}

auto Tensor::GetDevice(std::source_location location) const -> Device {
  return GetImpl(location).GetStorage()->GetDevice();
}

auto Tensor::GetDType(std::source_location location) const -> DType { return GetImpl(location).GetDType(); }

auto Tensor::GetShape(std::source_location location) const -> const Shape & { return GetImpl(location).GetShape(); }

auto Tensor::GetStrides(std::source_location location) const -> const Strides & {
  return GetImpl(location).GetStrides();
}

auto Tensor::GetRank(std::source_location location) const -> size_t { return GetImpl(location).GetShape().GetRank(); }

auto Tensor::GetNumElements(std::source_location location) const -> int64_t {
  return GetImpl(location).GetNumElements();
}

auto Tensor::GetStorageOffset(std::source_location location) const -> int64_t {
  return GetImpl(location).GetStorageOffset();
}

auto Tensor::IsContiguous(std::source_location location) const -> bool {
  return GetImpl(location).HasFlag(internal::TensorFlag::CONTIGUOUS);
}

auto Tensor::IsNonOverlappingDense(std::source_location location) const -> bool {
  return GetImpl(location).HasFlag(internal::TensorFlag::NON_OVERLAPPING_DENSE);
}

auto Tensor::HasZeroStride(std::source_location location) const -> bool {
  return GetImpl(location).HasFlag(internal::TensorFlag::HAS_ZERO_STRIDE);
}

void Tensor::RecordUsage(const Stream &stream, std::source_location location) const {
  if (stream.GetDevice(location) != GetDevice(location)) {
    throw InvalidArgumentError("tensor usage stream belongs to a different device", location);
  }
  internal::TensorAccess::RecordUsage(*this, stream, location);
}

auto Tensor::GetDataPointer(DType expected_dtype, std::source_location location) const -> const void * {
  const auto &impl = internal::TensorAccess::GetImpl(*this, location);
  if (expected_dtype != impl.GetDType()) {
    throw InvalidArgumentError(internal::FormatDTypeMismatch(expected_dtype, impl.GetDType(), location), location);
  }

  const auto *base_pointer = static_cast<const std::byte *>(impl.GetStorage()->GetBasePointer());
  if (base_pointer == nullptr) {
    return nullptr;
  }
  const auto byte_offset =
      internal::CheckedElementOffsetToBytes(impl.GetStorageOffset(), GetDTypeSize(impl.GetDType(), location), location);
  return base_pointer + byte_offset;
}

auto ClassifyAlias(const Tensor &lhs, const Tensor &rhs, std::source_location location) -> AliasKind {
  const auto &lhs_impl = internal::TensorAccess::GetImpl(lhs, location);
  const auto &rhs_impl = internal::TensorAccess::GetImpl(rhs, location);
  if (lhs_impl.GetStorage()->GetDevice() != rhs_impl.GetStorage()->GetDevice()) {
    return AliasKind::DISJOINT;
  }
  const auto shares_storage = lhs_impl.GetStorage().get() == rhs_impl.GetStorage().get();
  if (shares_storage && lhs_impl.GetDType() == rhs_impl.GetDType() && lhs_impl.GetShape() == rhs_impl.GetShape() &&
      lhs_impl.GetStrides() == rhs_impl.GetStrides() && lhs_impl.GetStorageOffset() == rhs_impl.GetStorageOffset()) {
    return AliasKind::EXACT;
  }
  if (lhs_impl.GetShape().IsEmpty() || rhs_impl.GetShape().IsEmpty()) {
    return AliasKind::DISJOINT;
  }

  const auto lhs_range =
      internal::ComputeTensorByteRange(lhs_impl.GetShape(), lhs_impl.GetStrides(), lhs_impl.GetStorageOffset(),
                                       GetDTypeSize(lhs_impl.GetDType(), location), location);
  const auto rhs_range =
      internal::ComputeTensorByteRange(rhs_impl.GetShape(), rhs_impl.GetStrides(), rhs_impl.GetStorageOffset(),
                                       GetDTypeSize(rhs_impl.GetDType(), location), location);
  if (shares_storage && (lhs_range.end_ <= rhs_range.begin_ || rhs_range.end_ <= lhs_range.begin_)) {
    return AliasKind::DISJOINT;
  }
  if (!shares_storage) {
    const auto lhs_base = reinterpret_cast<uintptr_t>(lhs_impl.GetStorage()->GetBasePointer());
    const auto rhs_base = reinterpret_cast<uintptr_t>(rhs_impl.GetStorage()->GetBasePointer());
    const auto lhs_begin = lhs_base + lhs_range.begin_;
    const auto lhs_end = lhs_base + lhs_range.end_;
    const auto rhs_begin = rhs_base + rhs_range.begin_;
    const auto rhs_end = rhs_base + rhs_range.end_;
    if (lhs_end <= rhs_begin || rhs_end <= lhs_begin) {
      return AliasKind::DISJOINT;
    }
  }
  return AliasKind::MAY_OVERLAP;
}

auto Empty(ExecutionContext &context, const Shape &shape, DType dtype, std::source_location location) -> Tensor {
  return EmptyStrided(context, shape, GetContiguousStrides(shape, location), dtype, location);
}

auto EmptyStrided(ExecutionContext &context, const Shape &shape, const Strides &strides, DType dtype,
                  std::source_location location) -> Tensor {
  internal::ContextUseGuard use_guard{context, internal::ContextUseMode::SUBMIT, location};
  if (shape.GetRank() != strides.GetRank()) {
    throw InvalidArgumentError("empty-strided shape and strides must have the same rank", location);
  }
  if (!internal::IsNonOverlappingDenseLayout(shape, strides)) {
    throw InvalidArgumentError("empty-strided layout must be non-overlapping and dense", location);
  }

  const auto dtype_info = GetDTypeInfo(dtype, location);
  const auto bytes = internal::CheckedBytes(shape.GetNumElements(), dtype_info.size_bytes_, location);
  auto storage = internal::ContextAccess::GetAllocator(context, location)
                     ->Allocate(context.GetStream(), bytes, 256,
                                internal::AllocationContext{
                                    .operation_ = "EmptyStrided",
                                    .output_shape_ = shape,
                                    .dtype_ = dtype,
                                    .location_ = location,
                                });
  return internal::TensorFactory::Create(std::move(storage), dtype, shape, strides, 0, location);
}

}  // namespace ttl
