#include "ttl/ops/attention.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <source_location>
#include <span>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/ops/attention.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/layout.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

struct SdpaShapeInfo final {
  Shape output_shape_;
  Shape score_shape_;
  int64_t batch_size_;
  int64_t query_head_count_;
  int64_t key_value_head_count_;
  int64_t query_length_;
  int64_t key_length_;
  int64_t head_dimension_;
  int64_t value_dimension_;
};

void ValidateCausalAlignment(CausalAlignment alignment, std::source_location location) {
  switch (alignment) {
    case CausalAlignment::UPPER_LEFT:
    case CausalAlignment::LOWER_RIGHT:
      return;
  }
  throw InvalidArgumentError("invalid SDPA causal alignment", location);
}

void ValidateSdpaOptions(const SdpaOptions &options, int64_t head_dimension, std::source_location location) {
  ValidateCausalAlignment(options.causal_alignment_, location);
  if (head_dimension == 0) {
    throw InvalidArgumentError("SDPA head dimension must be positive", location);
  }
  if (options.scale_.has_value() && !std::isfinite(*options.scale_)) {
    throw InvalidArgumentError("SDPA scale must be finite", location);
  }
}

[[nodiscard]] auto InferSdpaShape(const Tensor &query, const Tensor &key, const Tensor &value,
                                  std::source_location location) -> SdpaShapeInfo {
  if (query.GetRank() != 4 || key.GetRank() != 4 || value.GetRank() != 4) {
    throw InvalidArgumentError("SDPA requires rank-4 query, key, and value tensors", location);
  }
  const auto batch_size = query.GetShape().GetDimension(0, location);
  const auto query_head_count = query.GetShape().GetDimension(1, location);
  const auto query_length = query.GetShape().GetDimension(2, location);
  const auto head_dimension = query.GetShape().GetDimension(3, location);
  const auto key_value_head_count = key.GetShape().GetDimension(1, location);
  const auto key_length = key.GetShape().GetDimension(2, location);
  const auto value_dimension = value.GetShape().GetDimension(3, location);
  if (key.GetShape().GetDimension(0, location) != batch_size ||
      value.GetShape().GetDimension(0, location) != batch_size ||
      value.GetShape().GetDimension(1, location) != key_value_head_count ||
      value.GetShape().GetDimension(2, location) != key_length ||
      key.GetShape().GetDimension(3, location) != head_dimension) {
    throw InvalidArgumentError("SDPA query, key, and value dimensions are inconsistent", location);
  }
  if (query_head_count == 0 || key_value_head_count == 0 || query_head_count % key_value_head_count != 0) {
    throw InvalidArgumentError("SDPA requires positive head counts with Hq divisible by Hkv", location);
  }
  return {
      .output_shape_ = Shape{batch_size, query_head_count, query_length, value_dimension},
      .score_shape_ = Shape{batch_size, query_head_count, query_length, key_length},
      .batch_size_ = batch_size,
      .query_head_count_ = query_head_count,
      .key_value_head_count_ = key_value_head_count,
      .query_length_ = query_length,
      .key_length_ = key_length,
      .head_dimension_ = head_dimension,
      .value_dimension_ = value_dimension,
  };
}

void ValidateMask(const Tensor &mask, const Tensor &query, const Shape &score_shape, std::source_location location) {
  if (mask.GetDType() != DType::BOOL && mask.GetDType() != DType::FLOAT32 && mask.GetDType() != query.GetDType()) {
    throw InvalidArgumentError("SDPA mask must be BOOL, FLOAT32, or match the query dtype", location);
  }
  const std::array shapes{mask.GetShape(), score_shape};
  if (BroadcastShapes(shapes, location) != score_shape) {
    throw InvalidArgumentError("SDPA mask is not broadcastable to [B, Hq, Q, K]", location);
  }
}

[[nodiscard]] auto BuildSdpaParameters(Tensor &output, const Tensor &query, const Tensor &key, const Tensor &value,
                                       const std::optional<Tensor> &mask, const SdpaOptions &options,
                                       const SdpaShapeInfo &shape_info, std::source_location location)
    -> internal::SdpaParameters {
  const auto element_size = GetDTypeSize(query.GetDType(), location);
  auto parameters = internal::SdpaParameters{
      .output_ = static_cast<std::byte *>(internal::TensorAccess::GetMutableData(output, location)),
      .query_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(query, location)),
      .key_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(key, location)),
      .value_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(value, location)),
      .mask_ =
          mask.has_value() ? static_cast<const std::byte *>(internal::TensorAccess::GetData(*mask, location)) : nullptr,
      .batch_size_ = static_cast<uint64_t>(shape_info.batch_size_),
      .query_head_count_ = static_cast<uint64_t>(shape_info.query_head_count_),
      .key_value_head_count_ = static_cast<uint64_t>(shape_info.key_value_head_count_),
      .query_length_ = static_cast<uint64_t>(shape_info.query_length_),
      .key_length_ = static_cast<uint64_t>(shape_info.key_length_),
      .head_dimension_ = static_cast<uint64_t>(shape_info.head_dimension_),
      .value_dimension_ = static_cast<uint64_t>(shape_info.value_dimension_),
      .scale_ = options.scale_.value_or(1.0F / std::sqrt(static_cast<float>(shape_info.head_dimension_))),
      .mask_dtype_ = mask.has_value() ? mask->GetDType() : DType::BOOL,
      .has_mask_ = mask.has_value(),
      .causal_ = options.causal_,
      .lower_right_ = options.causal_alignment_ == CausalAlignment::LOWER_RIGHT,
  };
  for (size_t axis = 0; axis < 4; ++axis) {
    parameters.output_strides_bytes_[axis] =
        internal::CheckedBytes(output.GetStrides().GetStride(axis, location), element_size, location);
    parameters.query_strides_bytes_[axis] =
        internal::CheckedBytes(query.GetStrides().GetStride(axis, location), element_size, location);
    parameters.key_strides_bytes_[axis] =
        internal::CheckedBytes(key.GetStrides().GetStride(axis, location), element_size, location);
    parameters.value_strides_bytes_[axis] =
        internal::CheckedBytes(value.GetStrides().GetStride(axis, location), element_size, location);
  }
  if (mask.has_value()) {
    const auto mask_size = GetDTypeSize(mask->GetDType(), location);
    const auto rank_offset = size_t{4} - mask->GetRank();
    for (size_t axis = 0; axis < 4; ++axis) {
      if (axis < rank_offset) {
        parameters.mask_strides_bytes_[axis] = 0;
        continue;
      }
      const auto mask_axis = axis - rank_offset;
      parameters.mask_strides_bytes_[axis] =
          mask->GetShape().GetDimension(mask_axis, location) == 1 &&
                  shape_info.score_shape_.GetDimension(axis, location) != 1
              ? 0
              : internal::CheckedBytes(mask->GetStrides().GetStride(mask_axis, location), mask_size, location);
    }
  }
  return parameters;
}

}  // namespace

void ScaledDotProductAttentionOut(ExecutionContext &context, Tensor &output, const Tensor &query, const Tensor &key,
                                  const Tensor &value, const std::optional<Tensor> &mask, const SdpaOptions &options,
                                  std::source_location location) {
  internal::OpGuard guard{context, "ScaledDotProductAttentionOut", location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(output);
  guard.ValidateTensor(query);
  guard.ValidateTensor(key);
  guard.ValidateTensor(value);
  if (!IsFloating(query.GetDType()) || key.GetDType() != query.GetDType() || value.GetDType() != query.GetDType() ||
      output.GetDType() != query.GetDType()) {
    throw InvalidArgumentError("SDPA query, key, value, and output must have the same floating dtype", location);
  }
  const auto shape_info = InferSdpaShape(query, key, value, location);
  if (output.GetShape() != shape_info.output_shape_) {
    throw InvalidArgumentError("SDPA output shape does not match inference", location);
  }
  ValidateSdpaOptions(options, shape_info.head_dimension_, location);
  internal::ValidateWritableOutput(output, "ScaledDotProductAttentionOut", location);

  auto input_pointers = std::array<const Tensor *, 4>{&query, &key, &value, nullptr};
  auto input_count = size_t{3};
  if (mask.has_value()) {
    guard.ValidateTensor(*mask);
    ValidateMask(*mask, query, shape_info.score_shape_, location);
    input_pointers[input_count++] = &*mask;
  }
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, output,
                          std::span<const Tensor *const>{input_pointers.data(), input_count},
                          "ScaledDotProductAttentionOut", location);
  if (output.GetNumElements() == 0) {
    return;
  }

  const auto parameters = BuildSdpaParameters(output, query, key, value, mask, options, shape_info, location);
  guard.RecordTensor(output);
  guard.RecordTensor(query);
  guard.RecordTensor(key);
  guard.RecordTensor(value);
  if (mask.has_value()) {
    guard.RecordTensor(*mask);
  }
  internal::LaunchSdpa(guard.GetNativeStream(), query.GetDType(), parameters, location);
  guard.CheckLaunch();
}

auto ScaledDotProductAttention(ExecutionContext &context, const Tensor &query, const Tensor &key, const Tensor &value,
                               const std::optional<Tensor> &mask, const SdpaOptions &options,
                               std::source_location location) -> Tensor {
  Shape shape;
  {
    internal::OpGuard guard{context, "ScaledDotProductAttention", location};
    guard.ValidateTensor(query);
    guard.ValidateTensor(key);
    guard.ValidateTensor(value);
    if (!IsFloating(query.GetDType()) || key.GetDType() != query.GetDType() || value.GetDType() != query.GetDType()) {
      throw InvalidArgumentError("SDPA query, key, and value must have the same floating dtype", location);
    }
    const auto shape_info = InferSdpaShape(query, key, value, location);
    ValidateSdpaOptions(options, shape_info.head_dimension_, location);
    if (mask.has_value()) {
      guard.ValidateTensor(*mask);
      ValidateMask(*mask, query, shape_info.score_shape_, location);
    }
    shape = shape_info.output_shape_;
  }
  auto output = Empty(context, shape, query.GetDType(), location);
  ScaledDotProductAttentionOut(context, output, query, key, value, mask, options, location);
  return output;
}

}  // namespace ttl
