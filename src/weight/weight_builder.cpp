#include "weight/weight_builder.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <ttl/ops/cast.hpp>
#include <ttl/ops/copy.hpp>

#include "common/exception.hpp"

namespace zephyr::weight {
auto Shard::Uniform(size_t axis, size_t rank, size_t world_size) -> Shard {
  if (world_size == 0) {
    throw InvalidArgumentException("uniform shard world size must be non-zero");
  }
  if (rank >= world_size) {
    throw InvalidArgumentException("uniform shard rank must be less than world size");
  }
  return Shard{UniformSpec{.axis_ = axis, .rank_ = rank, .world_size_ = world_size}};
}

auto Shard::Range(size_t axis, size_t offset, size_t length) -> Shard {
  if (length > std::numeric_limits<size_t>::max() - offset) {
    throw InvalidArgumentException("range shard offset and length overflow");
  }
  return Shard{RangeSpec{.axis_ = axis, .offset_ = offset, .length_ = length}};
}

WeightBuilder::WeightBuilder(const Checkpoint &checkpoint, ttl::Runtime &runtime, ttl::DType target_dtype)
    : checkpoint_(checkpoint), runtime_(runtime), target_dtype_(target_dtype) {}

auto WeightBuilder::PushPrefix(std::string_view component) const -> WeightBuilder {
  if (component.empty()) {
    throw InvalidArgumentException("weight prefix component must not be empty");
  }
  auto result = *this;
  result.prefix_.append(component);
  result.prefix_.push_back('.');
  return result;
}

auto WeightBuilder::PushPrefix(size_t component) const -> WeightBuilder {
  const auto text = std::to_string(component);
  return PushPrefix(text);
}

auto WeightBuilder::WithDType(ttl::DType target_dtype) const -> WeightBuilder {
  auto result = *this;
  result.target_dtype_ = target_dtype;
  return result;
}

auto WeightBuilder::GetParameterInfo(std::string_view parameter_name) const -> std::optional<ParameterInfo> {
  return checkpoint_.GetParameterInfo(FullName(parameter_name));
}

auto WeightBuilder::ResolveShard(const ttl::Shape &shape, const std::optional<Shard> &shard)
    -> std::optional<ResolvedShard> {
  if (!shard.has_value()) {
    return std::nullopt;
  }

  return std::visit(
      [&shape](const auto &spec) -> std::optional<ResolvedShard> {
        const auto rank = shape.GetRank();
        if (spec.axis_ >= rank) {
          throw InvalidArgumentException("weight shard axis is outside the parameter rank");
        }
        const auto dimension_value = shape.GetDimension(spec.axis_);
        if (dimension_value < 0 || static_cast<uint64_t>(dimension_value) > std::numeric_limits<size_t>::max()) {
          throw InternalException("parameter dimension cannot be represented by the host");
        }
        const auto dimension = static_cast<size_t>(dimension_value);

        using Spec = std::decay_t<decltype(spec)>;
        if constexpr (std::is_same_v<Spec, Shard::UniformSpec>) {
          if (spec.world_size_ == 1) {
            return std::nullopt;
          }
          if (dimension % spec.world_size_ != 0) {
            throw InvalidArgumentException("weight shard axis is not divisible by world size");
          }
          const auto length = dimension / spec.world_size_;
          return ResolvedShard{spec.axis_, spec.rank_ * length, length};
        } else {
          if (spec.offset_ > dimension || spec.length_ > dimension - spec.offset_) {
            throw InvalidArgumentException("weight shard range exceeds the parameter dimension");
          }
          if (spec.offset_ == 0 && spec.length_ == dimension) {
            return std::nullopt;
          }
          return ResolvedShard{spec.axis_, spec.offset_, spec.length_};
        }
      },
      shard->spec_);
}

auto WeightBuilder::FullName(std::string_view parameter_name) const -> std::string {
  if (parameter_name.empty()) {
    throw InvalidArgumentException("weight parameter name must not be empty");
  }
  if (prefix_.empty()) {
    return std::string{parameter_name};
  }
  std::string result{prefix_};
  result.append(parameter_name);
  return result;
}

auto WeightBuilder::Get(ttl::ExecutionContext &context, const ttl::Shape &expected_shape,
                        std::string_view parameter_name, std::optional<Shard> shard) const -> ttl::Tensor {
  const auto full_name = FullName(parameter_name);
  const auto *record = checkpoint_.FindParameter(full_name);
  if (record == nullptr) {
    std::string message{"weight parameter '"};
    message.append(full_name);
    message.append("' was not found in the checkpoint");
    throw ConfigurationException(message);
  }
  if (record->info_.shape_ != expected_shape) {
    std::string message{"shape mismatch for weight parameter '"};
    message.append(full_name);
    message.append("'");
    throw ConfigurationException(message);
  }

  const auto resolved_shard = ResolveShard(record->info_.shape_, shard);
  auto output_shape = expected_shape;
  if (resolved_shard.has_value()) {
    const auto expected_dimensions = expected_shape.GetDimensions();
    auto dimensions = std::vector<int64_t>{expected_dimensions.begin(), expected_dimensions.end()};
    dimensions[resolved_shard->axis_] = static_cast<int64_t>(resolved_shard->length_);
    output_shape = ttl::Shape{std::span<const int64_t>{dimensions.data(), dimensions.size()}};
  }
  const auto element_bytes = ttl::GetDTypeInfo(record->info_.dtype_).size_bytes_;
  const auto element_count = output_shape.GetNumElements();
  if (element_count < 0 || static_cast<uint64_t>(element_count) > std::numeric_limits<size_t>::max() / element_bytes) {
    throw InternalException("weight tensor byte size cannot be represented by the host");
  }
  const auto output_bytes = static_cast<size_t>(element_count) * element_bytes;
  const auto source_bytes = checkpoint_.GetParameterBytes(*record);
  if (!resolved_shard.has_value()) {
    if (output_bytes != source_bytes.size()) {
      throw InternalException("checkpoint parameter byte size disagrees with its metadata");
    }
  }

  auto staging = runtime_.AllocatePinned(output_bytes);
  auto destination = staging.AsBytes();
  if (!resolved_shard.has_value()) {
    if (output_bytes != 0) {
      std::memcpy(destination.data(), source_bytes.data(), output_bytes);
    }
  } else if (output_bytes != 0) {
    // Safetensors stores tensors in row-major order. A shard along an arbitrary
    // axis is therefore a sequence of contiguous inner rows for each outer row;
    // gathering those rows avoids materializing the full tensor on the device.
    const auto dimensions = record->info_.shape_.GetDimensions();
    const auto axis = resolved_shard->axis_;
    // ResolveShard validated this dimension and the selected range already.
    const auto source_dimension = static_cast<size_t>(dimensions[axis]);

    // ParseSafetensorsFile has already checked the complete element and byte
    // extents. The following products are prefixes of that validated extent,
    // so the row-major addresses cannot overflow size_t.
    size_t outer_elements = 1;
    for (const auto dimension : dimensions.first(axis)) {
      outer_elements *= static_cast<size_t>(dimension);
    }
    size_t inner_elements = 1;
    for (const auto dimension : dimensions.subspan(axis + 1)) {
      inner_elements *= static_cast<size_t>(dimension);
    }
    const auto row_bytes = outer_elements == 0 ? size_t{0} : resolved_shard->length_ * inner_elements * element_bytes;
    if (outer_elements * row_bytes != output_bytes) {
      throw InternalException("shard shape disagrees with its byte extent");
    }

    for (size_t outer = 0; outer < outer_elements; ++outer) {
      const auto source_row = (outer * source_dimension) + resolved_shard->start_;
      const auto source_offset = source_row * inner_elements * element_bytes;
      const auto destination_offset = outer * row_bytes;
      if (source_offset > source_bytes.size() || row_bytes > source_bytes.size() - source_offset) {
        throw InternalException("shard row extends beyond the checkpoint parameter");
      }
      if (row_bytes != 0) {
        std::memcpy(destination.data() + destination_offset, source_bytes.data() + source_offset, row_bytes);
      }
    }
  }

  auto tensor = ttl::Empty(context, output_shape, record->info_.dtype_);
  ttl::CopyFromPinnedAsync(context, tensor, staging);
  if (record->info_.dtype_ == target_dtype_) {
    return tensor;
  }
  return ttl::Cast(context, tensor, target_dtype_);
}

}  // namespace zephyr::weight
