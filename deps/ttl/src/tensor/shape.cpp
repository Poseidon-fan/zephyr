#include "ttl/tensor/shape.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"

namespace ttl {
namespace {

void ValidateRank(size_t rank, std::string_view description, std::source_location location) {
  if (rank <= TTL_MAX_RANK) {
    return;
  }

  std::string message{description};
  message.append(" rank exceeds TTL_MAX_RANK: ");
  message.append(std::to_string(rank));
  throw InvalidArgumentError(std::move(message), location);
}

[[noreturn]] void ThrowNegativeValue(std::string_view description, size_t axis, int64_t value,
                                     std::source_location location) {
  std::string message{description};
  message.append(" at axis ");
  message.append(std::to_string(axis));
  message.append(" must be non-negative, got ");
  message.append(std::to_string(value));
  throw InvalidArgumentError(std::move(message), location);
}

[[noreturn]] void ThrowAxisOutOfRange(std::string_view description, size_t axis, size_t rank,
                                      std::source_location location) {
  std::string message{description};
  message.append(" axis out of range: ");
  message.append(std::to_string(axis));
  message.append(" for rank ");
  message.append(std::to_string(rank));
  throw InvalidArgumentError(std::move(message), location);
}

[[nodiscard]] auto ValuesToString(std::span<const int64_t> values) -> std::string {
  std::string result{"["};
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      result.append(", ");
    }
    result.append(std::to_string(values[index]));
  }
  result.push_back(']');
  return result;
}

}  // namespace

Shape::Shape(std::span<const int64_t> dimensions, std::source_location location) {
  ValidateRank(dimensions.size(), "shape", location);
  rank_ = static_cast<uint8_t>(dimensions.size());

  auto is_empty = false;
  for (size_t axis = 0; axis < dimensions.size(); ++axis) {
    const auto dimension = dimensions[axis];
    if (dimension < 0) {
      ThrowNegativeValue("shape dimension", axis, dimension, location);
    }
    dimensions_[axis] = dimension;
    is_empty = is_empty || dimension == 0;
  }

  if (is_empty) {
    num_elements_ = 0;
    return;
  }

  for (const auto dimension : dimensions) {
    num_elements_ = internal::CheckedMultiply(num_elements_, dimension, "shape element count", location);
  }
}

Shape::Shape(std::initializer_list<int64_t> dimensions, std::source_location location)
    : Shape(std::span<const int64_t>{dimensions.begin(), dimensions.size()}, location) {}

auto Shape::GetDimension(size_t axis, std::source_location location) const -> int64_t {
  if (axis >= rank_) {
    ThrowAxisOutOfRange("shape", axis, rank_, location);
  }
  return dimensions_[axis];
}

auto Shape::ToString() const -> std::string { return ValuesToString(GetDimensions()); }

auto operator<<(std::ostream &stream, const Shape &shape) -> std::ostream & { return stream << shape.ToString(); }

Strides::Strides(std::span<const int64_t> strides, std::source_location location) {
  ValidateRank(strides.size(), "strides", location);
  rank_ = static_cast<uint8_t>(strides.size());

  for (size_t axis = 0; axis < strides.size(); ++axis) {
    const auto stride = strides[axis];
    if (stride < 0) {
      ThrowNegativeValue("stride", axis, stride, location);
    }
    strides_[axis] = stride;
  }
}

Strides::Strides(std::initializer_list<int64_t> strides, std::source_location location)
    : Strides(std::span<const int64_t>{strides.begin(), strides.size()}, location) {}

auto Strides::GetStride(size_t axis, std::source_location location) const -> int64_t {
  if (axis >= rank_) {
    ThrowAxisOutOfRange("stride", axis, rank_, location);
  }
  return strides_[axis];
}

auto Strides::ToString() const -> std::string { return ValuesToString(GetValues()); }

auto operator<<(std::ostream &stream, const Strides &strides) -> std::ostream & { return stream << strides.ToString(); }

auto NormalizeAxis(int64_t axis, size_t rank, std::source_location location) -> size_t {
  ValidateRank(rank, "axis normalization", location);
  const auto signed_rank = static_cast<int64_t>(rank);
  if (axis < -signed_rank || axis >= signed_rank) {
    std::string message{"axis out of range: "};
    message.append(std::to_string(axis));
    message.append(" for rank ");
    message.append(std::to_string(rank));
    throw InvalidArgumentError(std::move(message), location);
  }
  return static_cast<size_t>(axis < 0 ? axis + signed_rank : axis);
}

auto NormalizeAxes(std::span<const int64_t> axes, size_t rank, std::source_location location) -> std::vector<size_t> {
  ValidateRank(rank, "axis normalization", location);
  if (axes.size() > rank) {
    std::string message{"axis count exceeds rank: "};
    message.append(std::to_string(axes.size()));
    message.append(" for rank ");
    message.append(std::to_string(rank));
    throw InvalidArgumentError(std::move(message), location);
  }

  std::vector<size_t> normalized;
  normalized.reserve(axes.size());
  for (const auto axis : axes) {
    normalized.push_back(NormalizeAxis(axis, rank, location));
  }
  std::ranges::sort(normalized);

  const auto duplicate = std::ranges::adjacent_find(normalized);
  if (duplicate != normalized.end()) {
    std::string message{"duplicate axis after normalization: "};
    message.append(std::to_string(*duplicate));
    throw InvalidArgumentError(std::move(message), location);
  }
  return normalized;
}

auto GetContiguousStrides(const Shape &shape, std::source_location location) -> Strides {
  std::array<int64_t, TTL_MAX_RANK> values{};
  const auto dimensions = shape.GetDimensions();
  auto running = int64_t{1};

  for (size_t remaining = dimensions.size(); remaining > 0; --remaining) {
    const auto axis = remaining - 1;
    values[axis] = running;
    if (axis > 0) {
      const auto extent = dimensions[axis] == 0 ? int64_t{1} : dimensions[axis];
      running = internal::CheckedMultiply(running, extent, "contiguous stride", location);
    }
  }
  return Strides{std::span<const int64_t>{values.data(), dimensions.size()}, location};
}

auto IsContiguous(const Shape &shape, const Strides &strides, std::source_location location) -> bool {
  if (shape.GetRank() != strides.GetRank()) {
    std::string message{"shape and strides rank mismatch: "};
    message.append(std::to_string(shape.GetRank()));
    message.append(" vs ");
    message.append(std::to_string(strides.GetRank()));
    throw InvalidArgumentError(std::move(message), location);
  }
  if (shape.IsEmpty()) {
    return true;
  }

  const auto dimensions = shape.GetDimensions();
  const auto values = strides.GetValues();
  auto expected_stride = int64_t{1};
  for (size_t remaining = dimensions.size(); remaining > 0; --remaining) {
    const auto axis = remaining - 1;
    if (dimensions[axis] == 1) {
      continue;
    }
    if (values[axis] != expected_stride) {
      return false;
    }
    if (axis > 0) {
      expected_stride = internal::CheckedMultiply(expected_stride, dimensions[axis], "contiguous stride", location);
    }
  }
  return true;
}

}  // namespace ttl
