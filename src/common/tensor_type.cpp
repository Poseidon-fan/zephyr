#include "common/tensor_type.h"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <string>
#include <string_view>

#include "common/exception.h"

namespace zephyr {

namespace {

auto DTypeToString(ttl::DType dtype) -> std::string_view {
  switch (dtype) {
    case ttl::DType::BOOL:
      return "bool";
    case ttl::DType::INT32:
      return "i32";
    case ttl::DType::INT64:
      return "i64";
    case ttl::DType::FLOAT16:
      return "f16";
    case ttl::DType::BFLOAT16:
      return "bf16";
    case ttl::DType::FLOAT32:
      return "f32";
    case ttl::DType::UINT8:
      return "u8";
  }
  throw InvalidArgumentException{"tensor type contains an invalid dtype"};
}

}  // namespace

auto GetStaticExtent(const Dimension &dimension, std::string_view name) -> int64_t {
  const auto *extent = std::get_if<int64_t>(&dimension);
  if (extent == nullptr || *extent <= 0) {
    throw InvalidArgumentException{std::string{name} + " must be a positive static dimension"};
  }
  return *extent;
}

auto ResolveShape(const Shape &shape, std::span<const DynamicDimensionBinding> bindings) -> std::vector<int64_t> {
  auto resolved = std::vector<int64_t>{};
  resolved.reserve(shape.size());
  for (const auto &dimension : shape) {
    if (const auto *extent = std::get_if<int64_t>(&dimension); extent != nullptr) {
      resolved.push_back(*extent);
      continue;
    }

    const auto &name = std::get<DynamicDimension>(dimension).name_;
    const auto iterator = std::ranges::find_if(
        bindings, [&name](const DynamicDimensionBinding &binding) { return binding.name_ == name; });
    if (iterator == bindings.end()) {
      throw InvalidArgumentException{"missing dynamic dimension binding: " + name};
    }
    if (iterator->extent_ < 0) {
      throw InvalidArgumentException{"dynamic dimension extent must not be negative: " + name};
    }
    resolved.push_back(iterator->extent_);
  }
  return resolved;
}

auto MakeFullSlice(const Shape &shape, std::string_view name) -> TensorSlice {
  auto slice = TensorSlice{};
  slice.reserve(shape.size());
  for (const auto &dimension : shape) {
    slice.push_back(TensorRange{.begin_ = 0, .end_ = GetStaticExtent(dimension, name)});
  }
  return slice;
}

auto TensorType::ToString() const -> std::string {
  auto text = std::string{"tensor<["};
  for (size_t index = 0; index < shape_.size(); index++) {
    if (index != 0) {
      text += ", ";
    }
    std::visit(
        [&text](const auto &dimension) {
          using DimensionType = std::remove_cvref_t<decltype(dimension)>;
          if constexpr (std::same_as<DimensionType, int64_t>) {
            text += std::to_string(dimension);
          } else {
            text += dimension.ToString();
          }
        },
        shape_[index]);
  }
  text += "], ";
  text += DTypeToString(dtype_);
  text += ">";
  return text;
}

}  // namespace zephyr
