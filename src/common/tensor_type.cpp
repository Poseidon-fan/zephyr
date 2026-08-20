#include "common/tensor_type.h"

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
