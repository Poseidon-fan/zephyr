#include "ir/value.h"

#include <algorithm>
#include <cctype>
#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

#include "common/exception.h"

namespace zephyr::ir {

auto OpResult::ToString(const OperationIndices &operation_indices) const -> std::string {
  const auto iterator = operation_indices.find(operation_);
  if (iterator == operation_indices.end()) {
    throw ConfigurationException{"operation result references an unknown operation"};
  }
  return "%" + std::to_string(iterator->second + result_index_);
}

auto ValueToString(const Value &value, const OperationIndices &operation_indices) -> std::string {
  return std::visit(
      [&operation_indices](const auto &source) -> std::string {
        using Source = std::remove_cvref_t<decltype(source)>;
        if constexpr (std::same_as<Source, const Input *>) {
          if (source == nullptr) {
            throw InvalidArgumentException{"input value must not be null"};
          }
          return "%" + source->name_;
        } else if constexpr (std::same_as<Source, const Parameter *>) {
          if (source == nullptr) {
            throw InvalidArgumentException{"parameter value must not be null"};
          }
          return "@" + source->name_;
        } else {
          return source.ToString(operation_indices);
        }
      },
      value);
}

auto GetType(const Value &value) -> const TensorType & {
  return std::visit(
      [](const auto &source) -> const TensorType & {
        using Source = std::remove_cvref_t<decltype(source)>;
        if constexpr (std::same_as<Source, const Input *>) {
          if (source == nullptr) {
            throw InvalidArgumentException{"input value must not be null"};
          }
          return source->type_;
        } else if constexpr (std::same_as<Source, const Parameter *>) {
          if (source == nullptr) {
            throw InvalidArgumentException{"parameter value must not be null"};
          }
          return source->type_;
        } else {
          if (source.operation_ == nullptr) {
            throw InvalidArgumentException{"operation result must reference an operation"};
          }
          const auto result_types = source.operation_->GetResultTypes();
          if (source.result_index_ >= result_types.size()) {
            throw InvalidArgumentException{"operation result index is out of range"};
          }
          return result_types[source.result_index_];
        }
      },
      value);
}

namespace internal {

auto IsValidIdentifier(std::string_view name) noexcept -> bool {
  if (name.empty()) {
    return false;
  }
  const auto first = static_cast<unsigned char>(name.front());
  if (std::isalpha(first) == 0 && name.front() != '_') {
    return false;
  }
  return std::ranges::all_of(name.substr(1), [](char character) {
    const auto value = static_cast<unsigned char>(character);
    return std::isalnum(value) != 0 || character == '_' || character == '.';
  });
}

void VerifyTensorType(const TensorType &type) {
  for (const auto &dimension : type.shape_) {
    std::visit(
        [](const auto &extent) {
          using Extent = std::remove_cvref_t<decltype(extent)>;
          if constexpr (std::same_as<Extent, int64_t>) {
            if (extent <= 0) {
              throw InvalidArgumentException{"static tensor dimensions must be positive"};
            }
          } else if (!IsValidIdentifier(extent.name_)) {
            throw InvalidArgumentException{"dynamic dimension name is invalid: " + extent.name_};
          }
        },
        dimension);
  }
}

void VerifyParameterType(const Parameter &parameter) {
  VerifyTensorType(parameter.type_);
  if (std::ranges::any_of(parameter.type_.shape_, [](const Dimension &dimension) {
        return std::holds_alternative<DynamicDimension>(dimension);
      })) {
    throw InvalidArgumentException{"parameter shape must be static: " + parameter.name_};
  }
}

void VerifySameDType(const TensorType &lhs, const TensorType &rhs, std::string_view operation) {
  if (lhs.dtype_ != rhs.dtype_) {
    throw InvalidArgumentException{std::string{operation} + " requires matching dtypes"};
  }
}

namespace {

auto BroadcastDimension(const Dimension &lhs, const Dimension &rhs, std::string_view operation) -> Dimension {
  if (lhs == rhs) {
    return lhs;
  }
  if (const auto *extent = std::get_if<int64_t>(&lhs); extent != nullptr && *extent == 1) {
    return rhs;
  }
  if (const auto *extent = std::get_if<int64_t>(&rhs); extent != nullptr && *extent == 1) {
    return lhs;
  }
  throw InvalidArgumentException{std::string{operation} + " operands cannot be broadcast"};
}

}  // namespace

auto InferBroadcastType(const Value &lhs, const Value &rhs, std::string_view operation) -> TensorType {
  const auto &lhs_type = GetType(lhs);
  const auto &rhs_type = GetType(rhs);
  VerifyTensorType(lhs_type);
  VerifyTensorType(rhs_type);
  VerifySameDType(lhs_type, rhs_type, operation);

  const auto rank = std::max(lhs_type.shape_.size(), rhs_type.shape_.size());
  auto shape = Shape(rank, int64_t{1});
  for (size_t offset = 0; offset < rank; offset++) {
    const auto lhs_dimension =
        offset < lhs_type.shape_.size() ? lhs_type.shape_[lhs_type.shape_.size() - 1 - offset] : Dimension{int64_t{1}};
    const auto rhs_dimension =
        offset < rhs_type.shape_.size() ? rhs_type.shape_[rhs_type.shape_.size() - 1 - offset] : Dimension{int64_t{1}};
    shape[rank - 1 - offset] = BroadcastDimension(lhs_dimension, rhs_dimension, operation);
  }
  return TensorType{.dtype_ = lhs_type.dtype_, .shape_ = std::move(shape)};
}

}  // namespace internal
}  // namespace zephyr::ir
