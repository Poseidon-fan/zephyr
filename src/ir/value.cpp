#include "ir/value.h"

#include <algorithm>
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
          return "%" + source->name_;
        } else if constexpr (std::same_as<Source, const Parameter *>) {
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
        if constexpr (!std::same_as<Source, OpResult>) {
          return source->type_;
        } else {
          const auto result_types = source.operation_->GetResultTypes();
          return result_types[source.result_index_];
        }
      },
      value);
}

auto Operation::Format(const OperationIndices &operation_indices, std::string_view attributes) const -> std::string {
  const auto iterator = operation_indices.find(this);
  if (iterator == operation_indices.end()) {
    throw ConfigurationException{"operation is missing from the model result numbering"};
  }

  auto text = std::string{"%" + std::to_string(iterator->second)};
  for (size_t index = 1; index < result_types_.size(); index++) {
    text += ", %" + std::to_string(iterator->second + index);
  }
  text += " = ";
  text += GetName();
  text += "(";
  for (size_t index = 0; index < operands_.size(); index++) {
    if (index != 0) {
      text += ", ";
    }
    text += ValueToString(operands_[index], operation_indices);
  }
  text += ")";
  text += attributes;
  text += " : ";
  if (result_types_.size() == 1) {
    text += result_types_[0].ToString();
  } else {
    text += "(";
    for (size_t index = 0; index < result_types_.size(); index++) {
      if (index != 0) {
        text += ", ";
      }
      text += result_types_[index].ToString();
    }
    text += ")";
  }
  return text;
}

auto Operation::ToString(const OperationIndices &operation_indices) const -> std::string {
  return Format(operation_indices);
}

namespace internal {

namespace {

auto BroadcastDimension(const Dimension &lhs, const Dimension &rhs) -> Dimension {
  if (lhs == rhs) {
    return lhs;
  }
  if (const auto *extent = std::get_if<int64_t>(&lhs); extent != nullptr && *extent == 1) {
    return rhs;
  }
  if (const auto *extent = std::get_if<int64_t>(&rhs); extent != nullptr && *extent == 1) {
    return lhs;
  }
  return lhs;
}

}  // namespace

auto InferBroadcastType(const Value &lhs, const Value &rhs) -> TensorType {
  const auto &lhs_type = GetType(lhs);
  const auto &rhs_type = GetType(rhs);

  const auto rank = std::max(lhs_type.shape_.size(), rhs_type.shape_.size());
  auto shape = Shape(rank, int64_t{1});
  for (size_t offset = 0; offset < rank; offset++) {
    const auto lhs_dimension =
        offset < lhs_type.shape_.size() ? lhs_type.shape_[lhs_type.shape_.size() - 1 - offset] : Dimension{int64_t{1}};
    const auto rhs_dimension =
        offset < rhs_type.shape_.size() ? rhs_type.shape_[rhs_type.shape_.size() - 1 - offset] : Dimension{int64_t{1}};
    shape[rank - 1 - offset] = BroadcastDimension(lhs_dimension, rhs_dimension);
  }
  return TensorType{.dtype_ = lhs_type.dtype_, .shape_ = std::move(shape)};
}

}  // namespace internal
}  // namespace zephyr::ir
