#include "ir/value.h"

#include <algorithm>
#include <cctype>
#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

#include "common/exception.h"

namespace zephyr::ir {

void Value::AddUse(const Operation *user, uint32_t operand_index) const {
  if (user == nullptr) {
    throw InvalidArgumentException{"value use must reference an operation"};
  }
  users_.push_back(Use{.user_ = user, .operand_index_ = operand_index});
}

void Value::ClearUses() const noexcept { users_.clear(); }

void Value::RemoveUses(const Operation *user) const noexcept {
  std::erase_if(users_, [user](const Use &use) { return use.user_ == user; });
}

Input::Input(std::string name, TensorType type) : Value(std::move(type)), name_(std::move(name)) {}

auto Input::ToString(const OperationIndices & /*operation_indices*/) const -> std::string { return "%" + name_; }

Parameter::Parameter(std::string name, TensorType type) : Value(std::move(type)), name_(std::move(name)) {}

auto Parameter::ToString(const OperationIndices & /*operation_indices*/) const -> std::string { return "@" + name_; }

OperationResult::OperationResult(const Operation *operation, uint32_t result_index, TensorType type)
    : Value(std::move(type)), operation_(operation), result_index_(result_index) {}

auto OperationResult::ToString(const OperationIndices &operation_indices) const -> std::string {
  const auto iterator = operation_indices.find(operation_);
  if (iterator == operation_indices.end()) {
    throw ConfigurationException{"operation result references an unknown operation"};
  }
  return "%" + std::to_string(iterator->second + result_index_);
}

Operation::Operation(std::vector<const Value *> operands, std::vector<TensorType> result_types)
    : operands_(std::move(operands)), result_types_(std::move(result_types)) {
  results_.reserve(result_types_.size());
  for (uint32_t index = 0; index < result_types_.size(); index++) {
    results_.push_back(std::unique_ptr<OperationResult>(new OperationResult(this, index, result_types_[index])));
  }
}

Operation::~Operation() {
  for (const auto *operand : operands_) {
    if (operand != nullptr) {
      operand->RemoveUses(this);
    }
  }
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
  VerifyTensorType(parameter.GetType());
  if (std::ranges::any_of(parameter.GetType().shape_, [](const Dimension &dimension) {
        return std::holds_alternative<DynamicDimension>(dimension);
      })) {
    throw InvalidArgumentException{"parameter shape must be static"};
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
  const auto &lhs_type = lhs.GetType();
  const auto &rhs_type = rhs.GetType();
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
