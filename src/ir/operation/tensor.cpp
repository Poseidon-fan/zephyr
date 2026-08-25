#include "ir/operation/tensor.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "common/exception.h"

namespace zephyr::ir {
namespace {

auto RequireValue(const Value *value, std::string_view operation) -> const Value & {
  if (value == nullptr) {
    throw InvalidArgumentException{std::string{operation} + " operand must not be null"};
  }
  return *value;
}

auto FormatOperands(const Operation &operation, const OperationIndices &operation_indices) -> std::string {
  auto text = std::string{"("};
  for (size_t index = 0; index < operation.GetOperands().size(); index++) {
    if (index != 0) {
      text += ", ";
    }
    text += operation.GetOperands()[index]->ToString(operation_indices);
  }
  return text + ")";
}

auto FormatResultTypes(const Operation &operation) -> std::string {
  const auto result_types = operation.GetResultTypes();
  auto text = std::string{" : "};
  if (result_types.size() == 1) {
    return text + result_types[0].ToString();
  }
  text += "(";
  for (size_t index = 0; index < result_types.size(); index++) {
    if (index != 0) {
      text += ", ";
    }
    text += result_types[index].ToString();
  }
  return text + ")";
}

auto FormatResultNames(const Operation &operation, const OperationIndices &operation_indices) -> std::string {
  const auto iterator = operation_indices.find(&operation);
  if (iterator == operation_indices.end()) {
    throw ConfigurationException{"operation is missing from the model result numbering"};
  }
  auto text = std::string{};
  for (size_t index = 0; index < operation.GetResultTypes().size(); index++) {
    if (index != 0) {
      text += ", ";
    }
    text += "%" + std::to_string(iterator->second + index);
  }
  return text;
}

void VerifyResultTypes(const Operation &operation, std::span<const TensorType> expected) {
  if (!std::ranges::equal(operation.GetResultTypes(), expected)) {
    throw InvalidArgumentException{std::string{operation.GetName()} + " has inconsistent result types"};
  }
}

auto InferArithmeticType(const Value &lhs, const Value &rhs, std::string_view operation) -> TensorType {
  auto result = internal::InferBroadcastType(lhs, rhs, operation);
  if (result.dtype_ == ttl::DType::BOOL) {
    throw InvalidArgumentException{std::string{operation} + " does not accept boolean tensors"};
  }
  return result;
}

auto InferActivationType(const Value &input, std::string_view operation) -> TensorType {
  const auto &type = input.GetType();
  internal::VerifyTensorType(type);
  if (!ttl::IsFloating(type.dtype_)) {
    throw InvalidArgumentException{std::string{operation} + " requires a floating-point tensor"};
  }
  return type;
}

struct ShapeSignature final {
  uint64_t static_factor_{1};
  std::unordered_map<std::string, uint64_t> dynamic_factors_;

  [[nodiscard]] auto operator==(const ShapeSignature &) const -> bool = default;
};

auto GetShapeSignature(const Shape &shape) -> ShapeSignature {
  auto signature = ShapeSignature{};
  for (const auto &dimension : shape) {
    if (const auto *extent = std::get_if<int64_t>(&dimension); extent != nullptr) {
      if (*extent <= 0 || signature.static_factor_ > std::numeric_limits<uint64_t>::max() / *extent) {
        throw InvalidArgumentException{"reshape element count overflows"};
      }
      signature.static_factor_ *= static_cast<uint64_t>(*extent);
    } else {
      const auto &name = std::get<DynamicDimension>(dimension).name_;
      auto &count = signature.dynamic_factors_[name];
      if (count == std::numeric_limits<uint64_t>::max()) {
        throw InvalidArgumentException{"reshape dynamic dimension count overflows"};
      }
      count++;
    }
  }
  return signature;
}

auto InferReshapeType(const Value &input, const Shape &shape) -> TensorType {
  const auto &input_type = input.GetType();
  internal::VerifyTensorType(input_type);
  const auto output_type = TensorType{.dtype_ = input_type.dtype_, .shape_ = shape};
  internal::VerifyTensorType(output_type);
  if (GetShapeSignature(input_type.shape_) != GetShapeSignature(shape)) {
    throw InvalidArgumentException{"tensor.reshape must preserve element count"};
  }
  return output_type;
}

auto InferSplitTypes(const Value &input, size_t dimension, const std::vector<int64_t> &sizes)
    -> std::vector<TensorType> {
  const auto &input_type = input.GetType();
  internal::VerifyTensorType(input_type);
  const auto *extent = std::get_if<int64_t>(&input_type.shape_[dimension]);
  if (extent == nullptr) {
    throw InvalidArgumentException{"tensor.split target dimension must be static"};
  }
  if (sizes.empty()) {
    throw InvalidArgumentException{"tensor.split requires at least one result size"};
  }

  int64_t total = 0;
  for (const auto size : sizes) {
    if (size <= 0 || total > std::numeric_limits<int64_t>::max() - size) {
      throw InvalidArgumentException{"tensor.split sizes must be positive and representable"};
    }
    total += size;
  }
  if (total != *extent) {
    throw InvalidArgumentException{"tensor.split sizes do not cover the target dimension"};
  }

  auto results = std::vector<TensorType>{};
  results.reserve(sizes.size());
  for (const auto size : sizes) {
    auto shape = input_type.shape_;
    shape[dimension] = size;
    results.push_back(TensorType{.dtype_ = input_type.dtype_, .shape_ = std::move(shape)});
  }
  return results;
}

}  // namespace

Add::Add(const Value *lhs, const Value *rhs)
    : Operation({lhs, rhs}, {InferArithmeticType(RequireValue(lhs, NAME), RequireValue(rhs, NAME), NAME)}) {}

void Add::Verify() const {
  const auto operands = GetOperands();
  const auto expected = InferArithmeticType(*operands[0], *operands[1], NAME);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto Add::ToString(const OperationIndices &operation_indices) const -> std::string {
  return FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
         FormatOperands(*this, operation_indices) + FormatResultTypes(*this);
}

Multiply::Multiply(const Value *lhs, const Value *rhs)
    : Operation({lhs, rhs}, {InferArithmeticType(RequireValue(lhs, NAME), RequireValue(rhs, NAME), NAME)}) {}

void Multiply::Verify() const {
  const auto operands = GetOperands();
  const auto expected = InferArithmeticType(*operands[0], *operands[1], NAME);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto Multiply::ToString(const OperationIndices &operation_indices) const -> std::string {
  return FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
         FormatOperands(*this, operation_indices) + FormatResultTypes(*this);
}

Silu::Silu(const Value *input) : Operation({input}, {InferActivationType(RequireValue(input, NAME), NAME)}) {}

void Silu::Verify() const {
  const auto expected = InferActivationType(*GetOperands()[0], NAME);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto Silu::ToString(const OperationIndices &operation_indices) const -> std::string {
  return FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
         FormatOperands(*this, operation_indices) + FormatResultTypes(*this);
}

Sigmoid::Sigmoid(const Value *input) : Operation({input}, {InferActivationType(RequireValue(input, NAME), NAME)}) {}

void Sigmoid::Verify() const {
  const auto expected = InferActivationType(*GetOperands()[0], NAME);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto Sigmoid::ToString(const OperationIndices &operation_indices) const -> std::string {
  return FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
         FormatOperands(*this, operation_indices) + FormatResultTypes(*this);
}

Reshape::Reshape(const Value *input, Shape shape)
    : Operation({input}, {InferReshapeType(RequireValue(input, NAME), shape)}), shape_(std::move(shape)) {}

void Reshape::Verify() const {
  const auto expected = InferReshapeType(*GetOperands()[0], shape_);
  VerifyResultTypes(*this, std::span{&expected, 1});
}

auto Reshape::ToString(const OperationIndices &operation_indices) const -> std::string {
  return FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
         FormatOperands(*this, operation_indices) + FormatResultTypes(*this);
}

Split::Split(const Value *input, size_t dimension, std::vector<int64_t> sizes)
    : Operation({input}, InferSplitTypes(RequireValue(input, NAME), dimension, sizes)),
      dimension_(dimension),
      sizes_(std::move(sizes)) {}

void Split::Verify() const {
  const auto expected = InferSplitTypes(*GetOperands()[0], dimension_, sizes_);
  VerifyResultTypes(*this, expected);
}

auto Split::ToString(const OperationIndices &operation_indices) const -> std::string {
  auto text = FormatResultNames(*this, operation_indices) + " = " + std::string{NAME} +
              FormatOperands(*this, operation_indices) + " {dimension = " + std::to_string(dimension_) + ", sizes = [";
  for (size_t index = 0; index < sizes_.size(); index++) {
    if (index != 0) {
      text += ", ";
    }
    text += std::to_string(sizes_[index]);
  }
  return text + "]}" + FormatResultTypes(*this);
}

}  // namespace zephyr::ir
