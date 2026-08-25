#include "ir/operation/tensor.hpp"

#include <cstddef>
#include <utility>

namespace zephyr::ir {
namespace {

auto InferSplitTypes(const Value &input, size_t dimension, const std::vector<int64_t> &sizes)
    -> std::vector<TensorType> {
  const auto &input_type = GetType(input);
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

Add::Add(Value lhs, Value rhs) : Operation({lhs, rhs}, {internal::InferBroadcastType(lhs, rhs)}) {}

Multiply::Multiply(Value lhs, Value rhs) : Operation({lhs, rhs}, {internal::InferBroadcastType(lhs, rhs)}) {}

Silu::Silu(Value input) : Operation({input}, {GetType(input)}) {}

Sigmoid::Sigmoid(Value input) : Operation({input}, {GetType(input)}) {}

Reshape::Reshape(Value input, Shape shape)
    : Operation({input}, {TensorType{.dtype_ = GetType(input).dtype_, .shape_ = shape}}), shape_(std::move(shape)) {}

Split::Split(Value input, size_t dimension, std::vector<int64_t> sizes)
    : Operation({input}, InferSplitTypes(input, dimension, sizes)), dimension_(dimension), sizes_(std::move(sizes)) {}

auto Split::ToString(const OperationIndices &operation_indices) const -> std::string {
  auto attributes = std::string{" {dimension = "} + std::to_string(dimension_) + ", sizes = [";
  for (size_t index = 0; index < sizes_.size(); index++) {
    if (index != 0) {
      attributes += ", ";
    }
    attributes += std::to_string(sizes_[index]);
  }
  return Format(operation_indices, attributes + "]}");
}

}  // namespace zephyr::ir
