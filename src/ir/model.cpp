#include "ir/model.h"

#include <string>
#include <utility>

namespace zephyr::ir {

Model::Model(std::string name, std::vector<std::unique_ptr<const Input>> inputs,
             std::vector<std::unique_ptr<const Parameter>> parameters,
             std::vector<std::unique_ptr<const Operation>> operations, std::vector<Value> outputs)
    : name_(std::move(name)),
      inputs_(std::move(inputs)),
      parameters_(std::move(parameters)),
      operations_(std::move(operations)),
      outputs_(std::move(outputs)) {}

auto Model::ToString() const -> std::string {
  auto operation_indices = OperationIndices{};
  size_t result_number = 0;
  for (const auto &operation : operations_) {
    operation_indices.emplace(operation.get(), result_number);
    result_number += operation->GetResultTypes().size();
  }

  auto text = std::string{"model @"};
  text.append(name_);
  text.append(" {\n");
  for (const auto &input : inputs_) {
    text.append("  ");
    text.append(input->ToString());
    text.push_back('\n');
  }
  if (!inputs_.empty() && !parameters_.empty()) {
    text.push_back('\n');
  }
  for (const auto &parameter : parameters_) {
    text.append("  ");
    text.append(parameter->ToString());
    text.push_back('\n');
  }
  if ((!inputs_.empty() || !parameters_.empty()) && !operations_.empty()) {
    text.push_back('\n');
  }
  for (const auto &operation : operations_) {
    text.append("  ");
    text.append(operation->ToString(operation_indices));
    text.push_back('\n');
  }
  text.append("  return (");
  for (size_t index = 0; index < outputs_.size(); index++) {
    if (index != 0) {
      text.append(", ");
    }
    text.append(ValueToString(outputs_[index], operation_indices));
  }
  text.append(")\n}");
  return text;
}

}  // namespace zephyr::ir
