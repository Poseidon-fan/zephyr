#include "ir/model.h"

#include <string>
#include <unordered_set>
#include <utility>

#include "common/exception.h"

namespace zephyr::ir {
namespace {

void VerifyValueOwnership(const Value *value, const std::unordered_set<const Input *> &inputs,
                          const std::unordered_set<const Parameter *> &parameters,
                          const std::unordered_set<const Operation *> &operations,
                          const std::unordered_set<const Operation *> &previous_operations) {
  if (value == nullptr) {
    throw ConfigurationException{"model contains a null value"};
  }
  if (const auto *input = dynamic_cast<const Input *>(value); input != nullptr) {
    if (!inputs.contains(input)) {
      throw ConfigurationException{"value references an input outside its model"};
    }
    return;
  }
  if (const auto *parameter = dynamic_cast<const Parameter *>(value); parameter != nullptr) {
    if (!parameters.contains(parameter)) {
      throw ConfigurationException{"value references a parameter outside its model"};
    }
    return;
  }
  const auto *result = dynamic_cast<const OperationResult *>(value);
  if (result == nullptr || result->GetOperation() == nullptr || !operations.contains(result->GetOperation()) ||
      !previous_operations.contains(result->GetOperation()) ||
      result->GetResultIndex() >= result->GetOperation()->GetResultCount() ||
      &result->GetOperation()->GetResult(result->GetResultIndex()) != result) {
    throw ConfigurationException{"value references an invalid operation result"};
  }
}

}  // namespace

Model::Model(std::string name, std::vector<std::unique_ptr<const Input>> inputs,
             std::vector<std::unique_ptr<const Parameter>> parameters,
             std::vector<std::unique_ptr<const Operation>> operations, std::vector<const Value *> outputs)
    : name_(std::move(name)),
      inputs_(std::move(inputs)),
      parameters_(std::move(parameters)),
      operations_(std::move(operations)),
      outputs_(std::move(outputs)) {}

void Model::Verify() const {
  if (!internal::IsValidIdentifier(name_)) {
    throw ConfigurationException{"model name is invalid"};
  }

  auto inputs = std::unordered_set<const Input *>{};
  auto input_names = std::unordered_set<std::string>{};
  for (const auto &input : inputs_) {
    if (input == nullptr || !internal::IsValidIdentifier(input->name_) || !input_names.insert(input->name_).second) {
      throw ConfigurationException{"model input is null, duplicated, or has an invalid name"};
    }
    inputs.insert(input.get());
    internal::VerifyTensorType(input->GetType());
  }

  auto parameters = std::unordered_set<const Parameter *>{};
  auto parameter_names = std::unordered_set<std::string>{};
  for (const auto &parameter : parameters_) {
    if (parameter == nullptr || !internal::IsValidIdentifier(parameter->name_) ||
        !parameter_names.insert(parameter->name_).second) {
      throw ConfigurationException{"model parameter is null, duplicated, or has an invalid name"};
    }
    parameters.insert(parameter.get());
    internal::VerifyParameterType(*parameter);
  }
  for (const auto &input : inputs_) {
    if (parameter_names.contains(input->name_)) {
      throw ConfigurationException{"input and parameter names must be unique"};
    }
  }

  auto operations = std::unordered_set<const Operation *>{};
  for (const auto &operation : operations_) {
    if (operation == nullptr || !operations.insert(operation.get()).second) {
      throw ConfigurationException{"model operation is null or duplicated"};
    }
  }

  auto previous_operations = std::unordered_set<const Operation *>{};
  for (const auto &operation : operations_) {
    for (const auto *operand : operation->GetOperands()) {
      VerifyValueOwnership(operand, inputs, parameters, operations, previous_operations);
    }
    operation->Verify();
    if (operation->GetResultTypes().empty()) {
      throw ConfigurationException{"model operation must produce at least one result"};
    }
    previous_operations.insert(operation.get());
  }

  if (outputs_.empty()) {
    throw ConfigurationException{"model must have at least one output"};
  }
  for (const auto *output : outputs_) {
    VerifyValueOwnership(output, inputs, parameters, operations, previous_operations);
  }

  // Rebuild use-lists after all ownership and topological checks succeed.  Repeated Verify() calls are therefore
  // idempotent, and malformed external pointers can never enter a valid model's reverse edges.
  for (const auto &input : inputs_) {
    input->ClearUses();
  }
  for (const auto &parameter : parameters_) {
    parameter->ClearUses();
  }
  for (const auto &operation : operations_) {
    for (size_t index = 0; index < operation->GetResultCount(); index++) {
      operation->GetResult(static_cast<uint32_t>(index)).ClearUses();
    }
  }
  for (const auto &operation : operations_) {
    uint32_t operand_index = 0;
    for (const auto *operand : operation->GetOperands()) {
      operand->AddUse(operation.get(), operand_index++);
    }
  }
}

auto Model::ToString() const -> std::string {
  Verify();
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
    text.append("input %" + input->name_ + " : " + input->GetType().ToString());
    text.push_back('\n');
  }
  if (!inputs_.empty() && !parameters_.empty()) {
    text.push_back('\n');
  }
  for (const auto &parameter : parameters_) {
    text.append("  ");
    text.append("parameter @" + parameter->name_ + " : " + parameter->GetType().ToString());
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
    text.append(outputs_[index]->ToString(operation_indices));
  }
  text.append(")\n}");
  return text;
}

}  // namespace zephyr::ir
