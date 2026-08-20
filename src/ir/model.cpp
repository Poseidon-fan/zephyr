#include "ir/model.h"

#include <concepts>
#include <string>
#include <unordered_set>
#include <utility>

#include "common/exception.h"

namespace zephyr::ir {
namespace {

void VerifyValueOwnership(const Value &value, const std::unordered_set<const Input *> &inputs,
                          const std::unordered_set<const Parameter *> &parameters,
                          const std::unordered_set<const Operation *> &operations,
                          const std::unordered_set<const Operation *> &previous_operations) {
  std::visit(
      [&](const auto &source) {
        using Source = std::remove_cvref_t<decltype(source)>;
        if constexpr (std::same_as<Source, const Input *>) {
          if (source == nullptr || !inputs.contains(source)) {
            throw ConfigurationException{"value references an input outside its model"};
          }
        } else if constexpr (std::same_as<Source, const Parameter *>) {
          if (source == nullptr || !parameters.contains(source)) {
            throw ConfigurationException{"value references a parameter outside its model"};
          }
        } else if (source.operation_ == nullptr || !operations.contains(source.operation_) ||
                   !previous_operations.contains(source.operation_) ||
                   source.result_index_ >= source.operation_->GetResultTypes().size()) {
          throw ConfigurationException{"value references an invalid operation result"};
        }
      },
      value);
}

}  // namespace

Model::Model(std::string name, std::vector<std::unique_ptr<const Input>> inputs,
             std::vector<std::unique_ptr<const Parameter>> parameters,
             std::vector<std::unique_ptr<const Operation>> operations, std::vector<Value> outputs)
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
    internal::VerifyTensorType(input->type_);
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
    for (const auto &operand : operation->GetOperands()) {
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
  for (const auto &output : outputs_) {
    VerifyValueOwnership(output, inputs, parameters, operations, previous_operations);
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
