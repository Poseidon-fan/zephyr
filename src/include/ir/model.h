#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ir/value.h"

namespace zephyr::ir {

/** Owns one complete model SSA graph and its public tensor ports. */
class Model final {
 public:
  /** Takes ownership of all graph nodes; values retain non-owning node references. */
  Model(std::string name, std::vector<std::unique_ptr<const Input>> inputs,
        std::vector<std::unique_ptr<const Parameter>> parameters,
        std::vector<std::unique_ptr<const Operation>> operations, std::vector<Value> outputs);

  Model(const Model &) = delete;
  auto operator=(const Model &) -> Model & = delete;
  Model(Model &&) noexcept = default;
  auto operator=(Model &&) noexcept -> Model & = default;

  [[nodiscard]] auto GetName() const -> std::string_view { return name_; }
  [[nodiscard]] auto GetInputs() const -> std::span<const std::unique_ptr<const Input>> { return inputs_; }
  [[nodiscard]] auto GetParameters() const -> std::span<const std::unique_ptr<const Parameter>> { return parameters_; }
  [[nodiscard]] auto GetOperations() const -> std::span<const std::unique_ptr<const Operation>> { return operations_; }
  [[nodiscard]] auto GetOutputs() const -> std::span<const Value> { return outputs_; }

  /** Returns the canonical textual representation of this model. */
  [[nodiscard]] auto ToString() const -> std::string;

 private:
  /** User-visible model name. */
  std::string name_;

  /** Inputs in public port order. */
  std::vector<std::unique_ptr<const Input>> inputs_;

  /** Parameters in checkpoint/name order. */
  std::vector<std::unique_ptr<const Parameter>> parameters_;

  /** Operations in topological order. */
  std::vector<std::unique_ptr<const Operation>> operations_;

  /** Outputs in public port order. */
  std::vector<Value> outputs_;
};

}  // namespace zephyr::ir
