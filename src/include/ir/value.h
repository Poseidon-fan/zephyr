#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "common/tensor_type.h"

namespace zephyr::ir {

class Operation;

/** Maps each operation to the first SSA number assigned to its results while printing a model. */
using OperationIndices = std::unordered_map<const Operation *, size_t>;

/** A named tensor supplied by the caller. */
struct Input final {
  /** Stable model-facing input name. */
  std::string name_;

  /** Input element type and logical shape. */
  TensorType type_;

  /** Returns the complete input declaration. */
  [[nodiscard]] auto ToString() const -> std::string { return "input %" + name_ + " : " + type_.ToString(); }
};

/** A named, immutable model parameter backed by a checkpoint tensor. */
struct Parameter final {
  /** Checkpoint tensor name used by WeightPlan. */
  std::string name_;

  /** Static parameter element type and shape. */
  TensorType type_;

  /** Returns the complete parameter declaration. */
  [[nodiscard]] auto ToString() const -> std::string {
    return "parameter @" + name_ + " : " + type_.ToString();
  }
};

/** Identifies one result produced by an Operation. */
struct OpResult final {
  /** Operation that defines this result. */
  const Operation *operation_;

  /** Zero-based result position within the defining operation. */
  uint32_t result_index_;

  [[nodiscard]] auto operator==(const OpResult &) const -> bool = default;

  /** Returns the canonical SSA result reference in a model context. */
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string;
};

/** One SSA value in the model graph. */
using Value = std::variant<const Input *, const Parameter *, OpResult>;

/** Base class for one typed operation in the model SSA graph. */
class Operation {
 public:
  Operation(const Operation &) = delete;
  auto operator=(const Operation &) -> Operation & = delete;
  Operation(Operation &&) = delete;
  auto operator=(Operation &&) -> Operation & = delete;
  virtual ~Operation() = default;

  /** Returns operands in the operation's canonical order. */
  [[nodiscard]] auto GetOperands() const -> std::span<const Value> { return operands_; }

  /** Returns result types in result-index order. */
  [[nodiscard]] auto GetResultTypes() const -> std::span<const TensorType> { return result_types_; }

  /** Returns the stable dialect-qualified operation name. */
  [[nodiscard]] virtual auto GetName() const -> std::string_view = 0;

  /** Returns this operation in canonical textual form. */
  [[nodiscard]] virtual auto ToString(const OperationIndices &operation_indices) const -> std::string = 0;

  /** Checks operation-specific type and attribute invariants. */
  virtual void Verify() const = 0;

 protected:
  /** Constructs an operation with canonical operands and inferred result types. */
  Operation(std::vector<Value> operands, std::vector<TensorType> result_types)
      : operands_(std::move(operands)), result_types_(std::move(result_types)) {}

 private:
  /** Values consumed by this operation. */
  std::vector<Value> operands_;

  /** Types of values produced by this operation. */
  std::vector<TensorType> result_types_;
};

/** Returns the logical tensor type carried by an SSA value. */
[[nodiscard]] auto GetType(const Value &value) -> const TensorType &;

/** Returns the canonical textual reference for an SSA value. */
[[nodiscard]] auto ValueToString(const Value &value, const OperationIndices &operation_indices) -> std::string;

namespace internal {

[[nodiscard]] auto IsValidIdentifier(std::string_view name) noexcept -> bool;

void VerifyTensorType(const TensorType &type);
void VerifyParameterType(const Parameter &parameter);
void VerifySameDType(const TensorType &lhs, const TensorType &rhs, std::string_view operation);

[[nodiscard]] auto InferBroadcastType(const Value &lhs, const Value &rhs, std::string_view operation) -> TensorType;

}  // namespace internal

}  // namespace zephyr::ir
