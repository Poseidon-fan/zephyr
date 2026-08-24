#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/tensor_type.h"

namespace zephyr::ir {

class Model;
class Operation;

/** Maps each operation to the first SSA number assigned to its results while printing a model. */
using OperationIndices = std::unordered_map<const Operation *, size_t>;

/** One reverse SSA edge from a value to an operand slot. */
struct Use final {
  /** Operation consuming the value. */
  const Operation *user_;

  /** Operand position in user_->GetOperands(). */
  uint32_t operand_index_;
};

/** Base class for every typed SSA value in the IR. */
class Value {
 public:
  Value(const Value &) = delete;
  auto operator=(const Value &) -> Value & = delete;
  Value(Value &&) = delete;
  auto operator=(Value &&) -> Value & = delete;
  virtual ~Value() = default;

  /** All operation operands that use this value. */
  [[nodiscard]] auto GetUsers() const -> std::span<const Use> { return users_; }

  /** Element type and logical shape, immutable after construction. */
  [[nodiscard]] auto GetType() const -> const TensorType & { return type_; }

  /** Returns the canonical textual reference for this value. */
  [[nodiscard]] virtual auto ToString(const OperationIndices &operation_indices) const -> std::string = 0;

 protected:
  explicit Value(TensorType type) : type_(std::move(type)) {}

 private:
  friend class Model;
  friend class Operation;

  void AddUse(const Operation *user, uint32_t operand_index) const;
  void ClearUses() const noexcept;
  void RemoveUses(const Operation *user) const noexcept;

  const TensorType type_;
  mutable std::vector<Use> users_;
};

/** A named tensor supplied by the caller. */
class Input final : public Value {
 public:
  Input(std::string name, TensorType type);

  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;

 private:
  friend class Model;

  /** Stable model-facing input name. */
  std::string name_;
};

/** A named, immutable model parameter backed by a checkpoint tensor. */
class Parameter final : public Value {
 public:
  Parameter(std::string name, TensorType type);

  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;

 private:
  friend class Model;

  /** Checkpoint tensor name used by WeightPlan. */
  std::string name_;
};

/** One result produced by an Operation. */
class OperationResult final : public Value {
 public:
  /** Operation defining this result. */
  [[nodiscard]] auto GetOperation() const -> const Operation * { return operation_; }

  /** Zero-based result position within the defining operation. */
  [[nodiscard]] auto GetResultIndex() const -> uint32_t { return result_index_; }

  /** Returns the canonical SSA result reference in a model context. */
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;

 private:
  friend class Operation;

  OperationResult(const Operation *operation, uint32_t result_index, TensorType type);

  const Operation *operation_;
  uint32_t result_index_;
};

/** Base class for one typed operation in the model SSA graph. */
class Operation {
 public:
  Operation(const Operation &) = delete;
  auto operator=(const Operation &) -> Operation & = delete;
  Operation(Operation &&) = delete;
  auto operator=(Operation &&) -> Operation & = delete;
  virtual ~Operation();

  /** Returns operands in the operation's canonical order. */
  [[nodiscard]] auto GetOperands() const -> std::span<const Value *const> { return operands_; }

  /** Returns the number of results produced by this operation. */
  [[nodiscard]] auto GetResultCount() const noexcept -> size_t { return results_.size(); }

  /** Returns one result value in result-index order. */
  [[nodiscard]] auto GetResult(uint32_t result_index) const -> const OperationResult & {
    return *results_.at(result_index);
  }

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
  Operation(std::vector<const Value *> operands, std::vector<TensorType> result_types);

 private:
  /** Values consumed by this operation. */
  std::vector<const Value *> operands_;

  /** Types of values produced by this operation. */
  std::vector<TensorType> result_types_;

  /** Stable result objects whose addresses are used as SSA identities. */
  std::vector<std::unique_ptr<OperationResult>> results_;
};

namespace internal {

[[nodiscard]] auto IsValidIdentifier(std::string_view name) noexcept -> bool;

void VerifyTensorType(const TensorType &type);
void VerifyParameterType(const Parameter &parameter);
void VerifySameDType(const TensorType &lhs, const TensorType &rhs, std::string_view operation);

[[nodiscard]] auto InferBroadcastType(const Value &lhs, const Value &rhs, std::string_view operation) -> TensorType;

}  // namespace internal

}  // namespace zephyr::ir
