#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "ir/value.h"

namespace zephyr::ir {

/** Elementwise addition with shape broadcasting. */
class Add final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.add";

  Add(const Value *lhs, const Value *rhs);

  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;
};

/** Elementwise multiplication with shape broadcasting. */
class Multiply final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.multiply";

  Multiply(const Value *lhs, const Value *rhs);

  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;
};

/** Applies the SiLU activation elementwise. */
class Silu final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.silu";

  explicit Silu(const Value *input);

  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;
};

/** Applies the sigmoid activation elementwise. */
class Sigmoid final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.sigmoid";

  explicit Sigmoid(const Value *input);

  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;
};

/** Changes tensor shape without changing element order or count. */
class Reshape final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.reshape";

  Reshape(const Value *input, Shape shape);

  [[nodiscard]] auto GetShape() const -> const Shape & { return shape_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;

 private:
  Shape shape_;
};

/** Splits one static dimension into multiple result tensors. */
class Split final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.split";

  Split(const Value *input, int64_t dimension, std::vector<int64_t> sizes);

  [[nodiscard]] auto GetDimension() const -> int64_t { return dimension_; }
  [[nodiscard]] auto GetSizes() const -> std::span<const int64_t> { return sizes_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;
  void Verify() const override;

 private:
  int64_t dimension_;
  std::vector<int64_t> sizes_;
};

}  // namespace zephyr::ir
