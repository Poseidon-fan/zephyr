#pragma once

#include <cstddef>
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

  Add(Value lhs, Value rhs);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
};

/** Elementwise multiplication with shape broadcasting. */
class Multiply final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.multiply";

  Multiply(Value lhs, Value rhs);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
};

/** Applies the SiLU activation elementwise. */
class Silu final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.silu";

  explicit Silu(Value input);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
};

/** Applies the sigmoid activation elementwise. */
class Sigmoid final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.sigmoid";

  explicit Sigmoid(Value input);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
};

/** Changes tensor shape without changing element order or count. */
class Reshape final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.reshape";

  Reshape(Value input, Shape shape);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetShape() const -> const Shape & { return shape_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }

 private:
  Shape shape_;
};

/** Splits one static dimension into multiple result tensors. */
class Split final : public Operation {
 public:
  static constexpr std::string_view NAME = "tensor.split";

  Split(Value input, size_t dimension, std::vector<int64_t> sizes);

  void Accept(OperationVisitor &visitor) const override { visitor.Visit(*this); }

  [[nodiscard]] auto GetDimension() const -> size_t { return dimension_; }
  [[nodiscard]] auto GetSizes() const -> std::span<const int64_t> { return sizes_; }
  [[nodiscard]] auto GetName() const -> std::string_view override { return NAME; }
  [[nodiscard]] auto ToString(const OperationIndices &operation_indices) const -> std::string override;

 private:
  size_t dimension_;
  std::vector<int64_t> sizes_;
};

}  // namespace zephyr::ir
