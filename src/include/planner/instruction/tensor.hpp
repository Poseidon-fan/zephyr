#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "planner/buffer.hpp"
#include "planner/plan.hpp"

namespace zephyr::planner {

/** Adds two broadcast-compatible tensors. */
class Add final : public CloneableInstruction<Add> {
 public:
  Add(BufferView lhs, BufferView rhs, BufferView output)
      : lhs_(std::move(lhs)), rhs_(std::move(rhs)), output_(std::move(output)) {}

  BufferView lhs_;
  BufferView rhs_;
  BufferView output_;
};

/** Multiplies two broadcast-compatible tensors. */
class Multiply final : public CloneableInstruction<Multiply> {
 public:
  Multiply(BufferView lhs, BufferView rhs, BufferView output)
      : lhs_(std::move(lhs)), rhs_(std::move(rhs)), output_(std::move(output)) {}

  BufferView lhs_;
  BufferView rhs_;
  BufferView output_;
};

/** Applies the SiLU activation elementwise. */
class Silu final : public CloneableInstruction<Silu> {
 public:
  Silu(BufferView input, BufferView output) : input_(std::move(input)), output_(std::move(output)) {}

  BufferView input_;
  BufferView output_;
};

/** Applies the sigmoid activation elementwise. */
class Sigmoid final : public CloneableInstruction<Sigmoid> {
 public:
  Sigmoid(BufferView input, BufferView output) : input_(std::move(input)), output_(std::move(output)) {}

  BufferView input_;
  BufferView output_;
};

/** Reshapes a tensor without changing its element order. */
class Reshape final : public CloneableInstruction<Reshape> {
 public:
  Reshape(BufferView input, BufferView output) : input_(std::move(input)), output_(std::move(output)) {}

  BufferView input_;
  BufferView output_;
};

/** Materializes one contiguous slice for each result of a split. */
class Split final : public CloneableInstruction<Split> {
 public:
  Split(BufferView input, size_t dimension, std::vector<BufferView> outputs)
      : input_(std::move(input)), dimension_(dimension), outputs_(std::move(outputs)) {}

  BufferView input_;
  size_t dimension_;
  std::vector<BufferView> outputs_;
};

}  // namespace zephyr::planner
