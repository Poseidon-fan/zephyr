#pragma once

namespace zephyr::ir {

class Add;
class Embedding;
class Linear;
class Moe;
class Multiply;
class Reshape;
class RmsNorm;
class RotaryEmbedding;
class SelfAttention;
class Sigmoid;
class Silu;
class Split;

/** Receives one callback for each concrete IR operation type. */
class OperationVisitor {
 public:
  virtual ~OperationVisitor() = default;

  virtual void Visit(const Linear &) = 0;
  virtual void Visit(const Embedding &) = 0;
  virtual void Visit(const RmsNorm &) = 0;
  virtual void Visit(const RotaryEmbedding &) = 0;
  virtual void Visit(const SelfAttention &) = 0;
  virtual void Visit(const Moe &) = 0;
  virtual void Visit(const Add &) = 0;
  virtual void Visit(const Multiply &) = 0;
  virtual void Visit(const Silu &) = 0;
  virtual void Visit(const Sigmoid &) = 0;
  virtual void Visit(const Reshape &) = 0;
  virtual void Visit(const Split &) = 0;
};

}  // namespace zephyr::ir
