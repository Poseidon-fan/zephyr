#pragma once

#include <cstdint>
#include <memory>

namespace zephyr::planner {

/** Identifies one communicator specification referenced by a collective instruction. */
using communicator_id_t = uint32_t;

/** One strongly typed action in a plan's execution order. */
class Instruction {
 public:
  Instruction() = default;
  Instruction(const Instruction &) = default;
  auto operator=(const Instruction &) -> Instruction & = default;
  Instruction(Instruction &&) noexcept = default;
  auto operator=(Instruction &&) noexcept -> Instruction & = default;
  virtual ~Instruction() = default;

  /** Makes an owning copy while preserving the concrete instruction type. */
  [[nodiscard]] virtual auto Clone() const -> std::unique_ptr<Instruction> = 0;
};

/** Supplies the common deep-copy implementation for concrete instructions. */
template <typename Derived>
class CloneableInstruction : public Instruction {
 public:
  [[nodiscard]] auto Clone() const -> std::unique_ptr<Instruction> override {
    return std::make_unique<Derived>(static_cast<const Derived &>(*this));
  }

 private:
  friend Derived;

  CloneableInstruction() = default;
};

}  // namespace zephyr::planner
