#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <source_location>
#include <span>
#include <string_view>
#include <type_traits>

#include "ttl/internal/common/index_width.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::internal {

inline constexpr size_t TTL_MAX_ITERATOR_OPERANDS = 4;
inline constexpr size_t TTL_MAX_ITERATOR_INPUTS = TTL_MAX_ITERATOR_OPERANDS - 1;

enum class AliasPolicy : uint8_t {
  NO_ALIAS,
  EXACT_UNARY,
  EXACT_ONE_BINARY_INPUT,
  EXACT_ONE_WHERE_VALUE,
  COPY,
};

enum class IteratorPath : uint8_t {
  CONTIGUOUS,
  CONTIGUOUS_WITH_SCALAR_INPUTS,
  SINGLE_INNER_STRIDE,
  GENERIC_STRIDED,
};

struct ElementwiseParameters32 final {
  std::byte *pointers_[TTL_MAX_ITERATOR_OPERANDS]{};
  uint32_t strides_bytes_[TTL_MAX_ITERATOR_OPERANDS][TTL_MAX_RANK]{};
  uint32_t shape_[TTL_MAX_RANK]{};
  uint32_t num_elements_{0};
  uint8_t rank_{0};
  uint8_t operand_count_{0};
};

struct ElementwiseParameters64 final {
  std::byte *pointers_[TTL_MAX_ITERATOR_OPERANDS]{};
  uint64_t strides_bytes_[TTL_MAX_ITERATOR_OPERANDS][TTL_MAX_RANK]{};
  uint64_t shape_[TTL_MAX_RANK]{};
  uint64_t num_elements_{0};
  uint8_t rank_{0};
  uint8_t operand_count_{0};
};

static_assert(std::is_trivially_copyable_v<ElementwiseParameters32>);
static_assert(std::is_standard_layout_v<ElementwiseParameters32>);
static_assert(std::is_trivially_copyable_v<ElementwiseParameters64>);
static_assert(std::is_standard_layout_v<ElementwiseParameters64>);

/** Validate the common writable-output contract used by elementwise and copy operators. */
void ValidateWritableOutput(const Tensor &output, std::string_view operation,
                            std::source_location location = std::source_location::current());

/** Validate a conservative output/input alias policy. */
void ValidateAlias(AliasPolicy policy, const Tensor &output, std::span<const Tensor *const> inputs,
                   std::string_view operation, std::source_location location = std::source_location::current());

/**
 * @brief Immutable host-side iteration plan for one output and up to three inputs.
 *
 * The iterator validates and lowers layout metadata. It neither allocates nor launches work and its device parameter
 * records contain no host metadata pointers or ownership objects.
 */
class ElementwiseIterator final {
 public:
  class Builder final {
   public:
    auto AddOutput(Tensor &tensor) -> Builder &;
    auto AddInput(const Tensor &tensor) -> Builder &;
    auto SetAliasPolicy(AliasPolicy policy) noexcept -> Builder &;
    auto SetRequireSameDType(bool value) noexcept -> Builder &;

    [[nodiscard]] auto Build(std::string_view operation,
                             std::source_location location = std::source_location::current()) -> ElementwiseIterator;

   private:
    std::optional<Tensor> output_;
    std::array<std::optional<Tensor>, TTL_MAX_ITERATOR_INPUTS> inputs_;
    size_t input_count_{0};
    AliasPolicy alias_policy_{AliasPolicy::NO_ALIAS};
    bool require_same_dtype_{false};
  };

  [[nodiscard]] auto GetShape() const noexcept -> const Shape &;
  [[nodiscard]] auto GetNumElements() const noexcept -> int64_t;
  [[nodiscard]] auto GetRank() const noexcept -> size_t;
  [[nodiscard]] auto GetOperandCount() const noexcept -> size_t;
  [[nodiscard]] auto GetIndexWidth() const noexcept -> IndexWidth;
  [[nodiscard]] auto GetVectorWidthElements() const noexcept -> uint8_t;
  [[nodiscard]] auto GetPath() const noexcept -> IteratorPath;

  /** Return the compact 32-bit record selected by GetIndexWidth(). */
  [[nodiscard]] auto MakeParameters32(std::source_location location = std::source_location::current()) const
      -> ElementwiseParameters32;

  /** Return the full-width record. This is valid for either index-width classification. */
  [[nodiscard]] auto MakeParameters64() const noexcept -> ElementwiseParameters64;

 private:
  friend class Builder;

  ElementwiseIterator() = default;

  Shape shape_;
  std::array<std::optional<Tensor>, TTL_MAX_ITERATOR_OPERANDS> operands_;
  std::array<std::byte *, TTL_MAX_ITERATOR_OPERANDS> pointers_{};
  std::array<std::array<uint64_t, TTL_MAX_RANK>, TTL_MAX_ITERATOR_OPERANDS> strides_bytes_{};
  std::array<uint64_t, TTL_MAX_RANK> iteration_shape_{};
  int64_t num_elements_{0};
  uint8_t rank_{0};
  uint8_t operand_count_{0};
  uint8_t vector_width_elements_{1};
  IndexWidth index_width_{IndexWidth::UINT64};
  IteratorPath path_{IteratorPath::GENERIC_STRIDED};
};

}  // namespace ttl::internal
