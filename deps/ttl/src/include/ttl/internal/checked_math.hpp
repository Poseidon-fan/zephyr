#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "ttl/error.hpp"

namespace ttl::internal {

template <typename T>
concept CheckedInteger = std::integral<T> && std::same_as<T, std::remove_cv_t<T>> && !std::same_as<T, bool> &&
                         !std::same_as<T, char> && !std::same_as<T, wchar_t> && !std::same_as<T, char8_t> &&
                         !std::same_as<T, char16_t> && !std::same_as<T, char32_t> && (sizeof(T) <= sizeof(uint64_t));

template <CheckedInteger T>
[[nodiscard]] auto CheckedIntegerToString(T value) -> std::string {
  if constexpr (std::signed_integral<T>) {
    return std::to_string(static_cast<int64_t>(value));
  }
  return std::to_string(static_cast<uint64_t>(value));
}

template <CheckedInteger T>
[[noreturn]] void ThrowCheckedBinaryOverflow(std::string_view description, T lhs, std::string_view operation, T rhs,
                                             std::source_location location) {
  std::string message;
  message.reserve(description.size() + operation.size() + 64);
  message.append(description);
  message.append(" overflow: ");
  message.append(CheckedIntegerToString(lhs));
  message.push_back(' ');
  message.append(operation);
  message.push_back(' ');
  message.append(CheckedIntegerToString(rhs));
  throw OverflowError(std::move(message), location);
}

template <CheckedInteger From>
[[noreturn]] void ThrowCheckedNarrowingError(std::string_view description, From value, std::source_location location) {
  std::string message;
  message.reserve(description.size() + 64);
  message.append(description);
  message.append(" out of range: ");
  message.append(CheckedIntegerToString(value));
  throw OverflowError(std::move(message), location);
}

[[noreturn]] inline void ThrowInvalidCheckedMathArgument(std::string_view description, std::string_view reason,
                                                         std::source_location location) {
  std::string message;
  message.reserve(description.size() + reason.size() + 2);
  message.append(description);
  message.append(": ");
  message.append(reason);
  throw InvalidArgumentError(std::move(message), location);
}

template <CheckedInteger T>
[[nodiscard]] auto CheckedAdd(T lhs, T rhs, std::string_view description,
                              std::source_location location = std::source_location::current()) -> T {
  // Arithmetic is checked against T's range; these helpers never widen the result type.
  T result;
  if (__builtin_add_overflow(lhs, rhs, &result)) {
    ThrowCheckedBinaryOverflow(description, lhs, "+", rhs, location);
  }
  return result;
}

template <CheckedInteger T>
[[nodiscard]] auto CheckedSubtract(T lhs, T rhs, std::string_view description,
                                   std::source_location location = std::source_location::current()) -> T {
  T result;
  if (__builtin_sub_overflow(lhs, rhs, &result)) {
    ThrowCheckedBinaryOverflow(description, lhs, "-", rhs, location);
  }
  return result;
}

template <CheckedInteger T>
[[nodiscard]] auto CheckedMultiply(T lhs, T rhs, std::string_view description,
                                   std::source_location location = std::source_location::current()) -> T {
  T result;
  if (__builtin_mul_overflow(lhs, rhs, &result)) {
    ThrowCheckedBinaryOverflow(description, lhs, "*", rhs, location);
  }
  return result;
}

template <CheckedInteger To, CheckedInteger From>
[[nodiscard]] auto CheckedNarrow(From value, std::string_view description,
                                 std::source_location location = std::source_location::current()) -> To {
  if (!std::in_range<To>(value)) {
    ThrowCheckedNarrowingError(description, value, location);
  }
  return static_cast<To>(value);
}

template <CheckedInteger T>
[[nodiscard]] auto CheckedBytes(T num_elements, size_t element_size,
                                std::source_location location = std::source_location::current()) -> size_t {
  if constexpr (std::signed_integral<T>) {
    if (num_elements < 0) {
      ThrowInvalidCheckedMathArgument("byte size", "element count must be non-negative", location);
    }
  }
  const auto count = CheckedNarrow<size_t>(num_elements, "element count", location);
  return CheckedMultiply(count, element_size, "byte size", location);
}

template <CheckedInteger T>
[[nodiscard]] auto CheckedElementOffsetToBytes(T offset, size_t element_size,
                                               std::source_location location = std::source_location::current())
    -> size_t {
  if constexpr (std::signed_integral<T>) {
    if (offset < 0) {
      ThrowInvalidCheckedMathArgument("element offset", "must be non-negative", location);
    }
  }
  const auto converted_offset = CheckedNarrow<size_t>(offset, "element offset", location);
  return CheckedMultiply(converted_offset, element_size, "element offset in bytes", location);
}

[[nodiscard]] inline auto AlignUp(size_t value, size_t alignment, std::string_view description = "alignment",
                                  std::source_location location = std::source_location::current()) -> size_t {
  if (!std::has_single_bit(alignment)) {
    ThrowInvalidCheckedMathArgument(description, "alignment must be a non-zero power of two", location);
  }
  const auto mask = alignment - 1;
  return CheckedAdd(value, mask, description, location) & ~mask;
}

template <CheckedInteger T>
[[nodiscard]] auto CeilDivide(T dividend, T divisor, std::string_view description = "ceil divide",
                              std::source_location location = std::source_location::current()) -> T {
  if (divisor <= 0) {
    ThrowInvalidCheckedMathArgument(description, "divisor must be positive", location);
  }
  if constexpr (std::signed_integral<T>) {
    if (dividend < 0) {
      ThrowInvalidCheckedMathArgument(description, "dividend must be non-negative", location);
    }
  }
  return static_cast<T>((dividend / divisor) + static_cast<T>(dividend % divisor != 0));
}

}  // namespace ttl::internal
