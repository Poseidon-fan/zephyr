#include "ttl/tensor/scalar.hpp"

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "ttl/common/error.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl {
namespace {

[[noreturn]] void ThrowScalarOverflow(std::string_view dtype, std::source_location location) {
  std::string message{"scalar value cannot be represented as "};
  message.append(dtype);
  throw OverflowError(std::move(message), location);
}

[[noreturn]] void ThrowNonIntegralScalar(std::string_view dtype, std::source_location location) {
  std::string message{"floating scalar converted to "};
  message.append(dtype);
  message.append(" must be finite and integral");
  throw InvalidArgumentError(std::move(message), location);
}

template <typename Integer>
[[nodiscard]] auto CastInteger(int64_t value, std::string_view dtype, std::source_location location) -> Integer {
  if (!std::in_range<Integer>(value)) {
    ThrowScalarOverflow(dtype, location);
  }
  return static_cast<Integer>(value);
}

template <typename Integer>
[[nodiscard]] auto CastFloating(double value, std::string_view dtype, std::source_location location) -> Integer {
  if (!std::isfinite(value) || std::trunc(value) != value) {
    ThrowNonIntegralScalar(dtype, location);
  }

  constexpr auto minimum = static_cast<double>(std::numeric_limits<Integer>::lowest());
  if constexpr (std::same_as<Integer, int64_t>) {
    constexpr auto exclusive_maximum = 0x1p63;
    if (value < minimum || value >= exclusive_maximum) {
      ThrowScalarOverflow(dtype, location);
    }
  } else {
    constexpr auto maximum = static_cast<double>(std::numeric_limits<Integer>::max());
    if (value < minimum || value > maximum) {
      ThrowScalarOverflow(dtype, location);
    }
  }
  return static_cast<Integer>(value);
}

template <typename Integer>
[[nodiscard]] auto CastToInteger(const std::variant<bool, int64_t, double> &value, std::string_view dtype,
                                 std::source_location location) -> Integer {
  return std::visit(
      [dtype, location](const auto &stored) -> Integer {
        using Stored = std::remove_cvref_t<decltype(stored)>;
        if constexpr (std::same_as<Stored, bool>) {
          return stored ? Integer{1} : Integer{0};
        } else if constexpr (std::same_as<Stored, int64_t>) {
          return CastInteger<Integer>(stored, dtype, location);
        } else {
          return CastFloating<Integer>(stored, dtype, location);
        }
      },
      value);
}

[[nodiscard]] auto CastToFloat(const std::variant<bool, int64_t, double> &value, double maximum, std::string_view dtype,
                               std::source_location location) -> float {
  const auto converted = std::visit([](const auto &stored) { return static_cast<double>(stored); }, value);
  if (std::isfinite(converted) && std::abs(converted) > maximum) {
    ThrowScalarOverflow(dtype, location);
  }
  return static_cast<float>(converted);
}

}  // namespace

Scalar::Scalar(bool value) noexcept : value_(value) {}

Scalar::Scalar(int64_t value) noexcept : value_(value) {}

Scalar::Scalar(double value) noexcept : value_(value) {}

auto Scalar::IsBoolean() const noexcept -> bool { return std::holds_alternative<bool>(value_); }

auto Scalar::IsIntegral() const noexcept -> bool { return std::holds_alternative<int64_t>(value_); }

auto Scalar::IsFloating() const noexcept -> bool { return std::holds_alternative<double>(value_); }

auto Scalar::ToDouble() const noexcept -> double {
  return std::visit([](const auto &stored) { return static_cast<double>(stored); }, value_);
}

auto Scalar::CastToBool() const noexcept -> bool {
  return std::visit([](const auto &stored) { return stored != 0; }, value_);
}

auto Scalar::CastToUInt8(std::source_location location) const -> uint8_t {
  return CastToInteger<uint8_t>(value_, "uint8", location);
}

auto Scalar::CastToInt32(std::source_location location) const -> int32_t {
  return CastToInteger<int32_t>(value_, "int32", location);
}

auto Scalar::CastToInt64(std::source_location location) const -> int64_t {
  return CastToInteger<int64_t>(value_, "int64", location);
}

auto Scalar::CastToFloat16(std::source_location location) const -> Float16 {
  return FloatToFloat16(CastToFloat(value_, 65504.0, "float16", location));
}

auto Scalar::CastToBFloat16(std::source_location location) const -> BFloat16 {
  return FloatToBFloat16(CastToFloat(value_, 0x1.fep127, "bfloat16", location));
}

auto Scalar::CastToFloat32(std::source_location location) const -> float {
  return CastToFloat(value_, std::numeric_limits<float>::max(), "float32", location);
}

}  // namespace ttl
