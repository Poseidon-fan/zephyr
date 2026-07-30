#pragma once

#include <concepts>
#include <cstdint>
#include <source_location>
#include <type_traits>
#include <variant>

#include "ttl/dtype.hpp"

namespace ttl {

/**
 * Host scalar accepted by tensor creation and scalar operators.
 *
 * Scalar preserves whether its value originated as boolean, signed integer, or floating point. Conversion to a tensor
 * storage type is explicit and checked.
 */
class Scalar final {
 public:
  explicit Scalar(bool value) noexcept;
  explicit Scalar(int64_t value) noexcept;
  explicit Scalar(double value) noexcept;

  [[nodiscard]] auto IsBoolean() const noexcept -> bool;
  [[nodiscard]] auto IsIntegral() const noexcept -> bool;
  [[nodiscard]] auto IsFloating() const noexcept -> bool;

  template <TensorStorageType T>
  [[nodiscard]] auto Cast(std::source_location location = std::source_location::current()) const
      -> std::remove_cv_t<T> {
    using Value = std::remove_cv_t<T>;
    if constexpr (std::same_as<Value, bool>) {
      return CastToBool();
    } else if constexpr (std::same_as<Value, uint8_t>) {
      return CastToUInt8(location);
    } else if constexpr (std::same_as<Value, int32_t>) {
      return CastToInt32(location);
    } else if constexpr (std::same_as<Value, int64_t>) {
      return CastToInt64(location);
    } else if constexpr (std::same_as<Value, Float16>) {
      return CastToFloat16();
    } else if constexpr (std::same_as<Value, BFloat16>) {
      return CastToBFloat16();
    } else {
      static_assert(std::same_as<Value, float>);
      return CastToFloat32();
    }
  }

 private:
  [[nodiscard]] auto CastToBool() const noexcept -> bool;
  [[nodiscard]] auto CastToUInt8(std::source_location location) const -> uint8_t;
  [[nodiscard]] auto CastToInt32(std::source_location location) const -> int32_t;
  [[nodiscard]] auto CastToInt64(std::source_location location) const -> int64_t;
  [[nodiscard]] auto CastToFloat16() const noexcept -> Float16;
  [[nodiscard]] auto CastToBFloat16() const noexcept -> BFloat16;
  [[nodiscard]] auto CastToFloat32() const noexcept -> float;

  std::variant<bool, int64_t, double> value_;
};

}  // namespace ttl
