#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <source_location>
#include <string>
#include <type_traits>

namespace ttl {

/**
 * Identifies one concrete CUDA device by ordinal.
 *
 * Device construction does not initialize the CUDA runtime or verify that the ordinal exists. Runtime performs all
 * environment-dependent validation.
 */
class Device final {
 public:
  constexpr explicit Device(int32_t ordinal, std::source_location location = std::source_location::current())
      : ordinal_(ordinal) {
    if (ordinal < 0) {
      ThrowInvalidOrdinal(ordinal, location);
    }
  }

  [[nodiscard]] constexpr auto GetOrdinal() const noexcept -> int32_t { return ordinal_; }
  [[nodiscard]] auto ToString() const -> std::string;

  [[nodiscard]] constexpr auto operator==(const Device &) const noexcept -> bool = default;

 private:
  [[noreturn]] static void ThrowInvalidOrdinal(int32_t ordinal, std::source_location location);

  int32_t ordinal_;
};

auto operator<<(std::ostream &stream, const Device &device) -> std::ostream &;

static_assert(!std::default_initializable<Device>);
static_assert(sizeof(Device) == sizeof(int32_t));
static_assert(alignof(Device) == alignof(int32_t));
static_assert(std::is_trivially_copyable_v<Device>);
static_assert(std::is_standard_layout_v<Device>);

}  // namespace ttl

template <>
struct std::hash<ttl::Device> {
  [[nodiscard]] auto operator()(ttl::Device device) const noexcept -> size_t {
    return std::hash<int32_t>{}(device.GetOrdinal());
  }
};
