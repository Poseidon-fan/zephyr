#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <source_location>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace ttl {

inline constexpr size_t TTL_MAX_RANK = 8;

/** Immutable tensor dimensions stored inline. A default-constructed Shape represents a scalar. */
class Shape final {
 public:
  constexpr Shape() noexcept = default;
  explicit Shape(std::span<const int64_t> dimensions, std::source_location location = std::source_location::current());
  Shape(std::initializer_list<int64_t> dimensions, std::source_location location = std::source_location::current());

  [[nodiscard]] constexpr auto GetRank() const noexcept -> size_t { return rank_; }
  [[nodiscard]] auto GetDimension(size_t axis, std::source_location location = std::source_location::current()) const
      -> int64_t;
  [[nodiscard]] constexpr auto GetDimensions() const noexcept -> std::span<const int64_t> {
    return {dimensions_.data(), rank_};
  }
  [[nodiscard]] constexpr auto GetNumElements() const noexcept -> int64_t { return num_elements_; }
  [[nodiscard]] constexpr auto IsScalar() const noexcept -> bool { return rank_ == 0; }
  [[nodiscard]] constexpr auto IsEmpty() const noexcept -> bool { return num_elements_ == 0; }
  [[nodiscard]] auto ToString() const -> std::string;

  [[nodiscard]] constexpr auto operator==(const Shape &) const noexcept -> bool = default;

 private:
  std::array<int64_t, TTL_MAX_RANK> dimensions_{};
  int64_t num_elements_{1};
  uint8_t rank_{0};
};

auto operator<<(std::ostream &stream, const Shape &shape) -> std::ostream &;

/** Immutable element strides stored inline. A default-constructed Strides represents rank-zero strides. */
class Strides final {
 public:
  constexpr Strides() noexcept = default;
  explicit Strides(std::span<const int64_t> strides, std::source_location location = std::source_location::current());
  Strides(std::initializer_list<int64_t> strides, std::source_location location = std::source_location::current());

  [[nodiscard]] constexpr auto GetRank() const noexcept -> size_t { return rank_; }
  [[nodiscard]] auto GetStride(size_t axis, std::source_location location = std::source_location::current()) const
      -> int64_t;
  [[nodiscard]] constexpr auto GetValues() const noexcept -> std::span<const int64_t> {
    return {strides_.data(), rank_};
  }
  [[nodiscard]] auto ToString() const -> std::string;

  [[nodiscard]] constexpr auto operator==(const Strides &) const noexcept -> bool = default;

 private:
  std::array<int64_t, TTL_MAX_RANK> strides_{};
  uint8_t rank_{0};
};

auto operator<<(std::ostream &stream, const Strides &strides) -> std::ostream &;

/** Normalize an axis from [-rank, rank) to [0, rank). */
[[nodiscard]] auto NormalizeAxis(int64_t axis, size_t rank,
                                 std::source_location location = std::source_location::current()) -> size_t;

/** Normalize, sort, and reject duplicate axes. */
[[nodiscard]] auto NormalizeAxes(std::span<const int64_t> axes, size_t rank,
                                 std::source_location location = std::source_location::current())
    -> std::vector<size_t>;

/** Compute the canonical row-major element strides for a shape. */
[[nodiscard]] auto GetContiguousStrides(const Shape &shape,
                                        std::source_location location = std::source_location::current()) -> Strides;

/** Return whether strides describe a row-major contiguous layout for shape. */
[[nodiscard]] auto IsContiguous(const Shape &shape, const Strides &strides,
                                std::source_location location = std::source_location::current()) -> bool;

static_assert(std::is_trivially_copyable_v<Shape>);
static_assert(std::is_standard_layout_v<Shape>);
static_assert(std::is_trivially_copyable_v<Strides>);
static_assert(std::is_standard_layout_v<Strides>);

}  // namespace ttl
