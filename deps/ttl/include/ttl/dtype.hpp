#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <string_view>
#include <type_traits>

#include "ttl/error.hpp"

namespace ttl {

/** Element types supported by TTL tensors. */
enum class DType : uint8_t {
  BOOL,
  UINT8,
  INT32,
  INT64,
  FLOAT16,
  BFLOAT16,
  FLOAT32,
};

/** Broad semantic category used by schema validation and dtype dispatch. */
enum class DTypeCategory : uint8_t {
  BOOLEAN,
  UNSIGNED_INTEGER,
  SIGNED_INTEGER,
  FLOATING,
};

/** IEEE 754 binary16 storage. Arithmetic is deliberately not provided by this host-visible wrapper. */
struct alignas(2) Float16 final {
  uint16_t bits_{0};

  [[nodiscard]] constexpr auto operator==(const Float16 &) const noexcept -> bool = default;
};

/** Brain floating-point binary16 storage. Arithmetic is deliberately not provided by this host-visible wrapper. */
struct alignas(2) BFloat16 final {
  uint16_t bits_{0};

  [[nodiscard]] constexpr auto operator==(const BFloat16 &) const noexcept -> bool = default;
};

static_assert(sizeof(bool) == 1);
static_assert(alignof(bool) == 1);
static_assert(std::is_trivially_copyable_v<bool>);

static_assert(sizeof(Float16) == 2);
static_assert(alignof(Float16) == 2);
static_assert(std::is_trivially_copyable_v<Float16>);
static_assert(std::is_standard_layout_v<Float16>);

static_assert(sizeof(BFloat16) == 2);
static_assert(alignof(BFloat16) == 2);
static_assert(std::is_trivially_copyable_v<BFloat16>);
static_assert(std::is_standard_layout_v<BFloat16>);

/** Metadata for one supported DType. */
struct DTypeInfo final {
  DType dtype_;
  std::string_view name_;
  uint8_t size_bytes_;
  uint8_t alignment_bytes_;
  DTypeCategory category_;

  [[nodiscard]] constexpr auto operator==(const DTypeInfo &) const noexcept -> bool = default;
};

/** Return whether a raw DType enum value names a supported TTL dtype. */
[[nodiscard]] constexpr auto IsValidDType(DType dtype) noexcept -> bool {
  switch (dtype) {
    case DType::BOOL:
    case DType::UINT8:
    case DType::INT32:
    case DType::INT64:
    case DType::FLOAT16:
    case DType::BFLOAT16:
    case DType::FLOAT32:
      return true;
  }
  return false;
}

/** Return complete metadata for a dtype, or reject an invalid enum value. */
[[nodiscard]] constexpr auto GetDTypeInfo(DType dtype, std::source_location location = std::source_location::current())
    -> DTypeInfo {
  switch (dtype) {
    case DType::BOOL:
      return {
          .dtype_ = DType::BOOL,
          .name_ = "bool",
          .size_bytes_ = sizeof(bool),
          .alignment_bytes_ = alignof(bool),
          .category_ = DTypeCategory::BOOLEAN,
      };
    case DType::UINT8:
      return {
          .dtype_ = DType::UINT8,
          .name_ = "uint8",
          .size_bytes_ = sizeof(uint8_t),
          .alignment_bytes_ = alignof(uint8_t),
          .category_ = DTypeCategory::UNSIGNED_INTEGER,
      };
    case DType::INT32:
      return {
          .dtype_ = DType::INT32,
          .name_ = "int32",
          .size_bytes_ = sizeof(int32_t),
          .alignment_bytes_ = alignof(int32_t),
          .category_ = DTypeCategory::SIGNED_INTEGER,
      };
    case DType::INT64:
      return {
          .dtype_ = DType::INT64,
          .name_ = "int64",
          .size_bytes_ = sizeof(int64_t),
          .alignment_bytes_ = alignof(int64_t),
          .category_ = DTypeCategory::SIGNED_INTEGER,
      };
    case DType::FLOAT16:
      return {
          .dtype_ = DType::FLOAT16,
          .name_ = "float16",
          .size_bytes_ = sizeof(Float16),
          .alignment_bytes_ = alignof(Float16),
          .category_ = DTypeCategory::FLOATING,
      };
    case DType::BFLOAT16:
      return {
          .dtype_ = DType::BFLOAT16,
          .name_ = "bfloat16",
          .size_bytes_ = sizeof(BFloat16),
          .alignment_bytes_ = alignof(BFloat16),
          .category_ = DTypeCategory::FLOATING,
      };
    case DType::FLOAT32:
      return {
          .dtype_ = DType::FLOAT32,
          .name_ = "float32",
          .size_bytes_ = sizeof(float),
          .alignment_bytes_ = alignof(float),
          .category_ = DTypeCategory::FLOATING,
      };
  }
  throw InvalidArgumentError("invalid dtype", location);
}

[[nodiscard]] constexpr auto GetDTypeName(DType dtype, std::source_location location = std::source_location::current())
    -> std::string_view {
  return GetDTypeInfo(dtype, location).name_;
}

[[nodiscard]] constexpr auto GetDTypeSize(DType dtype, std::source_location location = std::source_location::current())
    -> size_t {
  return GetDTypeInfo(dtype, location).size_bytes_;
}

[[nodiscard]] constexpr auto GetDTypeAlignment(DType dtype,
                                               std::source_location location = std::source_location::current())
    -> size_t {
  return GetDTypeInfo(dtype, location).alignment_bytes_;
}

[[nodiscard]] constexpr auto IsBoolean(DType dtype, std::source_location location = std::source_location::current())
    -> bool {
  return GetDTypeInfo(dtype, location).category_ == DTypeCategory::BOOLEAN;
}

[[nodiscard]] constexpr auto IsUnsignedInteger(DType dtype,
                                               std::source_location location = std::source_location::current())
    -> bool {
  return GetDTypeInfo(dtype, location).category_ == DTypeCategory::UNSIGNED_INTEGER;
}

[[nodiscard]] constexpr auto IsSignedInteger(DType dtype,
                                             std::source_location location = std::source_location::current()) -> bool {
  return GetDTypeInfo(dtype, location).category_ == DTypeCategory::SIGNED_INTEGER;
}

/** Return true for UINT8, INT32, and INT64. BOOL is deliberately excluded. */
[[nodiscard]] constexpr auto IsIntegral(DType dtype, std::source_location location = std::source_location::current())
    -> bool {
  const auto category = GetDTypeInfo(dtype, location).category_;
  return category == DTypeCategory::UNSIGNED_INTEGER || category == DTypeCategory::SIGNED_INTEGER;
}

[[nodiscard]] constexpr auto IsFloating(DType dtype, std::source_location location = std::source_location::current())
    -> bool {
  return GetDTypeInfo(dtype, location).category_ == DTypeCategory::FLOATING;
}

template <typename T>
struct DTypeOf;

template <>
struct DTypeOf<bool> {
  static constexpr DType VALUE = DType::BOOL;
};

template <>
struct DTypeOf<uint8_t> {
  static constexpr DType VALUE = DType::UINT8;
};

template <>
struct DTypeOf<int32_t> {
  static constexpr DType VALUE = DType::INT32;
};

template <>
struct DTypeOf<int64_t> {
  static constexpr DType VALUE = DType::INT64;
};

template <>
struct DTypeOf<Float16> {
  static constexpr DType VALUE = DType::FLOAT16;
};

template <>
struct DTypeOf<BFloat16> {
  static constexpr DType VALUE = DType::BFLOAT16;
};

template <>
struct DTypeOf<float> {
  static constexpr DType VALUE = DType::FLOAT32;
};

template <typename T>
concept TensorStorageType = !std::is_volatile_v<T> && requires {
  { DTypeOf<std::remove_cv_t<T>>::VALUE } -> std::convertible_to<DType>;
};

template <TensorStorageType T>
// CUDA-mode clang-tidy misclassifies this dependent constexpr variable template as dynamically initialized.
// NOLINTNEXTLINE(bugprone-dynamic-static-initializers)
inline constexpr DType DTYPE_OF = DTypeOf<std::remove_cv_t<T>>::VALUE;

template <DType dtype>
struct StorageTypeFor;

template <>
struct StorageTypeFor<DType::BOOL> {
  using Type = bool;
};

template <>
struct StorageTypeFor<DType::UINT8> {
  using Type = uint8_t;
};

template <>
struct StorageTypeFor<DType::INT32> {
  using Type = int32_t;
};

template <>
struct StorageTypeFor<DType::INT64> {
  using Type = int64_t;
};

template <>
struct StorageTypeFor<DType::FLOAT16> {
  using Type = Float16;
};

template <>
struct StorageTypeFor<DType::BFLOAT16> {
  using Type = BFloat16;
};

template <>
struct StorageTypeFor<DType::FLOAT32> {
  using Type = float;
};

template <DType dtype>
using StorageTypeForT = typename StorageTypeFor<dtype>::Type;

/** Convert using CUDA's round-to-nearest-even binary16 semantics. */
[[nodiscard]] auto FloatToFloat16(float value) noexcept -> Float16;

/** Convert an IEEE 754 binary16 bit pattern to float. */
[[nodiscard]] auto Float16ToFloat(Float16 value) noexcept -> float;

/** Convert using CUDA's round-to-nearest-even bfloat16 semantics. */
[[nodiscard]] auto FloatToBFloat16(float value) noexcept -> BFloat16;

/** Convert a bfloat16 bit pattern to float. */
[[nodiscard]] auto BFloat16ToFloat(BFloat16 value) noexcept -> float;

}  // namespace ttl
