#include "ttl/dtype.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ttl {

using testing::HasSubstr;

static_assert(sizeof(DType) == sizeof(uint8_t));
static_assert(std::same_as<StorageTypeForT<DType::BOOL>, bool>);
static_assert(std::same_as<StorageTypeForT<DType::UINT8>, uint8_t>);
static_assert(std::same_as<StorageTypeForT<DType::INT32>, int32_t>);
static_assert(std::same_as<StorageTypeForT<DType::INT64>, int64_t>);
static_assert(std::same_as<StorageTypeForT<DType::FLOAT16>, Float16>);
static_assert(std::same_as<StorageTypeForT<DType::BFLOAT16>, BFloat16>);
static_assert(std::same_as<StorageTypeForT<DType::FLOAT32>, float>);

static_assert(TensorStorageType<bool>);
static_assert(TensorStorageType<const Float16>);
static_assert(TensorStorageType<uint8_t>);
static_assert(TensorStorageType<int32_t>);
static_assert(TensorStorageType<int64_t>);
static_assert(TensorStorageType<float>);
static_assert(!TensorStorageType<double>);
static_assert(!TensorStorageType<volatile float>);
static_assert(!TensorStorageType<float &>);

static_assert(DTYPE_OF<bool> == DType::BOOL);
static_assert(DTYPE_OF<uint8_t> == DType::UINT8);
static_assert(DTYPE_OF<int32_t> == DType::INT32);
static_assert(DTYPE_OF<int64_t> == DType::INT64);
static_assert(DTYPE_OF<Float16> == DType::FLOAT16);
static_assert(DTYPE_OF<BFloat16> == DType::BFLOAT16);
static_assert(DTYPE_OF<const float> == DType::FLOAT32);

static_assert(GetDTypeSize(DType::FLOAT16) == 2);
static_assert(GetDTypeAlignment(DType::FLOAT32) == alignof(float));
static_assert(IsBoolean(DType::BOOL));
static_assert(IsIntegral(DType::UINT8));
static_assert(!IsIntegral(DType::BOOL));
static_assert(IsFloating(DType::BFLOAT16));

TEST(DTypeInfoTest, DescribesEverySupportedType) {
  constexpr std::array expected{
      DTypeInfo{.dtype_ = DType::BOOL,
                .name_ = "bool",
                .size_bytes_ = sizeof(bool),
                .alignment_bytes_ = alignof(bool),
                .category_ = DTypeCategory::BOOLEAN},
      DTypeInfo{.dtype_ = DType::UINT8,
                .name_ = "uint8",
                .size_bytes_ = 1,
                .alignment_bytes_ = alignof(uint8_t),
                .category_ = DTypeCategory::UNSIGNED_INTEGER},
      DTypeInfo{.dtype_ = DType::INT32,
                .name_ = "int32",
                .size_bytes_ = 4,
                .alignment_bytes_ = alignof(int32_t),
                .category_ = DTypeCategory::SIGNED_INTEGER},
      DTypeInfo{.dtype_ = DType::INT64,
                .name_ = "int64",
                .size_bytes_ = 8,
                .alignment_bytes_ = alignof(int64_t),
                .category_ = DTypeCategory::SIGNED_INTEGER},
      DTypeInfo{.dtype_ = DType::FLOAT16,
                .name_ = "float16",
                .size_bytes_ = 2,
                .alignment_bytes_ = alignof(Float16),
                .category_ = DTypeCategory::FLOATING},
      DTypeInfo{.dtype_ = DType::BFLOAT16,
                .name_ = "bfloat16",
                .size_bytes_ = 2,
                .alignment_bytes_ = alignof(BFloat16),
                .category_ = DTypeCategory::FLOATING},
      DTypeInfo{.dtype_ = DType::FLOAT32,
                .name_ = "float32",
                .size_bytes_ = 4,
                .alignment_bytes_ = alignof(float),
                .category_ = DTypeCategory::FLOATING},
  };

  for (const auto &info : expected) {
    EXPECT_TRUE(IsValidDType(info.dtype_));
    EXPECT_EQ(GetDTypeInfo(info.dtype_), info);
    EXPECT_EQ(GetDTypeName(info.dtype_), info.name_);
    EXPECT_EQ(GetDTypeSize(info.dtype_), info.size_bytes_);
    EXPECT_EQ(GetDTypeAlignment(info.dtype_), info.alignment_bytes_);
  }
}

TEST(DTypeInfoTest, ClassifiesBooleanIntegralAndFloatingSeparately) {
  EXPECT_TRUE(IsBoolean(DType::BOOL));
  EXPECT_FALSE(IsIntegral(DType::BOOL));
  EXPECT_FALSE(IsFloating(DType::BOOL));

  EXPECT_TRUE(IsUnsignedInteger(DType::UINT8));
  EXPECT_TRUE(IsIntegral(DType::UINT8));
  EXPECT_TRUE(IsSignedInteger(DType::INT32));
  EXPECT_TRUE(IsIntegral(DType::INT64));

  EXPECT_TRUE(IsFloating(DType::FLOAT16));
  EXPECT_TRUE(IsFloating(DType::BFLOAT16));
  EXPECT_TRUE(IsFloating(DType::FLOAT32));
}

TEST(DTypeInfoTest, RejectsInvalidEnumValues) {
  constexpr auto invalid = std::bit_cast<DType>(uint8_t{0xFF});
  EXPECT_FALSE(IsValidDType(invalid));
  EXPECT_THROW(static_cast<void>(GetDTypeInfo(invalid)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(IsFloating(invalid)), InvalidArgumentError);

  try {
    static_cast<void>(GetDTypeSize(invalid));
  } catch (const InvalidArgumentError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("invalid dtype"));
    EXPECT_THAT(std::string_view(error.GetLocation().file_name()), HasSubstr("dtype_test.cpp"));
    return;
  }
  FAIL() << "expected InvalidArgumentError";
}

TEST(Float16Test, ConvertsKnownValuesAndPreservesSpecialValues) {
  EXPECT_EQ(FloatToFloat16(0.0F).bits_, 0x0000);
  EXPECT_EQ(FloatToFloat16(-0.0F).bits_, 0x8000);
  EXPECT_EQ(FloatToFloat16(1.0F).bits_, 0x3C00);
  EXPECT_EQ(FloatToFloat16(-2.0F).bits_, 0xC000);
  EXPECT_EQ(FloatToFloat16(65504.0F).bits_, 0x7BFF);
  EXPECT_EQ(FloatToFloat16(std::numeric_limits<float>::infinity()).bits_, 0x7C00);

  EXPECT_EQ(Float16ToFloat(Float16{.bits_ = 0x3C00}), 1.0F);
  EXPECT_EQ(Float16ToFloat(Float16{.bits_ = 0xC000}), -2.0F);
  EXPECT_TRUE(std::signbit(Float16ToFloat(Float16{.bits_ = 0x8000})));
  EXPECT_TRUE(std::isnan(Float16ToFloat(FloatToFloat16(std::numeric_limits<float>::quiet_NaN()))));
  EXPECT_EQ(FloatToFloat16(std::ldexp(1.0F, -24)).bits_, 0x0001);
  EXPECT_EQ(Float16ToFloat(Float16{.bits_ = 0x0001}), std::ldexp(1.0F, -24));
  EXPECT_TRUE(std::isinf(Float16ToFloat(FloatToFloat16(70000.0F))));
}

TEST(Float16Test, RoundsTiesToEven) {
  constexpr auto halfway_to_first_fraction = 1.0F + (1.0F / 2048.0F);
  constexpr auto halfway_to_second_fraction = 1.0F + (3.0F / 2048.0F);

  EXPECT_EQ(FloatToFloat16(halfway_to_first_fraction).bits_, 0x3C00);
  EXPECT_EQ(FloatToFloat16(halfway_to_second_fraction).bits_, 0x3C02);
}

TEST(BFloat16Test, ConvertsKnownValuesAndRoundsTiesToEven) {
  EXPECT_EQ(FloatToBFloat16(0.0F).bits_, 0x0000);
  EXPECT_EQ(FloatToBFloat16(-0.0F).bits_, 0x8000);
  EXPECT_EQ(FloatToBFloat16(1.0F).bits_, 0x3F80);
  EXPECT_EQ(FloatToBFloat16(1.5F).bits_, 0x3FC0);
  EXPECT_EQ(FloatToBFloat16(std::numeric_limits<float>::infinity()).bits_, 0x7F80);
  EXPECT_EQ(FloatToBFloat16(1.0F + (1.0F / 256.0F)).bits_, 0x3F80);
  EXPECT_EQ(FloatToBFloat16(1.0F + (3.0F / 256.0F)).bits_, 0x3F82);
  EXPECT_EQ(FloatToBFloat16(std::ldexp(1.0F, -133)).bits_, 0x0001);

  EXPECT_EQ(BFloat16ToFloat(BFloat16{.bits_ = 0x3F80}), 1.0F);
  EXPECT_EQ(BFloat16ToFloat(BFloat16{.bits_ = 0xBFC0}), -1.5F);
  EXPECT_EQ(BFloat16ToFloat(BFloat16{.bits_ = 0x0001}), std::ldexp(1.0F, -133));
  EXPECT_TRUE(std::signbit(BFloat16ToFloat(BFloat16{.bits_ = 0x8000})));
  EXPECT_TRUE(std::isnan(BFloat16ToFloat(FloatToBFloat16(std::numeric_limits<float>::quiet_NaN()))));
}

}  // namespace ttl
