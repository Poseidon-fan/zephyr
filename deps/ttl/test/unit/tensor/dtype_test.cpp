#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>

#include <gtest/gtest.h>

#include "ttl/common/error.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl {

TEST(DTypeTest, ReportsCompleteMetadata) {
  constexpr std::array expected{
      DTypeInfo{DType::BOOL, "bool", 1, 1, DTypeCategory::BOOLEAN},
      DTypeInfo{DType::UINT8, "uint8", 1, 1, DTypeCategory::UNSIGNED_INTEGER},
      DTypeInfo{DType::INT32, "int32", 4, 4, DTypeCategory::SIGNED_INTEGER},
      DTypeInfo{DType::INT64, "int64", 8, 8, DTypeCategory::SIGNED_INTEGER},
      DTypeInfo{DType::FLOAT16, "float16", 2, 2, DTypeCategory::FLOATING},
      DTypeInfo{DType::BFLOAT16, "bfloat16", 2, 2, DTypeCategory::FLOATING},
      DTypeInfo{DType::FLOAT32, "float32", 4, 4, DTypeCategory::FLOATING},
  };
  for (const auto &info : expected) {
    EXPECT_EQ(GetDTypeInfo(info.dtype_), info);
    EXPECT_EQ(GetDTypeName(info.dtype_), info.name_);
    EXPECT_EQ(GetDTypeSize(info.dtype_), info.size_bytes_);
    EXPECT_EQ(GetDTypeAlignment(info.dtype_), info.alignment_bytes_);
  }
}

TEST(DTypeTest, RejectsInvalidEnumInReleaseChecks) {
  constexpr auto raw = uint8_t{255};
  EXPECT_EQ(TryParseDType(raw), std::nullopt);
  EXPECT_THROW(static_cast<void>(GetDTypeInfo(static_cast<DType>(raw))), InvalidArgumentError);
}

TEST(DTypeTest, ClassifiesDTypesWithoutTreatingBoolAsIntegral) {
  EXPECT_TRUE(IsBoolean(DType::BOOL));
  EXPECT_FALSE(IsIntegral(DType::BOOL));
  EXPECT_TRUE(IsUnsignedInteger(DType::UINT8));
  EXPECT_TRUE(IsIntegral(DType::UINT8));
  EXPECT_TRUE(IsSignedInteger(DType::INT32));
  EXPECT_TRUE(IsSignedInteger(DType::INT64));
  EXPECT_TRUE(IsFloating(DType::FLOAT16));
  EXPECT_TRUE(IsFloating(DType::BFLOAT16));
  EXPECT_TRUE(IsFloating(DType::FLOAT32));
}

TEST(DTypeTest, ConvertsHalfSpecialValuesAndFiniteValues) {
  constexpr std::array values{0.0F,
                              -0.0F,
                              1.0F,
                              -2.5F,
                              65504.0F,
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()};
  for (const auto value : values) {
    const auto converted = Float16ToFloat(FloatToFloat16(value));
    if (value == 0.0F) {
      EXPECT_EQ(std::signbit(converted), std::signbit(value));
    } else {
      EXPECT_EQ(converted, value);
    }
  }
  EXPECT_TRUE(std::isnan(Float16ToFloat(FloatToFloat16(std::numeric_limits<float>::quiet_NaN()))));
}

TEST(DTypeTest, ConvertsBFloat16WithRoundToNearestEven) {
  EXPECT_EQ(BFloat16ToFloat(FloatToBFloat16(1.0F)), 1.0F);
  EXPECT_EQ(BFloat16ToFloat(FloatToBFloat16(-0.0F)), -0.0F);
  EXPECT_EQ(FloatToBFloat16(std::bit_cast<float>(uint32_t{0x3F808000})).bits_, uint16_t{0x3F80});
  EXPECT_EQ(FloatToBFloat16(std::bit_cast<float>(uint32_t{0x3F818000})).bits_, uint16_t{0x3F82});
  EXPECT_TRUE(std::isnan(BFloat16ToFloat(FloatToBFloat16(std::numeric_limits<float>::quiet_NaN()))));
}

}  // namespace ttl
