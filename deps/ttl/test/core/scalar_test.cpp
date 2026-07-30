#include "ttl/scalar.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"

namespace ttl {
namespace {

TEST(ScalarTest, PreservesSourceCategory) {
  EXPECT_TRUE(Scalar{true}.IsBoolean());
  EXPECT_TRUE(Scalar{int64_t{7}}.IsIntegral());
  EXPECT_TRUE(Scalar{7.0}.IsFloating());
}

TEST(ScalarTest, ConvertsBooleanAndIntegerValues) {
  EXPECT_EQ(Scalar{true}.Cast<int64_t>(), 1);
  EXPECT_EQ(Scalar{false}.Cast<float>(), 0.0F);
  EXPECT_TRUE(Scalar{int64_t{-1}}.Cast<bool>());
  EXPECT_FALSE(Scalar{int64_t{0}}.Cast<bool>());
  EXPECT_EQ(Scalar{int64_t{255}}.Cast<uint8_t>(), 255);
  EXPECT_EQ(Scalar{int64_t{-17}}.Cast<int32_t>(), -17);
}

TEST(ScalarTest, ChecksIntegralDestinationRange) {
  EXPECT_THROW([[maybe_unused]] const auto result = Scalar{int64_t{-1}}.Cast<uint8_t>(), OverflowError);
  EXPECT_THROW([[maybe_unused]] const auto result = Scalar{int64_t{256}}.Cast<uint8_t>(), OverflowError);
  EXPECT_THROW([[maybe_unused]] const auto result =
                   Scalar{static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1}.Cast<int32_t>(),
               OverflowError);
}

TEST(ScalarTest, RequiresFiniteIntegralFloatingValuesForIntegers) {
  EXPECT_EQ(Scalar{42.0}.Cast<int64_t>(), 42);
  EXPECT_THROW([[maybe_unused]] const auto result = Scalar{1.5}.Cast<int64_t>(), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = Scalar{std::numeric_limits<double>::infinity()}.Cast<int64_t>(),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = Scalar{std::numeric_limits<double>::quiet_NaN()}.Cast<int64_t>(),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = Scalar{0x1p63}.Cast<int64_t>(), OverflowError);
  EXPECT_THROW([[maybe_unused]] const auto result = Scalar{-1.0}.Cast<uint8_t>(), OverflowError);
}

TEST(ScalarTest, ConvertsFloatingStorageTypes) {
  const auto half = Scalar{1.5}.Cast<Float16>();
  const auto bfloat = Scalar{-2.25}.Cast<BFloat16>();
  EXPECT_FLOAT_EQ(Float16ToFloat(half), 1.5F);
  EXPECT_FLOAT_EQ(BFloat16ToFloat(bfloat), -2.25F);
  EXPECT_TRUE(std::isnan(Scalar{std::numeric_limits<double>::quiet_NaN()}.Cast<float>()));
  EXPECT_TRUE(Scalar{std::numeric_limits<double>::quiet_NaN()}.Cast<bool>());
}

}  // namespace
}  // namespace ttl
