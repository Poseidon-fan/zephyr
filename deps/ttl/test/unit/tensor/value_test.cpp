#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ttl/common/error.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/scalar.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::test {
namespace {

TEST(DTypeTest, MetadataAndClassificationCoverEverySupportedDType) {
  ASSERT_EQ(TTL_DTYPE_INFOS.size(), TTL_DTYPE_COUNT);
  for (size_t index = 0; index < TTL_DTYPE_INFOS.size(); ++index) {
    const DTypeInfo info = TTL_DTYPE_INFOS[index];
    EXPECT_EQ(static_cast<size_t>(info.dtype_), index);
    EXPECT_FALSE(info.name_.empty());
    EXPECT_GT(info.size_bytes_, 0U);
    EXPECT_GT(info.alignment_bytes_, 0U);
    EXPECT_EQ(GetDTypeInfo(info.dtype_), info);
    EXPECT_EQ(ParseDType(static_cast<uint8_t>(index)), info.dtype_);
  }

  EXPECT_TRUE(IsBoolean(DType::BOOL));
  EXPECT_TRUE(IsUnsignedInteger(DType::UINT8));
  EXPECT_TRUE(IsSignedInteger(DType::INT32));
  EXPECT_TRUE(IsSignedInteger(DType::INT64));
  EXPECT_TRUE(IsIntegral(DType::UINT8));
  EXPECT_FALSE(IsIntegral(DType::BOOL));
  EXPECT_TRUE(IsFloating(DType::FLOAT16));
  EXPECT_TRUE(IsFloating(DType::BFLOAT16));
  EXPECT_TRUE(IsFloating(DType::FLOAT32));
  EXPECT_EQ(ParseDType(static_cast<uint8_t>(TTL_DTYPE_COUNT)), std::nullopt);
  // Exercise validation of an ABI value from outside this build.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_THROW(static_cast<void>(GetDTypeInfo(static_cast<DType>(UINT8_MAX))), InvalidArgumentError);
}

TEST(DTypeTest, HalfConversionsPreserveSpecialValuesAndKnownRepresentations) {
  EXPECT_EQ(FloatToFloat16(1.0F).bits_, 0x3C00U);
  EXPECT_EQ(FloatToBFloat16(1.0F).bits_, 0x3F80U);
  EXPECT_EQ(Float16ToFloat(FloatToFloat16(-0.0F)), -0.0F);
  EXPECT_TRUE(std::signbit(Float16ToFloat(FloatToFloat16(-0.0F))));
  EXPECT_TRUE(std::isinf(Float16ToFloat(FloatToFloat16(std::numeric_limits<float>::infinity()))));
  EXPECT_TRUE(std::isnan(Float16ToFloat(FloatToFloat16(std::numeric_limits<float>::quiet_NaN()))));
  EXPECT_TRUE(std::isinf(BFloat16ToFloat(FloatToBFloat16(std::numeric_limits<float>::infinity()))));
  EXPECT_TRUE(std::isnan(BFloat16ToFloat(FloatToBFloat16(std::numeric_limits<float>::quiet_NaN()))));
  EXPECT_EQ(Float16ToFloat(FloatToFloat16(65504.0F)), 65504.0F);
}

TEST(ScalarTest, PreservesCategoryAndPerformsCheckedConversions) {
  const Scalar boolean{true};
  const Scalar integer{int64_t{255}};
  const Scalar floating{3.0};

  EXPECT_TRUE(boolean.IsBoolean());
  EXPECT_TRUE(integer.IsIntegral());
  EXPECT_TRUE(floating.IsFloating());
  EXPECT_EQ(boolean.ToDouble(), 1.0);
  EXPECT_EQ(integer.Cast<uint8_t>(), UINT8_MAX);
  EXPECT_EQ(floating.Cast<int32_t>(), 3);
  EXPECT_FLOAT_EQ(Scalar{1.25}.Cast<float>(), 1.25F);
  EXPECT_TRUE(Scalar{-2.0}.Cast<bool>());

  EXPECT_THROW(static_cast<void>(Scalar{int64_t{-1}}.Cast<uint8_t>()), OverflowError);
  EXPECT_THROW(static_cast<void>(Scalar{int64_t{256}}.Cast<uint8_t>()), OverflowError);
  EXPECT_THROW(static_cast<void>(Scalar{1.5}.Cast<int32_t>()), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Scalar{std::numeric_limits<double>::infinity()}.Cast<int64_t>()),
               InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Scalar{65520.0}.Cast<Float16>()), OverflowError);
  EXPECT_THROW(static_cast<void>(Scalar{std::numeric_limits<double>::max()}.Cast<float>()), OverflowError);
}

TEST(ShapeTest, SupportsScalarEmptyAndMaximumRankShapes) {
  const Shape scalar;
  EXPECT_TRUE(scalar.IsScalar());
  EXPECT_FALSE(scalar.IsEmpty());
  EXPECT_EQ(scalar.GetRank(), 0U);
  EXPECT_EQ(scalar.GetNumElements(), 1);
  EXPECT_EQ(scalar.ToString(), "[]");

  const Shape empty{2, 0, std::numeric_limits<int64_t>::max()};
  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.GetNumElements(), 0);

  const Shape maximum_rank{1, 2, 1, 3, 1, 4, 1, 5};
  EXPECT_EQ(maximum_rank.GetRank(), TTL_MAX_RANK);
  EXPECT_EQ(maximum_rank.GetNumElements(), 120);
  EXPECT_EQ(maximum_rank.GetDimension(7), 5);

  std::ostringstream stream;
  stream << Shape{2, 3};
  EXPECT_EQ(stream.str(), "[2, 3]");
}

TEST(ShapeTest, RejectsInvalidRankDimensionsAxesAndElementOverflow) {
  const std::array<int64_t, TTL_MAX_RANK + 1> too_many{};
  EXPECT_THROW(static_cast<void>(Shape{too_many}), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Shape{2, -1}), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Shape{std::numeric_limits<int64_t>::max(), 2}), OverflowError);
  EXPECT_THROW(static_cast<void>(Shape{2, 3}.GetDimension(2)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(NormalizeAxis(0, 0)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(NormalizeAxis(-4, 3)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(NormalizeAxis(3, 3)), InvalidArgumentError);
}

TEST(StridesTest, ComputesCanonicalStridesAndTreatsSingletonStridesAsDontCare) {
  EXPECT_EQ(GetContiguousStrides(Shape{}), Strides{});
  EXPECT_EQ(GetContiguousStrides(Shape{2, 3, 4}), Strides({12, 4, 1}));
  EXPECT_EQ(GetContiguousStrides(Shape{2, 0, 4}), Strides({4, 4, 1}));
  EXPECT_TRUE(IsContiguous(Shape{2, 1, 3}, Strides{3, 99, 1}));
  EXPECT_FALSE(IsContiguous(Shape{2, 3}, Strides{1, 2}));
  EXPECT_TRUE(IsContiguous(Shape{2, 0, 3}, Strides{0, 0, 0}));

  EXPECT_THROW(static_cast<void>(Strides{1, -1}), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(IsContiguous(Shape{2}, Strides{1, 1})), InvalidArgumentError);
}

TEST(AxisTest, NormalizesSortsAndRejectsDuplicates) {
  EXPECT_EQ(NormalizeAxis(-1, 4), 3U);
  EXPECT_EQ(NormalizeAxis(-4, 4), 0U);
  EXPECT_EQ(NormalizeAxis(2, 4), 2U);

  const std::array<int64_t, 3> axes{2, -3, 1};
  EXPECT_EQ(NormalizeAxes(axes, 3), (std::vector<size_t>{0, 1, 2}));
  const std::array<int64_t, 2> duplicates{0, -3};
  EXPECT_THROW(static_cast<void>(NormalizeAxes(duplicates, 3)), InvalidArgumentError);
  const std::array<int64_t, 4> too_many{0, 1, 2, 3};
  EXPECT_THROW(static_cast<void>(NormalizeAxes(too_many, 3)), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl::test
