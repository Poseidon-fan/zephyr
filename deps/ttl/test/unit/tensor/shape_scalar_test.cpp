#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "ttl/common/error.hpp"
#include "ttl/tensor/scalar.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl {

TEST(ShapeTest, RepresentsScalarEmptyAndRankEightShapes) {
  const Shape scalar;
  EXPECT_TRUE(scalar.IsScalar());
  EXPECT_FALSE(scalar.IsEmpty());
  EXPECT_EQ(scalar.GetNumElements(), 1);

  const Shape empty{2, 0, 7};
  EXPECT_TRUE(empty.IsEmpty());
  EXPECT_EQ(empty.GetNumElements(), 0);

  const Shape maximum_rank{1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_EQ(maximum_rank.GetRank(), TTL_MAX_RANK);
  EXPECT_EQ(maximum_rank.GetNumElements(), 40320);
}

TEST(ShapeTest, ChecksInvalidDimensionsRankAndOverflow) {
  EXPECT_THROW(static_cast<void>(Shape{2, -1}), InvalidArgumentError);
  const std::array<int64_t, TTL_MAX_RANK + 1> too_many_dimensions{};
  EXPECT_THROW(static_cast<void>(Shape{std::span<const int64_t>{too_many_dimensions}}), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Shape{std::numeric_limits<int64_t>::max(), 2}), OverflowError);
  EXPECT_THROW(static_cast<void>(Shape{2, 3}.GetDimension(2)), InvalidArgumentError);
}

TEST(ShapeTest, NormalizesSortsAndRejectsAxes) {
  EXPECT_EQ(NormalizeAxis(-1, 4), 3);
  EXPECT_EQ(NormalizeAxis(-4, 4), 0);
  EXPECT_EQ(NormalizeAxes(std::array<int64_t, 3>{-1, 0, 2}, 4), (std::vector<size_t>{0, 2, 3}));
  EXPECT_THROW(static_cast<void>(NormalizeAxis(0, 0)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(NormalizeAxis(4, 4)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(NormalizeAxes(std::array<int64_t, 2>{-1, 3}, 4)), InvalidArgumentError);
}

TEST(ShapeTest, ComputesCanonicalContiguity) {
  const Shape shape{2, 1, 3, 4};
  EXPECT_EQ(GetContiguousStrides(shape), Strides({12, 12, 4, 1}));
  EXPECT_TRUE(IsContiguous(shape, Strides{12, 999, 4, 1}));
  EXPECT_FALSE(IsContiguous(shape, Strides{12, 12, 1, 3}));
  EXPECT_TRUE(IsContiguous(Shape{2, 0, 4}, Strides{100, 10, 1}));
  EXPECT_THROW(static_cast<void>(IsContiguous(shape, Strides{1, 1})), InvalidArgumentError);
}

TEST(ScalarTest, PreservesCategoryAndChecksIntegerConversions) {
  const Scalar boolean{true};
  const Scalar integer{int64_t{-7}};
  const Scalar floating{3.0};
  EXPECT_TRUE(boolean.IsBoolean());
  EXPECT_TRUE(integer.IsIntegral());
  EXPECT_TRUE(floating.IsFloating());
  EXPECT_EQ(boolean.Cast<int32_t>(), 1);
  EXPECT_EQ(integer.Cast<int32_t>(), -7);
  EXPECT_EQ(floating.Cast<int64_t>(), 3);
  EXPECT_THROW(static_cast<void>(Scalar{-1.0}.Cast<uint8_t>()), OverflowError);
  EXPECT_THROW(static_cast<void>(Scalar{1.5}.Cast<int32_t>()), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Scalar{std::numeric_limits<double>::infinity()}.Cast<int64_t>()),
               InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Scalar{std::numeric_limits<double>::max()}.Cast<float>()), OverflowError);
  EXPECT_THROW(static_cast<void>(Scalar{65505.0}.Cast<Float16>()), OverflowError);
  EXPECT_THROW(static_cast<void>(Scalar{0x1.ffp127}.Cast<BFloat16>()), OverflowError);
  EXPECT_TRUE(std::isinf(Scalar{std::numeric_limits<double>::infinity()}.Cast<float>()));
}

}  // namespace ttl
