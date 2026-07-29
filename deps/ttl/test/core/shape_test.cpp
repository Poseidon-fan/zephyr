#include "ttl/shape.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ttl/error.hpp"

namespace ttl {

using testing::ElementsAre;
using testing::HasSubstr;

static_assert(TTL_MAX_RANK == 8);

constexpr Shape SCALAR_SHAPE{};
constexpr Strides SCALAR_STRIDES{};
static_assert(SCALAR_SHAPE.GetRank() == 0);
static_assert(SCALAR_SHAPE.GetNumElements() == 1);
static_assert(SCALAR_SHAPE.IsScalar());
static_assert(!SCALAR_SHAPE.IsEmpty());
static_assert(SCALAR_SHAPE.GetDimensions().empty());
static_assert(SCALAR_STRIDES.GetRank() == 0);
static_assert(SCALAR_STRIDES.GetValues().empty());
static_assert(std::is_nothrow_copy_constructible_v<Shape>);
static_assert(std::is_nothrow_copy_constructible_v<Strides>);

TEST(ShapeTest, RepresentsScalarAndDenseDimensions) {
  const Shape scalar;
  EXPECT_TRUE(scalar.IsScalar());
  EXPECT_FALSE(scalar.IsEmpty());
  EXPECT_EQ(scalar.GetNumElements(), 1);
  EXPECT_TRUE(scalar.GetDimensions().empty());

  const Shape shape{2, 3, 4};
  EXPECT_EQ(shape.GetRank(), 3);
  EXPECT_EQ(shape.GetNumElements(), 24);
  EXPECT_FALSE(shape.IsScalar());
  EXPECT_FALSE(shape.IsEmpty());
  EXPECT_THAT(shape.GetDimensions(), ElementsAre(2, 3, 4));
  EXPECT_EQ(shape.GetDimension(1), 3);
}

TEST(ShapeTest, CopiesDimensionsIntoInlineStorage) {
  std::array<int64_t, 2> dimensions{2, 3};
  const Shape shape{dimensions};
  dimensions = {7, 11};

  EXPECT_THAT(shape.GetDimensions(), ElementsAre(2, 3));
  EXPECT_EQ(shape.GetNumElements(), 6);
}

TEST(ShapeTest, RepresentsEmptyShapeAndValidatesEveryDimension) {
  const Shape shape{2, 0, 4};
  EXPECT_TRUE(shape.IsEmpty());
  EXPECT_EQ(shape.GetNumElements(), 0);
  EXPECT_THAT(shape.GetDimensions(), ElementsAre(2, 0, 4));

  EXPECT_THROW(static_cast<void>(Shape{0, -1}), InvalidArgumentError);
  EXPECT_NO_THROW(
      static_cast<void>(Shape{std::numeric_limits<int64_t>::max(), 0, std::numeric_limits<int64_t>::max()}));
}

TEST(ShapeTest, SupportsMaximumRankAndRejectsLargerRank) {
  constexpr std::array<int64_t, TTL_MAX_RANK> maximum_rank{1, 2, 3, 4, 5, 6, 7, 8};
  const Shape shape{maximum_rank};
  EXPECT_EQ(shape.GetRank(), TTL_MAX_RANK);
  EXPECT_EQ(shape.GetNumElements(), 40320);

  constexpr std::array<int64_t, TTL_MAX_RANK + 1> excessive_rank{};
  EXPECT_THROW(static_cast<void>(Shape{excessive_rank}), InvalidArgumentError);
}

TEST(ShapeTest, ChecksElementCountOverflow) {
  const Shape boundary{std::numeric_limits<int64_t>::max(), 1};
  EXPECT_EQ(boundary.GetNumElements(), std::numeric_limits<int64_t>::max());

  EXPECT_THROW(static_cast<void>(Shape{std::numeric_limits<int64_t>::max(), 2}), OverflowError);
}

TEST(ShapeTest, PreservesOverflowCallSite) {
  uint_least32_t expected_line = 0;
  try {
    expected_line = __LINE__ + 1;
    static_cast<void>(Shape{std::numeric_limits<int64_t>::max(), 2});
  } catch (const OverflowError &error) {
    EXPECT_EQ(error.GetLocation().line(), expected_line);
    EXPECT_THAT(error.GetMessage(), HasSubstr("shape element count"));
    EXPECT_THAT(std::string_view(error.GetLocation().file_name()), HasSubstr("shape_test.cpp"));
    return;
  }
  FAIL() << "expected OverflowError";
}

TEST(ShapeTest, PreservesValidationCallSite) {
  uint_least32_t expected_line = 0;
  try {
    expected_line = __LINE__ + 1;
    static_cast<void>(Shape{2, -3});
  } catch (const InvalidArgumentError &error) {
    EXPECT_EQ(error.GetLocation().line(), expected_line);
    EXPECT_THAT(error.GetMessage(), HasSubstr("axis 1"));
    EXPECT_THAT(error.GetMessage(), HasSubstr("-3"));
    EXPECT_THAT(std::string_view(error.GetLocation().file_name()), HasSubstr("shape_test.cpp"));
    return;
  }
  FAIL() << "expected InvalidArgumentError";
}

TEST(ShapeTest, RejectsOutOfRangeDimensionAccess) {
  const Shape shape{2, 3};
  EXPECT_EQ(shape.GetDimension(0), 2);
  EXPECT_THROW(static_cast<void>(shape.GetDimension(2)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Shape{}.GetDimension(0)), InvalidArgumentError);
}

TEST(ShapeTest, ComparesAndFormatsDimensions) {
  EXPECT_EQ(Shape({2, 3}), Shape({2, 3}));
  EXPECT_NE(Shape({2, 3}), Shape({3, 2}));
  EXPECT_EQ(Shape{}.ToString(), "[]");
  EXPECT_EQ(Shape({2, 3}).ToString(), "[2, 3]");

  std::ostringstream stream;
  stream << std::hex << Shape{12, 16};
  EXPECT_EQ(stream.str(), "[12, 16]");
}

TEST(StridesTest, StoresElementStridesAndAllowsZero) {
  const Strides strides{12, 0, 1};
  EXPECT_EQ(strides.GetRank(), 3);
  EXPECT_THAT(strides.GetValues(), ElementsAre(12, 0, 1));
  EXPECT_EQ(strides.GetStride(1), 0);
  EXPECT_THROW(static_cast<void>(Strides{4, -1}), InvalidArgumentError);
}

TEST(StridesTest, CopiesValuesIntoInlineStorage) {
  std::array<int64_t, 2> values{3, 1};
  const Strides strides{values};
  values = {7, 2};

  EXPECT_THAT(strides.GetValues(), ElementsAre(3, 1));
}

TEST(StridesTest, SupportsMaximumRankAndChecksAccess) {
  constexpr std::array<int64_t, TTL_MAX_RANK> maximum_rank{8, 7, 6, 5, 4, 3, 2, 1};
  const Strides strides{maximum_rank};
  EXPECT_EQ(strides.GetRank(), TTL_MAX_RANK);
  EXPECT_THROW(static_cast<void>(strides.GetStride(TTL_MAX_RANK)), InvalidArgumentError);

  constexpr std::array<int64_t, TTL_MAX_RANK + 1> excessive_rank{};
  EXPECT_THROW(static_cast<void>(Strides{excessive_rank}), InvalidArgumentError);
}

TEST(StridesTest, ComparesAndFormatsValues) {
  EXPECT_EQ(Strides({3, 1}), Strides({3, 1}));
  EXPECT_NE(Strides({3, 1}), Strides({1, 3}));
  EXPECT_EQ(Strides{}.ToString(), "[]");
  EXPECT_EQ(Strides({3, 1}).ToString(), "[3, 1]");

  std::ostringstream stream;
  stream << std::hex << Strides{12, 1};
  EXPECT_EQ(stream.str(), "[12, 1]");
}

TEST(AxisTest, NormalizesPositiveAndNegativeAxes) {
  EXPECT_EQ(NormalizeAxis(0, 3), 0);
  EXPECT_EQ(NormalizeAxis(2, 3), 2);
  EXPECT_EQ(NormalizeAxis(-1, 3), 2);
  EXPECT_EQ(NormalizeAxis(-3, 3), 0);

  EXPECT_THROW(static_cast<void>(NormalizeAxis(3, 3)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(NormalizeAxis(-4, 3)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(NormalizeAxis(0, 0)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(NormalizeAxis(0, TTL_MAX_RANK + 1)), InvalidArgumentError);
}

TEST(AxisTest, NormalizesSortsAndRejectsDuplicateAxes) {
  constexpr std::array<int64_t, 3> axes{2, -3, 1};
  EXPECT_THAT(NormalizeAxes(axes, 3), ElementsAre(0, 1, 2));

  constexpr std::array<int64_t, 2> duplicates{1, -2};
  EXPECT_THROW(static_cast<void>(NormalizeAxes(duplicates, 3)), InvalidArgumentError);

  constexpr std::array<int64_t, 2> too_many{0, 0};
  EXPECT_THROW(static_cast<void>(NormalizeAxes(too_many, 1)), InvalidArgumentError);

  EXPECT_TRUE(NormalizeAxes(std::span<const int64_t>{}, 0).empty());
}

TEST(ContiguousStridesTest, ComputesCanonicalStrides) {
  EXPECT_EQ(GetContiguousStrides(Shape{}), Strides{});
  EXPECT_EQ(GetContiguousStrides(Shape{2, 3, 4}), Strides({12, 4, 1}));
  EXPECT_EQ(GetContiguousStrides(Shape{2, 1, 3}), Strides({3, 3, 1}));
  EXPECT_EQ(GetContiguousStrides(Shape{2, 0, 4}), Strides({4, 4, 1}));
}

TEST(ContiguousStridesTest, AvoidsUnusedOuterProductAndChecksRequiredSuffixes) {
  const auto maximum = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(GetContiguousStrides(Shape{maximum, 0, maximum}), Strides({maximum, maximum, 1}));
  EXPECT_THROW(static_cast<void>(GetContiguousStrides(Shape{0, maximum, maximum})), OverflowError);
}

TEST(ContiguousTest, HandlesScalarEmptyAndSizeOneDimensions) {
  EXPECT_TRUE(IsContiguous(Shape{}, Strides{}));
  EXPECT_TRUE(IsContiguous(Shape{2, 0, 4}, Strides{99, 0, 7}));
  EXPECT_TRUE(IsContiguous(Shape{2, 3, 4}, Strides{12, 4, 1}));
  EXPECT_TRUE(IsContiguous(Shape{2, 1, 3}, Strides{3, 0, 1}));
  EXPECT_TRUE(IsContiguous(Shape{1}, Strides{0}));
}

TEST(ContiguousTest, RejectsNonContiguousAndRankMismatch) {
  EXPECT_FALSE(IsContiguous(Shape{2, 3}, Strides{1, 2}));
  EXPECT_FALSE(IsContiguous(Shape{2, 3}, Strides{4, 1}));
  EXPECT_THROW(static_cast<void>(IsContiguous(Shape{0}, Strides{})), InvalidArgumentError);
}

}  // namespace ttl
