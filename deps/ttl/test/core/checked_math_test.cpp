#include "ttl/internal/checked_math.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ttl::internal {

using testing::HasSubstr;

static_assert(CheckedInteger<int8_t>);
static_assert(CheckedInteger<uint8_t>);
static_assert(CheckedInteger<uint64_t>);
static_assert(!CheckedInteger<bool>);
static_assert(!CheckedInteger<char>);
static_assert(!CheckedInteger<wchar_t>);
static_assert(!CheckedInteger<char8_t>);
static_assert(!CheckedInteger<char16_t>);
static_assert(!CheckedInteger<char32_t>);
static_assert(!CheckedInteger<const int32_t>);
static_assert(!CheckedInteger<volatile int32_t>);

TEST(CheckedAddTest, ReturnsRepresentableResults) {
  EXPECT_EQ(CheckedAdd<int32_t>(17, 25, "dimension sum"), 42);
  EXPECT_EQ(CheckedAdd<int64_t>(std::numeric_limits<int64_t>::min(), 1, "dimension sum"),
            std::numeric_limits<int64_t>::min() + 1);
  EXPECT_EQ(CheckedAdd<uint64_t>(std::numeric_limits<uint64_t>::max() - 1, 1, "byte count"),
            std::numeric_limits<uint64_t>::max());
}

TEST(CheckedAddTest, RejectsSignedAndUnsignedOverflow) {
  EXPECT_THROW(static_cast<void>(CheckedAdd<int64_t>(std::numeric_limits<int64_t>::max(), 1, "dimension sum")),
               OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedAdd<int64_t>(std::numeric_limits<int64_t>::min(), -1, "dimension sum")),
               OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedAdd<uint64_t>(std::numeric_limits<uint64_t>::max(), 1, "byte count")),
               OverflowError);
}

TEST(CheckedSubtractTest, ChecksBothBounds) {
  EXPECT_EQ(CheckedSubtract<int32_t>(20, 8, "slice length"), 12);
  EXPECT_THROW(static_cast<void>(CheckedSubtract<int64_t>(std::numeric_limits<int64_t>::min(), 1, "slice length")),
               OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedSubtract<int64_t>(std::numeric_limits<int64_t>::max(), -1, "slice length")),
               OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedSubtract<uint32_t>(0, 1, "slice length")), OverflowError);
}

TEST(CheckedMultiplyTest, ReturnsRepresentableResults) {
  EXPECT_EQ(CheckedMultiply<int64_t>(-7, 6, "stride product"), -42);
  EXPECT_EQ(CheckedMultiply<int64_t>(-7, -6, "stride product"), 42);
  EXPECT_EQ(CheckedMultiply<int64_t>(std::numeric_limits<int64_t>::min(), 1, "stride product"),
            std::numeric_limits<int64_t>::min());
  EXPECT_EQ(CheckedMultiply<uint64_t>(0, std::numeric_limits<uint64_t>::max(), "byte count"), 0);
}

TEST(CheckedMultiplyTest, RejectsSignedAndUnsignedOverflow) {
  EXPECT_THROW(static_cast<void>(CheckedMultiply<uint8_t>(200, 2, "small integer product")), OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedMultiply<int64_t>(std::numeric_limits<int64_t>::max(), 2, "stride product")),
               OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedMultiply<int64_t>(std::numeric_limits<int64_t>::min(), -1, "stride product")),
               OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedMultiply<int64_t>(std::numeric_limits<int64_t>::min(), 2, "stride product")),
               OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedMultiply<uint64_t>(std::numeric_limits<uint64_t>::max(), 2, "byte count")),
               OverflowError);
}

TEST(CheckedNarrowTest, HandlesSignednessAndWidth) {
  EXPECT_EQ(CheckedNarrow<uint8_t>(255, "rank"), 255);
  EXPECT_EQ(CheckedNarrow<int64_t>(std::numeric_limits<uint32_t>::max(), "element count"),
            std::numeric_limits<uint32_t>::max());
  EXPECT_THROW(static_cast<void>(CheckedNarrow<uint32_t>(-1, "element count")), OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedNarrow<int32_t>(std::numeric_limits<uint32_t>::max(), "element count")),
               OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedNarrow<int64_t>(std::numeric_limits<uint64_t>::max(), "element count")),
               OverflowError);
}

TEST(CheckedBytesTest, ChecksSignAndProduct) {
  EXPECT_EQ(CheckedBytes(0, sizeof(float)), 0);
  EXPECT_EQ(CheckedBytes(1024, sizeof(float)), 4096);
  EXPECT_EQ(CheckedBytes(size_t{1024}, sizeof(float)), 4096);
  EXPECT_THROW(static_cast<void>(CheckedBytes(-1, sizeof(float))), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(CheckedBytes(std::numeric_limits<int64_t>::max(), 3)), OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedBytes(std::numeric_limits<size_t>::max(), 2)), OverflowError);
}

TEST(CheckedElementOffsetToBytesTest, ChecksSignAndProduct) {
  EXPECT_EQ(CheckedElementOffsetToBytes(7, sizeof(float)), 28);
  EXPECT_EQ(CheckedElementOffsetToBytes(size_t{7}, sizeof(float)), 28);
  EXPECT_THROW(static_cast<void>(CheckedElementOffsetToBytes(-1, sizeof(float))), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(CheckedElementOffsetToBytes(std::numeric_limits<int64_t>::max(), 3)), OverflowError);
  EXPECT_THROW(static_cast<void>(CheckedElementOffsetToBytes(std::numeric_limits<size_t>::max(), 2)), OverflowError);
}

TEST(AlignUpTest, AlignsAndValidates) {
  EXPECT_EQ(AlignUp(0, 256), 0);
  EXPECT_EQ(AlignUp(256, 256), 256);
  EXPECT_EQ(AlignUp(257, 256), 512);
  EXPECT_EQ(AlignUp(13, 1), 13);
  EXPECT_THROW(static_cast<void>(AlignUp(13, 0)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(AlignUp(13, 3)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(AlignUp(std::numeric_limits<size_t>::max(), 2)), OverflowError);
}

TEST(CeilDivideTest, AvoidsAdditionOverflow) {
  EXPECT_EQ(CeilDivide<uint8_t>(255, 2), 128);
  EXPECT_EQ(CeilDivide<int64_t>(0, 7), 0);
  EXPECT_EQ(CeilDivide<int64_t>(1, 7), 1);
  EXPECT_EQ(CeilDivide<int64_t>(14, 7), 2);
  EXPECT_EQ(CeilDivide<int64_t>(15, 7), 3);
  EXPECT_EQ(CeilDivide<uint64_t>(std::numeric_limits<uint64_t>::max(), 2),
            (std::numeric_limits<uint64_t>::max() / 2) + 1);
}

TEST(CeilDivideTest, RejectsInvalidOperands) {
  EXPECT_THROW(static_cast<void>(CeilDivide<int64_t>(1, 0)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(CeilDivide<int64_t>(1, -1)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(CeilDivide<int64_t>(-1, 1)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(CeilDivide<uint64_t>(1, 0)), InvalidArgumentError);
}

TEST(CheckedMathTest, PreservesDescriptionAndCallSite) {
  uint_least32_t expected_line = 0;
  try {
    expected_line = __LINE__ + 1;
    static_cast<void>(CheckedMultiply<int64_t>(std::numeric_limits<int64_t>::max(), 2, "tensor byte size"));
  } catch (const OverflowError &error) {
    EXPECT_EQ(error.GetCode(), ErrorCode::OVERFLOW);
    EXPECT_EQ(error.GetLocation().line(), expected_line);
    EXPECT_THAT(error.GetMessage(), HasSubstr("tensor byte size"));
    EXPECT_THAT(error.GetMessage(), HasSubstr("overflow"));
    EXPECT_THAT(std::string_view(error.GetLocation().file_name()), HasSubstr("checked_math_test.cpp"));
    return;
  }
  FAIL() << "expected OverflowError";
}

}  // namespace ttl::internal
