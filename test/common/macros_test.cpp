#include <cstdint>

#include "common/exception.hpp"
#include "common/macros.hpp"
#include "gtest/gtest.h"

namespace zephyr {
namespace {

TEST(MacrosTest, AssertEvaluation) {
  int evaluations = 0;
  ZEPHYR_ASSERT(++evaluations == 1, "expression must be evaluated once");

#ifndef NDEBUG
  EXPECT_EQ(evaluations, 1);
#else
  EXPECT_EQ(evaluations, 0);
#endif
}

#ifndef NDEBUG
TEST(MacrosTest, AssertFailure) {
  EXPECT_DEATH({ ZEPHYR_ASSERT(false, "assertion failed"); }, "assertion `false` failed: assertion failed");
}
#endif

TEST(MacrosTest, EnsureEvaluation) {
  int evaluations = 0;
  ZEPHYR_ENSURE(++evaluations == 1, "expression must be evaluated once");
  EXPECT_EQ(evaluations, 1);
}

TEST(MacrosTest, EnsureFailure) {
  EXPECT_DEATH({ ZEPHYR_ENSURE(false, "ensure failed"); }, "ensure `false` failed: ensure failed");
}

TEST(MacrosTest, Unimplemented) {
  uint_least32_t expected_line = 0;
  try {
    expected_line = __LINE__ + 1;
    ZEPHYR_UNIMPLEMENTED("feature is not implemented");
  } catch (const NotImplementedException &exception) {
    EXPECT_STREQ(exception.what(), "feature is not implemented");
    EXPECT_STREQ(exception.GetLocation().file_name(), __FILE__);
    EXPECT_EQ(exception.GetLocation().line(), expected_line);
    return;
  }
  FAIL() << "ZEPHYR_UNIMPLEMENTED did not throw";
}

TEST(MacrosTest, Unreachable) {
  EXPECT_DEATH({ ZEPHYR_UNREACHABLE("invalid state"); }, "unreachable code reached: invalid state");
}

}  // namespace
}  // namespace zephyr
