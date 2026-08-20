#include "common/operator_attributes.h"
#include "gtest/gtest.h"

namespace zephyr {
namespace {

TEST(OperatorAttributesTest, AttentionWindowEquality) {
  const auto first = AttentionWindow{.left_ = 128, .right_ = 0};
  const auto same = AttentionWindow{.left_ = 128, .right_ = 0};
  const auto different = AttentionWindow{.left_ = 64, .right_ = 0};

  EXPECT_EQ(first, same);
  EXPECT_NE(first, different);
}

TEST(OperatorAttributesTest, ExpertGroupRoutingEquality) {
  const auto first = ExpertGroupRouting{
      .group_count_ = 8,
      .selected_group_count_ = 4,
      .score_function_ = GroupScoreFunction::TOP2_SUM,
  };
  const auto same = first;
  const auto different = ExpertGroupRouting{
      .group_count_ = 8,
      .selected_group_count_ = 2,
      .score_function_ = GroupScoreFunction::MAX,
  };

  EXPECT_EQ(first, same);
  EXPECT_NE(first, different);
}

}  // namespace
}  // namespace zephyr
