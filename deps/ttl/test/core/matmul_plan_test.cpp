#include "ttl/internal/matmul_plan.hpp"

#include <cstdint>

#include <gtest/gtest.h>

namespace ttl::internal {
namespace {

[[nodiscard]] auto MakeKey(int64_t batch_count) -> MatmulAlgorithmKey {
  const auto layout = MatrixLayoutSignature{
      .rows_ = 16,
      .columns_ = 32,
      .leading_dimension_ = 32,
      .batch_stride_ = 512,
      .order_ = 1,
  };
  return {
      .lhs_ = layout,
      .rhs_ = layout,
      .output_ = layout,
      .batch_count_ = batch_count,
      .data_type_ = 2,
      .compute_type_ = 3,
      .lhs_operation_ = 0,
      .rhs_operation_ = 1,
      .epilogue_ = 0,
      .lhs_alignment_ = 256,
      .rhs_alignment_ = 256,
      .output_alignment_ = 256,
      .bias_alignment_ = 1,
      .workspace_limit_bytes_ = 1U << 20U,
  };
}

TEST(MatmulAlgorithmCacheTest, StoresUpdatesAndEvictsLeastRecentlyUsedChoices) {
  auto cache = MatmulAlgorithmCache{};
  const auto choice = MatmulAlgorithmChoice{.algorithm_ = {}, .workspace_bytes_ = 4096, .supported_ = true};
  EXPECT_FALSE(cache.Find(MakeKey(0)).has_value());

  for (int64_t batch_count = 0; batch_count < 256; ++batch_count) {
    cache.Insert(MakeKey(batch_count), choice);
  }
  ASSERT_TRUE(cache.Find(MakeKey(0)).has_value());

  cache.Insert(MakeKey(256), choice);
  EXPECT_TRUE(cache.Find(MakeKey(0)).has_value());
  EXPECT_FALSE(cache.Find(MakeKey(1)).has_value());

  const auto unsupported = MatmulAlgorithmChoice{.algorithm_ = {}, .workspace_bytes_ = 0, .supported_ = false};
  cache.Insert(MakeKey(0), unsupported);
  const auto updated = cache.Find(MakeKey(0));
  ASSERT_TRUE(updated.has_value());
  EXPECT_FALSE(updated->supported_);
  EXPECT_EQ(updated->workspace_bytes_, 0);
}

}  // namespace
}  // namespace ttl::internal
