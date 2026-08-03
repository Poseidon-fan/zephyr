#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "support/tensor_test_utils.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/indexing.hpp"
#include "ttl/ops/normalization.hpp"
#include "ttl/ops/reduction.hpp"
#include "ttl/ops/softmax.hpp"
#include "ttl/ops/topk.hpp"

namespace ttl {

TEST(ReductionTest, ReducesAxesAndReturnsStableArgIndices) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  const auto rows = ReductionOptions{.axes_ = {1}, .keep_dimensions_ = false};
  EXPECT_EQ(test::Download<float>(context, Sum(context, input, rows)), (std::vector<float>{6, 15}));
  EXPECT_EQ(test::Download<float>(context, Mean(context, input, rows)), (std::vector<float>{2, 5}));
  EXPECT_EQ(test::Download<float>(context, Maximum(context, input, rows)), (std::vector<float>{3, 6}));
  EXPECT_EQ(test::Download<int64_t>(context, ArgMax(context, input, 1)), (std::vector<int64_t>{2, 2}));

  auto ties = test::Upload(context, Shape{4}, std::vector<int32_t>{7, 9, 9, 3});
  EXPECT_EQ(test::Download<int64_t>(context, ArgMax(context, ties, 0)), (std::vector<int64_t>{1}));
}

TEST(SoftmaxTest, NormalizesRowsAndComputesLogSoftmax) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{2, 3}, std::vector<float>{1, 2, 3, -1, -1, -1});
  auto softmax = test::Download<float>(context, Softmax(context, input, SoftmaxOptions{.axes_ = {1}}));
  EXPECT_NEAR(softmax[0], 0.0900306F, 1.0e-6F);
  EXPECT_NEAR(softmax[1], 0.244728F, 1.0e-6F);
  EXPECT_NEAR(softmax[2], 0.665241F, 1.0e-6F);
  EXPECT_NEAR(softmax[3], 1.0F / 3.0F, 1.0e-6F);
  EXPECT_NEAR(softmax[4], 1.0F / 3.0F, 1.0e-6F);
  EXPECT_NEAR(softmax[5], 1.0F / 3.0F, 1.0e-6F);

  auto log_softmax = test::Download<float>(context, LogSoftmax(context, input, SoftmaxOptions{.axes_ = {1}}));
  for (size_t index = 0; index < softmax.size(); ++index) {
    EXPECT_NEAR(log_softmax[index], std::log(softmax[index]), 2.0e-6F);
  }
  EXPECT_THROW(static_cast<void>(Softmax(context, input, SoftmaxOptions{})), InvalidArgumentError);
}

TEST(NormalizationTest, ComputesLayerAndRmsNormalizationInFloatAccumulation) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{1, 3}, std::vector<float>{1, 2, 3});
  const auto options = NormOptions{.normalized_rank_ = 1, .epsilon_ = 0.0F};
  auto layer = test::Download<float>(context, LayerNorm(context, input, std::nullopt, std::nullopt, options));
  EXPECT_NEAR(layer[0], -1.2247449F, 1.0e-5F);
  EXPECT_NEAR(layer[1], 0.0F, 1.0e-5F);
  EXPECT_NEAR(layer[2], 1.2247449F, 1.0e-5F);

  auto rms = test::Download<float>(context, RmsNorm(context, input, std::nullopt, options));
  EXPECT_NEAR(rms[0], 0.462910F, 1.0e-5F);
  EXPECT_NEAR(rms[1], 0.925820F, 1.0e-5F);
  EXPECT_NEAR(rms[2], 1.388730F, 1.0e-5F);
}

TEST(IndexingTest, SelectsAndGathersAlongArbitraryAxes) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{2, 3}, std::vector<int32_t>{10, 11, 12, 20, 21, 22});
  auto indices = test::Upload(context, Shape{2}, std::vector<int64_t>{2, 0});
  auto selected = IndexSelect(context, input, 1, indices);
  EXPECT_EQ(test::Download<int32_t>(context, selected), (std::vector<int32_t>{12, 10, 22, 20}));

  auto gather_indices = test::Upload(context, Shape{2, 2}, std::vector<int32_t>{1, 0, 2, 1});
  auto gathered = Gather(context, input, 1, gather_indices);
  EXPECT_EQ(test::Download<int32_t>(context, gathered), (std::vector<int32_t>{11, 10, 22, 21}));
}

TEST(IndexingTest, SurfacesDeviceSideBoundsFailureAtExplicitErrorBoundary) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{2}, std::vector<float>{1, 2});
  auto invalid_index = test::Upload(context, Shape{1}, std::vector<int64_t>{2});
  static_cast<void>(IndexSelect(context, input, 0, invalid_index));
  EXPECT_THROW(context.CheckAsyncErrors(), DeviceError);
}

TEST(TopKTest, ReturnsSortedValuesAndOriginalIndices) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto input = test::Upload(context, Shape{2, 4}, std::vector<float>{3, 9, 1, 7, -1, -5, 4, 2});
  auto [values, indices] = TopK(context, input, TopKOptions{.axis_ = 1, .k_ = 2});
  EXPECT_EQ(test::Download<float>(context, values), (std::vector<float>{9, 7, 4, 2}));
  EXPECT_EQ(test::Download<int64_t>(context, indices), (std::vector<int64_t>{1, 3, 2, 3}));
  EXPECT_THROW(static_cast<void>(TopK(context, input, TopKOptions{.axis_ = 1, .k_ = 5})), InvalidArgumentError);
}

}  // namespace ttl
