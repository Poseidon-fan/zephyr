#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/cast.hpp"
#include "ttl/ops/indexing.hpp"
#include "ttl/ops/normalization.hpp"
#include "ttl/ops/reduction.hpp"
#include "ttl/ops/scan.hpp"
#include "ttl/ops/softmax.hpp"
#include "ttl/ops/topk.hpp"
#include "ttl/tensor/layout.hpp"

namespace ttl::test {
namespace {

class OperatorContractSmokeTest : public SingleDeviceTest {};

TEST_F(OperatorContractSmokeTest, ReducesAxesAndReturnsStableArgIndices) {
  auto &context = GetContext();
  auto input = Upload(context, Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  const auto rows = ReductionOptions{.axes_ = {1}, .keep_dimensions_ = false};
  EXPECT_EQ(Download<float>(context, Sum(context, input, rows)), (std::vector<float>{6, 15}));
  EXPECT_EQ(Download<float>(context, Mean(context, input, rows)), (std::vector<float>{2, 5}));
  EXPECT_EQ(Download<float>(context, Maximum(context, input, rows)), (std::vector<float>{3, 6}));
  EXPECT_EQ(Download<int64_t>(context, ArgMax(context, input, 1)), (std::vector<int64_t>{2, 2}));

  auto ties = Upload(context, Shape{4}, std::vector<int32_t>{7, 9, 9, 3});
  EXPECT_EQ(Download<int64_t>(context, ArgMax(context, ties, 0)), (std::vector<int64_t>{1}));
}

TEST_F(OperatorContractSmokeTest, NormalizesRowsAndComputesLogSoftmax) {
  auto &context = GetContext();
  auto input = Upload(context, Shape{2, 3}, std::vector<float>{1, 2, 3, -1, -1, -1});
  auto softmax = Download<float>(context, Softmax(context, input, SoftmaxOptions{.axes_ = {1}}));
  EXPECT_NEAR(softmax[0], 0.0900306F, 1.0e-6F);
  EXPECT_NEAR(softmax[1], 0.244728F, 1.0e-6F);
  EXPECT_NEAR(softmax[2], 0.665241F, 1.0e-6F);
  EXPECT_NEAR(softmax[3], 1.0F / 3.0F, 1.0e-6F);
  EXPECT_NEAR(softmax[4], 1.0F / 3.0F, 1.0e-6F);
  EXPECT_NEAR(softmax[5], 1.0F / 3.0F, 1.0e-6F);

  auto log_softmax = Download<float>(context, LogSoftmax(context, input, SoftmaxOptions{.axes_ = {1}}));
  for (size_t index = 0; index < softmax.size(); ++index) {
    EXPECT_NEAR(log_softmax[index], std::log(softmax[index]), 2.0e-6F);
  }
  EXPECT_THROW(static_cast<void>(Softmax(context, input, SoftmaxOptions{})), InvalidArgumentError);
}

TEST_F(OperatorContractSmokeTest, ComputesLayerAndRmsNormalizationInFloatAccumulation) {
  auto &context = GetContext();
  auto input = Upload(context, Shape{1, 3}, std::vector<float>{1, 2, 3});
  const auto options = NormOptions{.normalized_rank_ = 1, .epsilon_ = 0.0F};
  auto layer = Download<float>(context, LayerNorm(context, input, std::nullopt, std::nullopt, options));
  EXPECT_NEAR(layer[0], -1.2247449F, 1.0e-5F);
  EXPECT_NEAR(layer[1], 0.0F, 1.0e-5F);
  EXPECT_NEAR(layer[2], 1.2247449F, 1.0e-5F);

  auto rms = Download<float>(context, RmsNorm(context, input, std::nullopt, options));
  EXPECT_NEAR(rms[0], 0.462910F, 1.0e-5F);
  EXPECT_NEAR(rms[1], 0.925820F, 1.0e-5F);
  EXPECT_NEAR(rms[2], 1.388730F, 1.0e-5F);
}

TEST_F(OperatorContractSmokeTest, SelectsAndGathersAlongArbitraryAxes) {
  auto &context = GetContext();
  auto input = Upload(context, Shape{2, 3}, std::vector<int32_t>{10, 11, 12, 20, 21, 22});
  auto indices = Upload(context, Shape{2}, std::vector<int64_t>{2, 0});
  auto selected = IndexSelect(context, input, 1, indices);
  EXPECT_EQ(Download<int32_t>(context, selected), (std::vector<int32_t>{12, 10, 22, 20}));

  auto gather_indices = Upload(context, Shape{2, 2}, std::vector<int32_t>{1, 0, 2, 1});
  auto gathered = Gather(context, input, 1, gather_indices);
  EXPECT_EQ(Download<int32_t>(context, gathered), (std::vector<int32_t>{11, 10, 22, 21}));
}

TEST_F(OperatorContractSmokeTest, GathersBooleanRows) {
  auto &context = GetContext();
  auto table = Cast(context, Upload(context, Shape{3, 2}, std::vector<uint8_t>{1, 0, 0, 1, 1, 1}), DType::BOOL);
  auto indices = Upload(context, Shape{2}, std::vector<int32_t>{2, 0});
  auto output = GatherRows(context, table, indices);
  EXPECT_EQ(output.GetDType(), DType::BOOL);
  EXPECT_EQ(Download<uint8_t>(context, output), (std::vector<uint8_t>{1, 1, 1, 0}));
}

TEST_F(OperatorContractSmokeTest, SurfacesDeviceSideBoundsFailureAtExplicitErrorBoundary) {
  auto &context = GetContext();
  auto input = Upload(context, Shape{2}, std::vector<float>{1, 2});
  auto invalid_index = Upload(context, Shape{1}, std::vector<int64_t>{2});
  static_cast<void>(IndexSelect(context, input, 0, invalid_index));
  EXPECT_THROW(context.CheckAsyncErrors(), DeviceError);
}

TEST_F(OperatorContractSmokeTest, ScattersElementsIntoCopiedAndInPlaceOutputs) {
  auto &context = GetContext();
  auto input = Upload(context, Shape{2, 4}, std::vector<int32_t>{0, 1, 2, 3, 10, 11, 12, 13});
  auto indices = Upload(context, Shape{2, 2}, std::vector<int64_t>{3, 1, 0, 2});
  auto source = Upload(context, Shape{2, 2}, std::vector<int32_t>{30, 10, 100, 120});

  auto scattered = ScatterElements(context, input, 1, indices, source);
  EXPECT_EQ(Download<int32_t>(context, scattered), (std::vector<int32_t>{0, 10, 2, 30, 100, 11, 120, 13}));
  EXPECT_EQ(Download<int32_t>(context, input), (std::vector<int32_t>{0, 1, 2, 3, 10, 11, 12, 13}));

  ScatterElementsOut(context, input, input, 1, indices, source);
  EXPECT_EQ(Download<int32_t>(context, input), (std::vector<int32_t>{0, 10, 2, 30, 100, 11, 120, 13}));
}

TEST_F(OperatorContractSmokeTest, ComputesStridedAndModularCumulativeSums) {
  auto &context = GetContext();
  auto input = Upload(context, Shape{2, 3}, std::vector<int32_t>{1, 2, 3, 4, 5, 6});
  EXPECT_EQ(Download<int32_t>(context, CumulativeSum(context, input, 1)), (std::vector<int32_t>{1, 3, 6, 4, 9, 15}));

  auto transposed = Transpose(input, 0, 1);
  EXPECT_EQ(Download<int32_t>(context, CumulativeSum(context, transposed, 0)),
            (std::vector<int32_t>{1, 4, 3, 9, 6, 15}));

  CumulativeSumOut(context, input, input, 1);
  EXPECT_EQ(Download<int32_t>(context, input), (std::vector<int32_t>{1, 3, 6, 4, 9, 15}));

  auto bytes = Upload(context, Shape{2}, std::vector<uint8_t>{250, 10});
  EXPECT_EQ(Download<uint8_t>(context, CumulativeSum(context, bytes, 0)), (std::vector<uint8_t>{250, 4}));
}

TEST_F(OperatorContractSmokeTest, ReturnsSortedValuesAndOriginalIndices) {
  auto &context = GetContext();
  auto input = Upload(context, Shape{2, 4}, std::vector<float>{3, 9, 1, 7, -1, -5, 4, 2});
  auto [values, indices] = TopK(context, input, TopKOptions{.axis_ = 1, .k_ = 2});
  EXPECT_EQ(Download<float>(context, values), (std::vector<float>{9, 7, 4, 2}));
  EXPECT_EQ(Download<int64_t>(context, indices), (std::vector<int64_t>{1, 3, 2, 3}));
  EXPECT_THROW(static_cast<void>(TopK(context, input, TopKOptions{.axis_ = 1, .k_ = 5})), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl::test
