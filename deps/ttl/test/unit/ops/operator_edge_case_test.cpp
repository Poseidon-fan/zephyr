#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/attention.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/ops/matmul.hpp"
#include "ttl/ops/random.hpp"
#include "ttl/ops/reduction.hpp"
#include "ttl/ops/softmax.hpp"
#include "ttl/ops/topk.hpp"
#include "ttl/runtime/generator.hpp"

namespace ttl::test {
namespace {

class OperatorEdgeCaseTest : public SingleDeviceTest {};

TEST_F(OperatorEdgeCaseTest, FloatingMinimumAndMaximumPropagateNanFromEitherOperand) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor lhs = FloatingTensorFromValues(GetContext(), Shape{3}, DType::FLOAT32, {1.0F, nan, 3.0F});
  Tensor rhs = FloatingTensorFromValues(GetContext(), Shape{3}, DType::FLOAT32, {nan, 2.0F, 4.0F});
  const auto minimum = FloatingTensorToValues(GetContext(), Minimum(GetContext(), lhs, rhs));
  const auto maximum = FloatingTensorToValues(GetContext(), Maximum(GetContext(), lhs, rhs));
  ASSERT_EQ(minimum.size(), 3U);
  ASSERT_EQ(maximum.size(), 3U);
  EXPECT_TRUE(std::isnan(minimum[0]));
  EXPECT_TRUE(std::isnan(minimum[1]));
  EXPECT_FLOAT_EQ(minimum[2], 3.0F);
  EXPECT_TRUE(std::isnan(maximum[0]));
  EXPECT_TRUE(std::isnan(maximum[1]));
  EXPECT_FLOAT_EQ(maximum[2], 4.0F);
}

TEST_F(OperatorEdgeCaseTest, LinearSupportsExactAndTanhGeluActivations) {
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{3}, DType::FLOAT32, {-1.0F, 0.0F, 1.0F});
  Tensor weight = FloatingTensorFromValues(GetContext(), Shape{1, 3}, DType::FLOAT32, {1.0F, 1.0F, 1.0F});
  Tensor bias = FloatingTensorFromValues(GetContext(), Shape{1}, DType::FLOAT32, {0.0F});
  const auto exact = FloatingTensorToValues(GetContext(), Linear(GetContext(), input, weight, bias,
                                                                 {.activation_ = LinearActivation::GELU,
                                                                  .gelu_approximation_ = GeluApproximation::NONE,
                                                                  .matmul_ = {.allow_tf32_ = false}}));
  const auto tanh = FloatingTensorToValues(GetContext(), Linear(GetContext(), input, weight, bias,
                                                                {.activation_ = LinearActivation::GELU,
                                                                 .gelu_approximation_ = GeluApproximation::TANH,
                                                                 .matmul_ = {.allow_tf32_ = false}}));
  ASSERT_EQ(exact.size(), 1U);
  ASSERT_EQ(tanh.size(), 1U);
  EXPECT_NEAR(exact[0], 0.0F, 1.0e-6F);
  EXPECT_NEAR(tanh[0], 0.0F, 1.0e-6F);

  Tensor positive_input = FloatingTensorFromValues(GetContext(), Shape{1}, DType::FLOAT32, {1.0F});
  Tensor positive_weight = FloatingTensorFromValues(GetContext(), Shape{1, 1}, DType::FLOAT32, {1.0F});
  Tensor positive_exact = Linear(GetContext(), positive_input, positive_weight, bias,
                                 {.activation_ = LinearActivation::GELU,
                                  .gelu_approximation_ = GeluApproximation::NONE,
                                  .matmul_ = {.allow_tf32_ = false}});
  Tensor positive_tanh = Linear(GetContext(), positive_input, positive_weight, bias,
                                {.activation_ = LinearActivation::GELU,
                                 .gelu_approximation_ = GeluApproximation::TANH,
                                 .matmul_ = {.allow_tf32_ = false}});
  EXPECT_NEAR(FloatingTensorToValues(GetContext(), positive_exact)[0], 0.8413447F, 2.0e-5F);
  EXPECT_NEAR(FloatingTensorToValues(GetContext(), positive_tanh)[0], 0.841191F, 2.0e-5F);
}

TEST_F(OperatorEdgeCaseTest, MatmulAllowTf32OptionIsAcceptedForFloat32) {
  Tensor lhs = FloatingTensorFromValues(GetContext(), Shape{2, 2}, DType::FLOAT32, {1, 2, 3, 4});
  Tensor rhs = FloatingTensorFromValues(GetContext(), Shape{2, 2}, DType::FLOAT32, {5, 6, 7, 8});
  const auto strict = FloatingTensorToValues(GetContext(), Matmul(GetContext(), lhs, rhs, {.allow_tf32_ = false}));
  const auto fast = FloatingTensorToValues(GetContext(), Matmul(GetContext(), lhs, rhs, {.allow_tf32_ = true}));
  EXPECT_EQ(strict, (std::vector<float>{19, 22, 43, 50}));
  ASSERT_EQ(fast.size(), strict.size());
  for (size_t index = 0; index < strict.size(); ++index) {
    EXPECT_NEAR(fast[index], strict[index], 1.0e-3F);
  }
}

TEST_F(OperatorEdgeCaseTest, SdpaSupportsAdditiveMaskLowerRightCausalAlignmentAndGqa) {
  Tensor query = FloatingTensorFromValues(GetContext(), Shape{1, 1, 2, 1}, DType::FLOAT32, {0, 0});
  Tensor key = FloatingTensorFromValues(GetContext(), Shape{1, 1, 3, 1}, DType::FLOAT32, {0, 0, 0});
  Tensor value = FloatingTensorFromValues(GetContext(), Shape{1, 1, 3, 1}, DType::FLOAT32, {10, 20, 30});
  Tensor additive =
      FloatingTensorFromValues(GetContext(), Shape{1, 1, 2, 3}, DType::FLOAT32, {0, -100.0F, -100.0F, 0, 0, -100.0F});
  const auto masked = FloatingTensorToValues(
      GetContext(), ScaledDotProductAttention(GetContext(), query, key, value, additive, {.scale_ = 1.0F}));
  ASSERT_EQ(masked.size(), 2U);
  EXPECT_NEAR(masked[0], 10.0F, 1.0e-3F);
  EXPECT_NEAR(masked[1], 15.0F, 1.0e-3F);

  const auto lower_right = FloatingTensorToValues(
      GetContext(),
      ScaledDotProductAttention(GetContext(), query, key, value, std::nullopt,
                                {.scale_ = 1.0F, .causal_ = true, .causal_alignment_ = CausalAlignment::LOWER_RIGHT}));
  ASSERT_EQ(lower_right.size(), 2U);
  EXPECT_NEAR(lower_right[0], 15.0F, 1.0e-3F);
  EXPECT_NEAR(lower_right[1], 20.0F, 1.0e-3F);

  Tensor gqa_query = FloatingTensorFromValues(GetContext(), Shape{1, 2, 2, 1}, DType::FLOAT32, {0, 0, 0, 0});
  const Tensor gqa = ScaledDotProductAttention(GetContext(), gqa_query, key, value);
  EXPECT_EQ(gqa.GetShape(), Shape({1, 2, 2, 1}));
  for (float result : FloatingTensorToValues(GetContext(), gqa)) {
    EXPECT_NEAR(result, 20.0F, 1.0e-3F);
  }
}

TEST_F(OperatorEdgeCaseTest, NormalNonZeroStandardDeviationHasStableStatistics) {
  Generator generator{GetContext(), 1234};
  const Tensor samples =
      Normal(GetContext(), Shape{4096}, DType::FLOAT32, generator, {.mean_ = 3.0, .standard_deviation_ = 2.0});
  const auto values = FloatingTensorToValues(GetContext(), samples);
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double variance = 0.0;
  for (float value : values) {
    const double delta = value - mean;
    variance += delta * delta;
  }
  variance /= values.size();
  EXPECT_NEAR(mean, 3.0, 0.15);
  EXPECT_NEAR(std::sqrt(variance), 2.0, 0.15);
}

TEST_F(OperatorEdgeCaseTest, FloatingDtypesSupportElementwiseAndReductionContracts) {
  for (DType dtype : {DType::FLOAT16, DType::BFLOAT16}) {
    Tensor lhs = FloatingTensorFromValues(GetContext(), Shape{2, 2}, dtype, {1, 2, 3, 4});
    Tensor rhs = FloatingTensorFromValues(GetContext(), Shape{2, 2}, dtype, {5, 6, 7, 8});
    ExpectFloatValues(GetContext(), Add(GetContext(), lhs, rhs), {6, 8, 10, 12}, 0.02F, 0.02F);
    ExpectFloatValues(GetContext(), Sum(GetContext(), lhs, {.axes_ = {1}}), {3, 7}, 0.02F, 0.02F);
    ExpectFloatValues(GetContext(), Softmax(GetContext(), lhs, {.axes_ = {1}}),
                      {0.2689414F, 0.7310586F, 0.2689414F, 0.7310586F}, 0.02F, 0.02F);
  }
}

TEST_F(OperatorEdgeCaseTest, TopKSupportsSortedFallbackSizeAndInt64Indices) {
  std::vector<float> host(2048);
  std::iota(host.begin(), host.end(), 0.0F);
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{2048}, DType::FLOAT32, host);
  auto [values, indices] = TopK(GetContext(), input, {.axis_ = 0, .k_ = 1025, .largest_ = true, .sorted_ = true});
  EXPECT_EQ(values.GetShape(), Shape({1025}));
  EXPECT_EQ(indices.GetDType(), DType::INT64);
  const auto result_indices = TensorToValues<int64_t>(GetContext(), indices);
  ASSERT_EQ(result_indices.size(), 1025U);
  EXPECT_EQ(result_indices.front(), 2047);
  EXPECT_EQ(result_indices.back(), 1023);
}

}  // namespace
}  // namespace ttl::test
