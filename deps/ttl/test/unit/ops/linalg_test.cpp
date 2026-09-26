#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/attention.hpp"
#include "ttl/ops/matmul.hpp"
#include "ttl/ops/normalization.hpp"
#include "ttl/ops/softmax.hpp"

namespace ttl::test {
namespace {

class LinalgNormalizationTest : public SingleDeviceTest {};

TEST_F(LinalgNormalizationTest, MatmulMatchesHostReferenceAcrossFloatingDTypesAndEmptyDimensions) {
  for (DType dtype : {DType::FLOAT32, DType::FLOAT16, DType::BFLOAT16}) {
    Tensor lhs = FloatingTensorFromValues(GetContext(), Shape{2, 3}, dtype, {1, 2, 3, 4, 5, 6});
    Tensor rhs = FloatingTensorFromValues(GetContext(), Shape{3, 2}, dtype, {7, 8, 9, 10, 11, 12});
    Tensor output = Matmul(GetContext(), lhs, rhs, {.allow_tf32_ = false});
    ExpectShape(output, {2, 2});
    ExpectFloatValues(GetContext(), output, {58, 64, 139, 154}, dtype == DType::FLOAT32 ? 1.0e-4F : 0.5F,
                      dtype == DType::FLOAT32 ? 1.0e-5F : 0.01F);
  }

  Tensor empty_lhs = Empty(GetContext(), Shape{0, 3}, DType::FLOAT32);
  Tensor rhs = Empty(GetContext(), Shape{3, 4}, DType::FLOAT32);
  Tensor empty_output = Matmul(GetContext(), empty_lhs, rhs);
  EXPECT_EQ(empty_output.GetShape(), Shape({0, 4}));

  EXPECT_THROW(static_cast<void>(Matmul(GetContext(), Empty(GetContext(), Shape{2, 3}, DType::FLOAT32),
                                        Empty(GetContext(), Shape{4, 2}, DType::FLOAT32))),
               InvalidArgumentError);
}

TEST_F(LinalgNormalizationTest, BatchedMatmulBroadcastsLeadingDimensions) {
  Tensor lhs = FloatingTensorFromValues(GetContext(), Shape{2, 2, 2}, DType::FLOAT32, {1, 2, 3, 4, 5, 6, 7, 8});
  Tensor rhs = FloatingTensorFromValues(GetContext(), Shape{1, 2, 2}, DType::FLOAT32, {1, 0, 0, 2});
  Tensor output = BatchedMatmul(GetContext(), lhs, rhs, {.allow_tf32_ = false});
  ExpectShape(output, {2, 2, 2});
  ExpectFloatValues(GetContext(), output, {1, 4, 3, 8, 5, 12, 7, 16});

  EXPECT_THROW(static_cast<void>(BatchedMatmul(GetContext(), Empty(GetContext(), Shape{2, 2}, DType::FLOAT32),
                                               Empty(GetContext(), Shape{2, 2}, DType::FLOAT32))),
               InvalidArgumentError);
}

TEST_F(LinalgNormalizationTest, BatchedMatmulBroadcastsMultipleLeadingDimensions) {
  std::vector<float> lhs_values(1 * 3 * 2 * 4, 0.0F);
  std::vector<float> rhs_values(3 * 4 * 5, 0.0F);
  std::vector<float> expected;
  expected.reserve(1 * 3 * 2 * 5);
  for (size_t batch = 0; batch < 3; ++batch) {
    for (size_t row = 0; row < 2; ++row) {
      lhs_values[(((batch * 2) + row) * 4) + row] = 1.0F;
    }
    for (size_t reduction = 0; reduction < 4; ++reduction) {
      for (size_t column = 0; column < 5; ++column) {
        rhs_values[(((batch * 4) + reduction) * 5) + column] =
            static_cast<float>((batch * 100) + (reduction * 10) + column);
      }
    }
    for (size_t row = 0; row < 2; ++row) {
      for (size_t column = 0; column < 5; ++column) {
        expected.push_back(static_cast<float>((batch * 100) + (row * 10) + column));
      }
    }
  }

  Tensor lhs = FloatingTensorFromValues(GetContext(), Shape{1, 3, 2, 4}, DType::FLOAT32, lhs_values);
  Tensor rhs = FloatingTensorFromValues(GetContext(), Shape{3, 4, 5}, DType::FLOAT32, rhs_values);
  Tensor output = BatchedMatmul(GetContext(), lhs, rhs, {.allow_tf32_ = false});
  ExpectShape(output, {1, 3, 2, 5});
  ExpectFloatValues(GetContext(), output, expected);
}

TEST_F(LinalgNormalizationTest, LinearAppliesWeightBiasAndActivation) {
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{2, 3}, DType::FLOAT32, {1, 2, 3, -1, -2, -3});
  Tensor weight = FloatingTensorFromValues(GetContext(), Shape{2, 3}, DType::FLOAT32, {1, 0, 1, 0, 2, 0});
  Tensor bias = FloatingTensorFromValues(GetContext(), Shape{2}, DType::FLOAT32, {-5, 1});

  Tensor plain = Linear(GetContext(), input, weight, bias,
                        {.activation_ = LinearActivation::NONE, .matmul_ = {.allow_tf32_ = false}});
  ExpectFloatValues(GetContext(), plain, {-1, 5, -9, -3});

  Tensor relu = Linear(GetContext(), input, weight, bias,
                       {.activation_ = LinearActivation::RELU, .matmul_ = {.allow_tf32_ = false}});
  ExpectFloatValues(GetContext(), relu, {0, 5, 0, 0});

  EXPECT_THROW(static_cast<void>(Linear(GetContext(), input, weight,
                                        FloatingTensorFromValues(GetContext(), Shape{3}, DType::FLOAT32, {1, 2, 3}))),
               InvalidArgumentError);
}

TEST_F(LinalgNormalizationTest, LayerNormAndRmsNormApplyTrailingAxesAndAffineParameters) {
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{2, 3}, DType::FLOAT32, {1, 2, 3, 2, 2, 2});
  Tensor weight = FloatingTensorFromValues(GetContext(), Shape{3}, DType::FLOAT32, {1, 2, 3});
  Tensor bias = FloatingTensorFromValues(GetContext(), Shape{3}, DType::FLOAT32, {0.5F, 0, -0.5F});
  const NormOptions options{.normalized_rank_ = 1, .epsilon_ = 1.0e-5F};

  Tensor layer = LayerNorm(GetContext(), input, weight, bias, options);
  ExpectFloatValues(GetContext(), layer, {-0.7247357F, 0, 3.174207F, 0.5F, 0, -0.5F}, 2.0e-4F, 2.0e-4F);

  Tensor rms_input = FloatingTensorFromValues(GetContext(), Shape{1, 2}, DType::FLOAT32, {1, 2});
  Tensor rms_weight = FloatingTensorFromValues(GetContext(), Shape{2}, DType::FLOAT32, {2, 0.5F});
  Tensor rms = RmsNorm(GetContext(), rms_input, rms_weight, {.normalized_rank_ = 1, .epsilon_ = 0});
  const float denominator = std::sqrt(2.5F);
  ExpectFloatValues(GetContext(), rms, {2.0F / denominator, 1.0F / denominator});

  EXPECT_THROW(static_cast<void>(LayerNorm(GetContext(), input, std::nullopt, std::nullopt,
                                           {.normalized_rank_ = 0, .epsilon_ = 1.0e-5F})),
               InvalidArgumentError);
  EXPECT_THROW(
      static_cast<void>(RmsNorm(GetContext(), input, std::nullopt, {.normalized_rank_ = 1, .epsilon_ = -1.0F})),
      InvalidArgumentError);
}

TEST_F(LinalgNormalizationTest, SoftmaxAndLogSoftmaxNormalizeSelectedAxesStably) {
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{2, 3}, DType::FLOAT32, {1, 2, 3, 1000, 1000, 1000});
  const SoftmaxOptions options{.axes_ = {-1}};
  Tensor softmax = Softmax(GetContext(), input, options);
  const float denominator = std::exp(-2.0F) + std::exp(-1.0F) + 1.0F;
  ExpectFloatValues(GetContext(), softmax,
                    {std::exp(-2.0F) / denominator, std::exp(-1.0F) / denominator, 1.0F / denominator, 1.0F / 3.0F,
                     1.0F / 3.0F, 1.0F / 3.0F},
                    2.0e-5F, 2.0e-5F);

  Tensor log_softmax = LogSoftmax(GetContext(), input, options);
  const auto probabilities = FloatingTensorToValues(GetContext(), softmax);
  std::array<float, 6> expected{};
  for (size_t index = 0; index < expected.size(); ++index) {
    expected[index] = std::log(probabilities[index]);
  }
  ExpectFloatValues(GetContext(), log_softmax, expected);

  EXPECT_THROW(static_cast<void>(Softmax(GetContext(), input, {.axes_ = {}})), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Softmax(GetContext(), input, {.axes_ = {0, -2}})), InvalidArgumentError);
}

TEST_F(LinalgNormalizationTest, SoftmaxIgnoresNegativeInfinityAndPreservesInvalidRowsAcrossReductionPaths) {
  const auto infinity = std::numeric_limits<float>::infinity();
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  for (const auto width : {8, 128, 65539}) {
    std::vector<float> values(static_cast<size_t>(width) * 4, -infinity);
    values[0] = 0.0F;
    values[static_cast<size_t>(width) - 1] = 0.0F;
    values[static_cast<size_t>(width) * 2] = nan;
    values[static_cast<size_t>(width) * 3] = infinity;
    for (const auto dtype : {DType::FLOAT32, DType::FLOAT16, DType::BFLOAT16}) {
      Tensor input = FloatingTensorFromValues(GetContext(), Shape{4, width}, dtype, values);
      const auto probabilities = FloatingTensorToValues(GetContext(), Softmax(GetContext(), input, {.axes_ = {1}}));
      const auto logs = FloatingTensorToValues(GetContext(), LogSoftmax(GetContext(), input, {.axes_ = {1}}));
      EXPECT_EQ(probabilities.front(), 0.5F);
      EXPECT_EQ(probabilities[static_cast<size_t>(width) - 1], 0.5F);
      EXPECT_NEAR(logs.front(), -std::log(2.0F), 0.002F);
      EXPECT_NEAR(logs[static_cast<size_t>(width) - 1], -std::log(2.0F), 0.002F);
      EXPECT_TRUE(std::all_of(probabilities.begin() + 1, probabilities.begin() + width - 1,
                              [](float value) { return value == 0.0F; }));
      EXPECT_TRUE(std::all_of(logs.begin() + 1, logs.begin() + width - 1,
                              [infinity](float value) { return value == -infinity; }));
      EXPECT_TRUE(std::all_of(probabilities.begin() + width, probabilities.end(),
                              [](float value) { return std::isnan(value); }));
      EXPECT_TRUE(std::all_of(logs.begin() + width, logs.end(), [](float value) { return std::isnan(value); }));
    }
  }
}

TEST_F(LinalgNormalizationTest, LargeSoftmaxNormalizesDisjointAxesIntoPermutedOutput) {
  constexpr int64_t width = 4097;
  const auto shape = Shape{3, 2, width};
  const auto options = SoftmaxOptions{.axes_ = {0, 2}};
  std::vector<float> values(static_cast<size_t>(3 * 2 * width));
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = (static_cast<float>(index % 17) * 0.125F) - 4.0F;
  }
  for (const auto dtype : {DType::FLOAT32, DType::FLOAT16, DType::BFLOAT16}) {
    Tensor input = FloatingTensorFromValues(GetContext(), shape, dtype, values);
    const auto decoded = FloatingTensorToValues(GetContext(), input);
    std::vector<float> probabilities(values.size());
    std::vector<float> logs(values.size());
    for (size_t group = 0; group < 2; ++group) {
      double total = 0.0;
      for (size_t plane = 0; plane < 3; ++plane) {
        for (size_t column = 0; column < static_cast<size_t>(width); ++column) {
          const auto index = (((plane * 2) + group) * width) + column;
          total += std::exp(static_cast<double>(decoded[index]));
        }
      }
      for (size_t plane = 0; plane < 3; ++plane) {
        for (size_t column = 0; column < static_cast<size_t>(width); ++column) {
          const auto index = (((plane * 2) + group) * width) + column;
          probabilities[index] = static_cast<float>(std::exp(static_cast<double>(decoded[index])) / total);
          logs[index] = static_cast<float>(static_cast<double>(decoded[index]) - std::log(total));
        }
      }
    }
    const auto strides = Strides{width, 3 * width, 1};
    Tensor output = EmptyStrided(GetContext(), shape, strides, dtype);
    SoftmaxOut(GetContext(), output, input, options);
    ExpectFloatValues(GetContext(), Contiguous(GetContext(), output), probabilities, 1.0e-6F, 0.01F);
    LogSoftmaxOut(GetContext(), output, input, options);
    ExpectFloatValues(GetContext(), Contiguous(GetContext(), output), logs, 0.04F, 0.005F);
  }
}

TEST_F(LinalgNormalizationTest, ScaledDotProductAttentionHandlesUnmaskedCausalAndBooleanMaskCases) {
  Tensor query = FloatingTensorFromValues(GetContext(), Shape{1, 1, 2, 2}, DType::FLOAT32, {1, 0, 0, 1});
  Tensor key = FloatingTensorFromValues(GetContext(), Shape{1, 1, 2, 2}, DType::FLOAT32, {1, 0, 0, 1});
  Tensor value = FloatingTensorFromValues(GetContext(), Shape{1, 1, 2, 1}, DType::FLOAT32, {10, 20});

  Tensor output = ScaledDotProductAttention(GetContext(), query, key, value, std::nullopt, {.scale_ = 1.0F});
  const float p = std::numbers::e_v<float> / (std::numbers::e_v<float> + 1.0F);
  ExpectFloatValues(GetContext(), output, {(10.0F * p) + (20.0F * (1.0F - p)), (10.0F * (1.0F - p)) + (20.0F * p)},
                    2.0e-4F, 2.0e-4F);

  Tensor causal =
      ScaledDotProductAttention(GetContext(), query, key, value, std::nullopt, {.scale_ = 1.0F, .causal_ = true});
  ExpectFloatValues(GetContext(), causal, {10.0F, (10.0F * (1.0F - p)) + (20.0F * p)}, 2.0e-4F, 2.0e-4F);

  Tensor mask = BoolTensorFromValues(GetContext(), Shape{2, 2}, {1, 0, 1, 1});
  Tensor masked = ScaledDotProductAttention(GetContext(), query, key, value, mask, {.scale_ = 1.0F});
  ExpectFloatValues(GetContext(), masked, {10.0F, (10.0F * (1.0F - p)) + (20.0F * p)}, 2.0e-4F, 2.0e-4F);

  Tensor bad_key = Empty(GetContext(), Shape{1, 1, 2, 3}, DType::FLOAT32);
  EXPECT_THROW(static_cast<void>(ScaledDotProductAttention(GetContext(), query, bad_key, value)), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl::test
