#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>

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
