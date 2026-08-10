#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/attention.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/ops/normalization.hpp"
#include "ttl/ops/random.hpp"
#include "ttl/ops/reduction.hpp"
#include "ttl/ops/scan.hpp"
#include "ttl/ops/softmax.hpp"
#include "ttl/runtime/generator.hpp"

namespace ttl::test {
namespace {

class DTypeMatrixTest : public SingleDeviceTest {};

[[nodiscard]] auto FloatingTolerance(DType dtype) -> float { return dtype == DType::FLOAT32 ? 2.0e-5F : 0.03F; }

TEST_F(DTypeMatrixTest, FloatingElementwiseMatrixCoversEveryArithmeticAndRepresentativeUnaryOperation) {
  for (DType dtype : {DType::FLOAT32, DType::FLOAT16, DType::BFLOAT16}) {
    const float tolerance = FloatingTolerance(dtype);
    Tensor lhs = FloatingTensorFromValues(GetContext(), Shape{4}, dtype, {-2, -1, 2, 4});
    Tensor rhs = FloatingTensorFromValues(GetContext(), Shape{4}, dtype, {1, 2, 4, 2});
    ExpectFloatValues(GetContext(), Add(GetContext(), lhs, rhs), {-1, 1, 6, 6}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Subtract(GetContext(), lhs, rhs), {-3, -3, -2, 2}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Multiply(GetContext(), lhs, rhs), {-2, -2, 8, 8}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Divide(GetContext(), lhs, rhs), {-2, -0.5F, 0.5F, 2}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Minimum(GetContext(), lhs, rhs), {-2, -1, 2, 2}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Maximum(GetContext(), lhs, rhs), {1, 2, 4, 4}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Negate(GetContext(), lhs), {2, 1, -2, -4}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Abs(GetContext(), lhs), {2, 1, 2, 4}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Relu(GetContext(), lhs), {0, 0, 2, 4}, tolerance, tolerance);

    Tensor positive = FloatingTensorFromValues(GetContext(), Shape{3}, dtype, {0.25F, 1.0F, 4.0F});
    ExpectFloatValues(GetContext(), Sqrt(GetContext(), positive), {0.5F, 1.0F, 2.0F}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Exp(GetContext(), FloatingTensorFromValues(GetContext(), Shape{2}, dtype, {0, 1})),
                      {1.0F, std::numbers::e_v<float>}, tolerance * 2, tolerance * 2);
    ExpectFloatValues(GetContext(), Log(GetContext(), positive), {std::log(0.25F), 0.0F, std::log(4.0F)}, tolerance * 2,
                      tolerance * 2);
  }
}

TEST_F(DTypeMatrixTest, FloatingReductionSoftmaxAndNormalizationMatrixCoversEveryFloatingDtype) {
  for (DType dtype : {DType::FLOAT32, DType::FLOAT16, DType::BFLOAT16}) {
    const float tolerance = FloatingTolerance(dtype);
    Tensor input = FloatingTensorFromValues(GetContext(), Shape{2, 3}, dtype, {1, 2, 3, 2, 4, 6});
    const ReductionOptions rows{.axes_ = {1}};
    ExpectFloatValues(GetContext(), Sum(GetContext(), input, rows), {6, 12}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Mean(GetContext(), input, rows), {2, 4}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Minimum(GetContext(), input, rows), {1, 2}, tolerance, tolerance);
    ExpectFloatValues(GetContext(), Maximum(GetContext(), input, rows), {3, 6}, tolerance, tolerance);
    ExpectValues<int64_t>(GetContext(), ArgMin(GetContext(), input, 1), {0, 0});
    ExpectValues<int64_t>(GetContext(), ArgMax(GetContext(), input, 1), {2, 2});

    const Tensor softmax = Softmax(GetContext(), input, {.axes_ = {1}});
    const auto probabilities = FloatingTensorToValues(GetContext(), softmax);
    ASSERT_EQ(probabilities.size(), 6U);
    EXPECT_NEAR(probabilities[0] + probabilities[1] + probabilities[2], 1.0F, tolerance * 2);
    EXPECT_NEAR(probabilities[3] + probabilities[4] + probabilities[5], 1.0F, tolerance * 2);
    const auto log_probabilities =
        FloatingTensorToValues(GetContext(), LogSoftmax(GetContext(), input, {.axes_ = {1}}));
    for (size_t index = 0; index < probabilities.size(); ++index) {
      EXPECT_NEAR(log_probabilities[index], std::log(probabilities[index]), tolerance * 3);
    }

    const NormOptions norm_options{.normalized_rank_ = 1, .epsilon_ = 1.0e-5F};
    const auto layer =
        FloatingTensorToValues(GetContext(), LayerNorm(GetContext(), input, std::nullopt, std::nullopt, norm_options));
    ASSERT_EQ(layer.size(), 6U);
    EXPECT_NEAR(layer[0] + layer[1] + layer[2], 0.0F, tolerance * 3);
    EXPECT_NEAR(layer[3] + layer[4] + layer[5], 0.0F, tolerance * 3);
    const auto rms = FloatingTensorToValues(GetContext(), RmsNorm(GetContext(), input, std::nullopt, norm_options));
    ASSERT_EQ(rms.size(), 6U);
    EXPECT_TRUE(std::ranges::all_of(rms, [](float value) { return std::isfinite(value) && value > 0.0F; }));
  }
}

TEST_F(DTypeMatrixTest, FloatingRandomAndAttentionMatrixCoversEveryFloatingDtype) {
  for (DType dtype : {DType::FLOAT32, DType::FLOAT16, DType::BFLOAT16}) {
    Generator generator{GetContext(), 777};
    const auto uniform = FloatingTensorToValues(
        GetContext(), Uniform(GetContext(), Shape{257}, dtype, generator, {.low_ = -1.0, .high_ = 2.0}));
    EXPECT_TRUE(std::ranges::all_of(uniform, [](float value) { return value >= -1.0F && value < 2.0F; }));
    const auto normal = FloatingTensorToValues(
        GetContext(), Normal(GetContext(), Shape{257}, dtype, generator, {.mean_ = 5.0, .standard_deviation_ = 0.0}));
    EXPECT_TRUE(std::ranges::all_of(normal, [](float value) { return value == 5.0F; }));

    Tensor query = FloatingTensorFromValues(GetContext(), Shape{1, 1, 1, 2}, dtype, {0, 0});
    Tensor key = FloatingTensorFromValues(GetContext(), Shape{1, 1, 2, 2}, dtype, {1, 0, 0, 1});
    Tensor value = FloatingTensorFromValues(GetContext(), Shape{1, 1, 2, 1}, dtype, {10, 20});
    ExpectFloatValues(GetContext(), ScaledDotProductAttention(GetContext(), query, key, value), {15.0F},
                      FloatingTolerance(dtype), FloatingTolerance(dtype));
  }
}

TEST_F(DTypeMatrixTest, SignedIntegerMatrixCoversArithmeticReductionAndScan) {
  const auto verify = [&]<TensorStorageType T>() {
    Tensor lhs = TensorFromValues<T>(GetContext(), Shape{2, 3}, {1, -2, 3, 4, 5, -6});
    Tensor rhs = TensorFromValues<T>(GetContext(), Shape{2, 3}, {2, 2, -1, 2, -5, 3});
    ExpectValues<T>(GetContext(), Add(GetContext(), lhs, rhs), {3, 0, 2, 6, 0, -3});
    ExpectValues<T>(GetContext(), Subtract(GetContext(), lhs, rhs), {-1, -4, 4, 2, 10, -9});
    ExpectValues<T>(GetContext(), Multiply(GetContext(), lhs, rhs), {2, -4, -3, 8, -25, -18});
    ExpectValues<T>(GetContext(), Divide(GetContext(), lhs, rhs), {0, -1, -3, 2, -1, -2});
    ExpectValues<T>(GetContext(), Sum(GetContext(), lhs, {.axes_ = {1}}), {2, 3});
    ExpectValues<T>(GetContext(), Minimum(GetContext(), lhs, {.axes_ = {1}}), {-2, -6});
    ExpectValues<T>(GetContext(), Maximum(GetContext(), lhs, {.axes_ = {1}}), {3, 5});
    ExpectValues<T>(GetContext(), CumulativeSum(GetContext(), lhs, 1), {1, -1, 2, 4, 9, 3});
  };
  verify.template operator()<int32_t>();
  verify.template operator()<int64_t>();
}

TEST_F(DTypeMatrixTest, ControlDtypesAcceptOnlyDocumentedOperations) {
  Tensor bytes = TensorFromValues<uint8_t>(GetContext(), Shape{3}, {250, 10, 1});
  ExpectValues<uint8_t>(GetContext(), CumulativeSum(GetContext(), bytes, 0), {250, 4, 5});
  EXPECT_THROW(static_cast<void>(Add(GetContext(), bytes, bytes)), NotSupportedError);
  EXPECT_THROW(static_cast<void>(Mean(GetContext(), bytes)), NotSupportedError);

  Tensor boolean = BoolTensorFromValues(GetContext(), Shape{3}, {0, 1, 1});
  ExpectBoolValues(GetContext(), LogicalNot(GetContext(), boolean), {1, 0, 0});
  ExpectBoolValues(GetContext(), Any(GetContext(), boolean), {1});
  ExpectBoolValues(GetContext(), All(GetContext(), boolean), {0});
  EXPECT_THROW(static_cast<void>(Add(GetContext(), boolean, boolean)), NotSupportedError);
  EXPECT_THROW(static_cast<void>(Softmax(GetContext(), boolean, {.axes_ = {0}})), NotSupportedError);
}

}  // namespace
}  // namespace ttl::test
