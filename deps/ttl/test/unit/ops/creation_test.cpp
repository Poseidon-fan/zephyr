#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <span>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/cast.hpp"
#include "ttl/ops/composition.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/tensor/layout.hpp"

namespace ttl::test {
namespace {

class CreationOperatorTest : public SingleDeviceTest {};

TEST_F(CreationOperatorTest, FullZerosOnesAndFillCoverEveryDTypeAndEmptyTensor) {
  for (DType dtype : {DType::FLOAT32, DType::FLOAT16, DType::BFLOAT16}) {
    Tensor full = Full(GetContext(), Shape{2, 2}, Scalar{2.5}, dtype);
    ExpectFloatValues(GetContext(), full, {2.5F, 2.5F, 2.5F, 2.5F}, 0.01F, 0.01F);
    Tensor zero = Zeros(GetContext(), Shape{0, 3}, dtype);
    EXPECT_TRUE(zero.GetShape().IsEmpty());
    EXPECT_NO_THROW(FillOut(GetContext(), zero, Scalar{1.0}));
  }

  Tensor int32_ones = Ones(GetContext(), Shape{3}, DType::INT32);
  ExpectValues<int32_t>(GetContext(), int32_ones, {1, 1, 1});
  FillOut(GetContext(), int32_ones, Scalar{int64_t{-7}});
  ExpectValues<int32_t>(GetContext(), int32_ones, {-7, -7, -7});

  Tensor uint8_full = Full(GetContext(), Shape{2}, Scalar{int64_t{255}}, DType::UINT8);
  ExpectValues<uint8_t>(GetContext(), uint8_full, {255, 255});
  Tensor bool_full = Full(GetContext(), Shape{3}, Scalar{true}, DType::BOOL);
  ExpectBoolValues(GetContext(), bool_full, {1, 1, 1});
  EXPECT_THROW(static_cast<void>(Full(GetContext(), Shape{1}, Scalar{int64_t{256}}, DType::UINT8)), OverflowError);
}

TEST_F(CreationOperatorTest, ArangeHandlesPositiveNegativeFloatingEmptyAndInvalidRanges) {
  Tensor positive = Arange(GetContext(), Scalar{int64_t{-2}}, Scalar{int64_t{5}}, Scalar{int64_t{2}}, DType::INT32);
  ExpectValues<int32_t>(GetContext(), positive, {-2, 0, 2, 4});

  Tensor negative = Arange(GetContext(), Scalar{int64_t{5}}, Scalar{int64_t{-2}}, Scalar{int64_t{-3}}, DType::INT64);
  ExpectValues<int64_t>(GetContext(), negative, {5, 2, -1});

  Tensor floating = Arange(GetContext(), Scalar{0.5}, Scalar{2.0}, Scalar{0.5}, DType::FLOAT32);
  ExpectFloatValues(GetContext(), floating, {0.5F, 1.0F, 1.5F});

  Tensor empty = Arange(GetContext(), Scalar{int64_t{4}}, Scalar{int64_t{4}}, Scalar{int64_t{1}}, DType::INT32);
  EXPECT_EQ(empty.GetNumElements(), 0);
  EXPECT_THROW(static_cast<void>(Arange(GetContext(), Scalar{0.0}, Scalar{1.0}, Scalar{0.0}, DType::FLOAT32)),
               InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(
                   Arange(GetContext(), Scalar{int64_t{0}}, Scalar{int64_t{1}}, Scalar{int64_t{1}}, DType::FLOAT16)),
               InvalidArgumentError);
}

TEST_F(CreationOperatorTest, CastConvertsSupportedValuesAndReportsOutOfRangeValuesAsynchronously) {
  Tensor integers = TensorFromValues<int32_t>(GetContext(), Shape{4}, {-2, 0, 3, 9});
  Tensor floating = Cast(GetContext(), integers, DType::FLOAT32);
  ExpectFloatValues(GetContext(), floating, {-2, 0, 3, 9});
  Tensor round_trip = Cast(GetContext(), floating, DType::INT64);
  ExpectValues<int64_t>(GetContext(), round_trip, {-2, 0, 3, 9});

  Tensor invalid = FloatingTensorFromValues(GetContext(), Shape{2}, DType::FLOAT32, {1.5F, 300.0F});
  Tensor output = Cast(GetContext(), invalid, DType::UINT8);
  static_cast<void>(output);
  EXPECT_THROW(GetContext().CheckAsyncErrors(), DeviceError);
  EXPECT_NO_THROW(GetContext().Synchronize());
}

TEST_F(CreationOperatorTest, ConcatAndStackSupportNegativeAxesEmptyExtentsAndRejectMismatches) {
  const std::array<Tensor, 2> inputs{
      TensorFromValues<int32_t>(GetContext(), Shape{2, 1}, {1, 2}),
      TensorFromValues<int32_t>(GetContext(), Shape{2, 2}, {3, 4, 5, 6}),
  };
  Tensor concat = Concat(GetContext(), inputs, -1);
  ExpectShape(concat, {2, 3});
  ExpectValues<int32_t>(GetContext(), concat, {1, 3, 4, 2, 5, 6});

  const std::array<Tensor, 2> stack_inputs{
      TensorFromValues<int32_t>(GetContext(), Shape{2}, {1, 2}),
      TensorFromValues<int32_t>(GetContext(), Shape{2}, {3, 4}),
  };
  Tensor stack = Stack(GetContext(), stack_inputs, 1);
  ExpectShape(stack, {2, 2});
  ExpectValues<int32_t>(GetContext(), stack, {1, 3, 2, 4});

  const std::array<Tensor, 2> empty_inputs{
      Empty(GetContext(), Shape{2, 0}, DType::INT32),
      Empty(GetContext(), Shape{2, 0}, DType::INT32),
  };
  EXPECT_EQ(Concat(GetContext(), empty_inputs, 1).GetShape(), Shape({2, 0}));
  EXPECT_THROW(static_cast<void>(Concat(GetContext(), std::span<const Tensor>{}, 0)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Stack(GetContext(), inputs, 0)), InvalidArgumentError);
}

TEST_F(CreationOperatorTest, ArithmeticBroadcastsSupportsScalarsAndAllowsExactOutputAlias) {
  Tensor matrix = TensorFromValues<int32_t>(GetContext(), Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor row = TensorFromValues<int32_t>(GetContext(), Shape{3}, {10, 20, 30});

  ExpectValues<int32_t>(GetContext(), Add(GetContext(), matrix, row), {11, 22, 33, 14, 25, 36});
  ExpectValues<int32_t>(GetContext(), Subtract(GetContext(), matrix, Scalar{int64_t{1}}), {0, 1, 2, 3, 4, 5});
  ExpectValues<int32_t>(GetContext(), Multiply(GetContext(), matrix, Scalar{int64_t{-2}}), {-2, -4, -6, -8, -10, -12});
  ExpectValues<int32_t>(GetContext(), Divide(GetContext(), matrix, Scalar{int64_t{2}}), {0, 1, 1, 2, 2, 3});
  ExpectValues<int32_t>(GetContext(), Maximum(GetContext(), matrix, Scalar{int64_t{4}}), {4, 4, 4, 4, 5, 6});
  ExpectValues<int32_t>(GetContext(), Minimum(GetContext(), matrix, Scalar{int64_t{4}}), {1, 2, 3, 4, 4, 4});

  AddOut(GetContext(), matrix, matrix, row);
  ExpectValues<int32_t>(GetContext(), matrix, {11, 22, 33, 14, 25, 36});

  Tensor bad_shape = TensorFromValues<int32_t>(GetContext(), Shape{2}, {1, 2});
  EXPECT_THROW(static_cast<void>(Add(GetContext(), matrix, bad_shape)), InvalidArgumentError);
}

TEST_F(CreationOperatorTest, IntegerDivisionByZeroUsesExplicitAsynchronousErrorBoundary) {
  Tensor dividend = TensorFromValues<int32_t>(GetContext(), Shape{3}, {8, 9, 10});
  Tensor divisor = TensorFromValues<int32_t>(GetContext(), Shape{3}, {2, 0, 5});
  Tensor output = Divide(GetContext(), dividend, divisor);
  static_cast<void>(output);
  EXPECT_THROW(GetContext().CheckAsyncErrors(), DeviceError);
  EXPECT_NO_THROW(GetContext().Synchronize());
}

TEST_F(CreationOperatorTest, ComparisonsCoverTensorScalarBroadcastAndNanRules) {
  Tensor lhs = FloatingTensorFromValues(GetContext(), Shape{2, 2}, DType::FLOAT32,
                                        {1.0F, 2.0F, std::numeric_limits<float>::quiet_NaN(), 4.0F});
  Tensor rhs = FloatingTensorFromValues(GetContext(), Shape{2}, DType::FLOAT32, {2.0F, 2.0F});

  ExpectBoolValues(GetContext(), Equal(GetContext(), lhs, rhs), {0, 1, 0, 0});
  ExpectBoolValues(GetContext(), NotEqual(GetContext(), lhs, rhs), {1, 0, 1, 1});
  ExpectBoolValues(GetContext(), Less(GetContext(), lhs, rhs), {1, 0, 0, 0});
  ExpectBoolValues(GetContext(), LessEqual(GetContext(), lhs, Scalar{2.0}), {1, 1, 0, 0});
  ExpectBoolValues(GetContext(), Greater(GetContext(), lhs, rhs), {0, 0, 0, 1});
  ExpectBoolValues(GetContext(), GreaterEqual(GetContext(), lhs, Scalar{2.0}), {0, 1, 0, 1});
}

TEST_F(CreationOperatorTest, UnaryFloatingFunctionsMatchIndependentHostReferences) {
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{3}, DType::FLOAT32, {-1.0F, 0.0F, 1.0F});
  ExpectFloatValues(GetContext(), Negate(GetContext(), input), {1, 0, -1});
  ExpectFloatValues(GetContext(), Abs(GetContext(), input), {1, 0, 1});
  const float e = std::numbers::e_v<float>;
  ExpectFloatValues(GetContext(), Exp(GetContext(), input), {1.0F / e, 1.0F, e});
  ExpectFloatValues(GetContext(), Sin(GetContext(), input), {std::sin(-1.0F), 0.0F, std::sin(1.0F)});
  ExpectFloatValues(GetContext(), Cos(GetContext(), input), {std::cos(-1.0F), 1.0F, std::cos(1.0F)});
  ExpectFloatValues(GetContext(), Tanh(GetContext(), input), {std::tanh(-1.0F), 0.0F, std::tanh(1.0F)});
  ExpectFloatValues(GetContext(), Sigmoid(GetContext(), input), {1.0F / (1.0F + e), 0.5F, 1.0F / (1.0F + (1.0F / e))});
  ExpectFloatValues(GetContext(), Relu(GetContext(), input), {0, 0, 1});
  ExpectFloatValues(GetContext(), Silu(GetContext(), input), {-1.0F / (1.0F + e), 0, 1.0F / (1.0F + (1.0F / e))});

  Tensor positive = FloatingTensorFromValues(GetContext(), Shape{3}, DType::FLOAT32, {0.25F, 1.0F, 4.0F});
  ExpectFloatValues(GetContext(), Log(GetContext(), positive), {std::log(0.25F), 0, std::log(4.0F)});
  ExpectFloatValues(GetContext(), Sqrt(GetContext(), positive), {0.5F, 1, 2});
  ExpectFloatValues(GetContext(), Rsqrt(GetContext(), positive), {2, 1, 0.5F});

  Tensor clamp = Clamp(GetContext(), input, Scalar{-0.5}, Scalar{0.5});
  ExpectFloatValues(GetContext(), clamp, {-0.5F, 0, 0.5F});
  EXPECT_THROW(static_cast<void>(Clamp(GetContext(), input, std::nullopt, std::nullopt)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Clamp(GetContext(), input, Scalar{2.0}, Scalar{1.0})), InvalidArgumentError);
}

TEST_F(CreationOperatorTest, LogicalWhereAndGeluCoverBooleanBroadcastAndBothApproximations) {
  Tensor lhs = BoolTensorFromValues(GetContext(), Shape{2, 2}, {1, 0, 1, 0});
  Tensor rhs = BoolTensorFromValues(GetContext(), Shape{2}, {1, 1});
  ExpectBoolValues(GetContext(), LogicalAnd(GetContext(), lhs, rhs), {1, 0, 1, 0});
  ExpectBoolValues(GetContext(), LogicalOr(GetContext(), lhs, LogicalNot(GetContext(), rhs)), {1, 0, 1, 0});
  ExpectBoolValues(GetContext(), LogicalNot(GetContext(), lhs), {0, 1, 0, 1});

  Tensor true_values = TensorFromValues<int32_t>(GetContext(), Shape{2, 1}, {10, 20});
  Tensor false_values = TensorFromValues<int32_t>(GetContext(), Shape{2}, {1, 2});
  Tensor selected = Where(GetContext(), lhs, true_values, false_values);
  ExpectValues<int32_t>(GetContext(), selected, {10, 2, 20, 2});

  Tensor input = FloatingTensorFromValues(GetContext(), Shape{3}, DType::FLOAT32, {-1, 0, 1});
  Tensor exact = Gelu(GetContext(), input, GeluApproximation::NONE);
  Tensor approximate = Gelu(GetContext(), input, GeluApproximation::TANH);
  ExpectFloatValues(GetContext(), exact, {-0.15865526F, 0, 0.8413447F}, 2.0e-5F, 2.0e-5F);
  ExpectFloatValues(GetContext(), approximate, {-0.158808F, 0, 0.841192F}, 2.0e-5F, 2.0e-5F);
}

}  // namespace
}  // namespace ttl::test
