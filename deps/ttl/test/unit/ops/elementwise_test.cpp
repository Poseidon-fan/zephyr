#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/cast.hpp"
#include "ttl/ops/composition.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/tensor/layout.hpp"

namespace ttl {

class ElementwiseCreationTest : public test::SingleDeviceTest {};

TEST_F(ElementwiseCreationTest, BroadcastsBinaryOperandsAndSupportsStridedInputs) {
  auto &context = GetContext();
  auto lhs = test::Upload(context, Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto rhs = test::Upload(context, Shape{3}, std::vector<float>{10, 20, 30});
  auto sum = Add(context, lhs, rhs);
  EXPECT_EQ(test::Download<float>(context, sum), (std::vector<float>{11, 22, 33, 14, 25, 36}));

  auto transposed = Transpose(lhs, 0, 1);
  auto scaled = Multiply(context, transposed, Scalar{2.0});
  EXPECT_EQ(scaled.GetShape(), Shape({3, 2}));
  EXPECT_EQ(test::Download<float>(context, scaled), (std::vector<float>{2, 8, 4, 10, 6, 12}));
}

TEST_F(ElementwiseCreationTest, ComputesComparisonLogicalAndWhere) {
  auto &context = GetContext();
  auto input = test::Upload(context, Shape{4}, std::vector<int32_t>{-2, 0, 3, 5});
  auto positive = Greater(context, input, Scalar{int64_t{0}});
  EXPECT_EQ(test::Download<uint8_t>(context, positive), (std::vector<uint8_t>{0, 0, 1, 1}));

  auto small = Less(context, input, Scalar{int64_t{5}});
  auto selected = Where(context, LogicalAnd(context, positive, small), input,
                        Full(context, Shape{4}, Scalar{int64_t{-1}}, DType::INT32));
  EXPECT_EQ(test::Download<int32_t>(context, selected), (std::vector<int32_t>{-1, -1, 3, -1}));
}

TEST_F(ElementwiseCreationTest, AppliesSignedIntegerUnaryOperationsWithModuloSemantics) {
  auto &context = GetContext();
  const auto minimum = std::numeric_limits<int32_t>::min();
  auto input = test::Upload(context, Shape{5}, std::vector<int32_t>{minimum, -7, 0, 3, 9});

  EXPECT_EQ(test::Download<int32_t>(context, Negate(context, input)), (std::vector<int32_t>{minimum, 7, 0, -3, -9}));
  EXPECT_EQ(test::Download<int32_t>(context, Abs(context, input)), (std::vector<int32_t>{minimum, 7, 0, 3, 9}));
  EXPECT_EQ(test::Download<int32_t>(context, Relu(context, input)), (std::vector<int32_t>{0, 0, 0, 3, 9}));
  EXPECT_THROW(static_cast<void>(Exp(context, input)), NotSupportedError);
}

TEST_F(ElementwiseCreationTest, CreatesSequencesFillsAndCasts) {
  auto &context = GetContext();
  auto sequence = Arange(context, Scalar{int64_t{-2}}, Scalar{int64_t{5}}, Scalar{int64_t{2}}, DType::INT32);
  EXPECT_EQ(test::Download<int32_t>(context, sequence), (std::vector<int32_t>{-2, 0, 2, 4}));

  auto floating = Cast(context, sequence, DType::FLOAT32);
  EXPECT_EQ(test::Download<float>(context, floating), (std::vector<float>{-2, 0, 2, 4}));
  FillOut(context, floating, Scalar{3.5});
  EXPECT_EQ(test::Download<float>(context, floating), (std::vector<float>{3.5F, 3.5F, 3.5F, 3.5F}));
  EXPECT_THROW(static_cast<void>(Arange(context, Scalar{0.0}, Scalar{1.0}, Scalar{0.0}, DType::FLOAT32)),
               InvalidArgumentError);
}

TEST_F(ElementwiseCreationTest, ConcatenatesAndStacksWithoutAliasingInputs) {
  auto &context = GetContext();
  const std::array tensors{
      test::Upload(context, Shape{1, 2}, std::vector<int64_t>{1, 2}),
      test::Upload(context, Shape{1, 2}, std::vector<int64_t>{3, 4}),
  };
  auto concatenated = Concat(context, tensors, 0);
  EXPECT_EQ(concatenated.GetShape(), Shape({2, 2}));
  EXPECT_EQ(test::Download<int64_t>(context, concatenated), (std::vector<int64_t>{1, 2, 3, 4}));

  auto stacked = Stack(context, tensors, 1);
  EXPECT_EQ(stacked.GetShape(), Shape({1, 2, 2}));
  EXPECT_EQ(test::Download<int64_t>(context, stacked), (std::vector<int64_t>{1, 2, 3, 4}));
  EXPECT_EQ(ClassifyAlias(stacked, tensors[0]), AliasKind::DISJOINT);
}

TEST_F(ElementwiseCreationTest, RejectsShapeDTypeAndOutputAliasViolations) {
  auto &context = GetContext();
  auto lhs = Empty(context, Shape{2, 3}, DType::FLOAT32);
  auto wrong_shape = Empty(context, Shape{2, 2}, DType::FLOAT32);
  auto wrong_dtype = Empty(context, Shape{2, 3}, DType::INT32);
  EXPECT_THROW(static_cast<void>(Add(context, lhs, wrong_shape)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Add(context, lhs, wrong_dtype)), InvalidArgumentError);
  EXPECT_THROW(AddOut(context, lhs, lhs, lhs), InvalidArgumentError);
}

}  // namespace ttl
