#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/indexing.hpp"
#include "ttl/ops/reduction.hpp"
#include "ttl/ops/scan.hpp"
#include "ttl/tensor/layout.hpp"

namespace ttl::test {
namespace {

class ReductionIndexingTest : public SingleDeviceTest {};

TEST_F(ReductionIndexingTest, SumMeanMinimumAndMaximumSupportAxesKeepDimensionsAndNonContiguousInput) {
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{2, 3}, DType::FLOAT32, {1, 2, 3, 4, 5, 6});
  const ReductionOptions columns{.axes_ = {1}, .keep_dimensions_ = false};
  ExpectFloatValues(GetContext(), Sum(GetContext(), input, columns), {6, 15});
  ExpectFloatValues(GetContext(), Mean(GetContext(), input, columns), {2, 5});
  ExpectFloatValues(GetContext(), Minimum(GetContext(), input, columns), {1, 4});
  ExpectFloatValues(GetContext(), Maximum(GetContext(), input, columns), {3, 6});

  const ReductionOptions rows_keep{.axes_ = {0}, .keep_dimensions_ = true};
  Tensor row_sum = Sum(GetContext(), input, rows_keep);
  ExpectShape(row_sum, {1, 3});
  ExpectFloatValues(GetContext(), row_sum, {5, 7, 9});

  Tensor transposed = Transpose(input, 0, 1);
  const ReductionOptions last_axis{.axes_ = {-1}, .keep_dimensions_ = false};
  ExpectFloatValues(GetContext(), Sum(GetContext(), transposed, last_axis), {5, 7, 9});

  Tensor all = Sum(GetContext(), input);
  EXPECT_TRUE(all.GetShape().IsScalar());
  ExpectFloatValues(GetContext(), all, {21});

  EXPECT_THROW(static_cast<void>(Sum(GetContext(), input, {.axes_ = {0, -2}})), InvalidArgumentError);
}

TEST_F(ReductionIndexingTest, EmptyReductionUsesIdentitiesAndRejectsUndefinedMeanAndExtrema) {
  Tensor empty = Empty(GetContext(), Shape{2, 0, 3}, DType::FLOAT32);
  const ReductionOptions empty_axis{.axes_ = {1}, .keep_dimensions_ = false};
  Tensor sum = Sum(GetContext(), empty, empty_axis);
  ExpectShape(sum, {2, 3});
  ExpectFloatValues(GetContext(), sum, {0, 0, 0, 0, 0, 0});
  Tensor mean = Mean(GetContext(), empty, empty_axis);
  ExpectFloatValues(GetContext(), mean,
                    {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
                     std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
  EXPECT_THROW(static_cast<void>(Minimum(GetContext(), empty, empty_axis)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Maximum(GetContext(), empty, empty_axis)), InvalidArgumentError);
}

TEST_F(ReductionIndexingTest, ExtremaAndArgExtremaPropagateNanAndSelectLowestTieIndex) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{2, 4}, DType::FLOAT32, {3, 1, 1, 2, 5, nan, nan, 4});
  const ReductionOptions columns{.axes_ = {1}, .keep_dimensions_ = false};
  ExpectFloatValues(GetContext(), Minimum(GetContext(), input, columns), {1, nan});
  ExpectFloatValues(GetContext(), Maximum(GetContext(), input, columns), {3, nan});
  ExpectValues<int64_t>(GetContext(), ArgMin(GetContext(), input, 1), {1, 1});
  ExpectValues<int64_t>(GetContext(), ArgMax(GetContext(), input, 1), {0, 1});

  Tensor kept = ArgMax(GetContext(), input, -1, true);
  ExpectShape(kept, {2, 1});
  EXPECT_THROW(static_cast<void>(ArgMin(GetContext(), Empty(GetContext(), Shape{2, 0}, DType::FLOAT32), 1)),
               InvalidArgumentError);
}

TEST_F(ReductionIndexingTest, AnyAndAllUseBooleanIdentitiesAcrossSelectedAxes) {
  Tensor input = BoolTensorFromValues(GetContext(), Shape{2, 3}, {0, 0, 1, 1, 1, 1});
  const ReductionOptions columns{.axes_ = {1}, .keep_dimensions_ = false};
  ExpectBoolValues(GetContext(), Any(GetContext(), input, columns), {1, 1});
  ExpectBoolValues(GetContext(), All(GetContext(), input, columns), {0, 1});

  Tensor empty = Empty(GetContext(), Shape{2, 0}, DType::BOOL);
  ExpectBoolValues(GetContext(), Any(GetContext(), empty, columns), {0, 0});
  ExpectBoolValues(GetContext(), All(GetContext(), empty, columns), {1, 1});
  EXPECT_THROW(static_cast<void>(Any(GetContext(), TensorFromValues<int32_t>(GetContext(), Shape{1}, {1}))),
               NotSupportedError);
}

TEST_F(ReductionIndexingTest, CumulativeSumSupportsNegativeAxisIntegerAndFloatingInputs) {
  Tensor input = TensorFromValues<int32_t>(GetContext(), Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor result = CumulativeSum(GetContext(), input, -1);
  ExpectValues<int32_t>(GetContext(), result, {1, 3, 6, 4, 9, 15});

  Tensor columns = CumulativeSum(GetContext(), input, 0);
  ExpectValues<int32_t>(GetContext(), columns, {1, 2, 3, 5, 7, 9});
  EXPECT_THROW(static_cast<void>(CumulativeSum(GetContext(), input, 2)), InvalidArgumentError);
}

TEST_F(ReductionIndexingTest, IndexSelectGatherAndTakeAlongDimensionProduceIndependentCanonicalOutputs) {
  Tensor input = TensorFromValues<int32_t>(GetContext(), Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor selected_indices = TensorFromValues<int64_t>(GetContext(), Shape{2}, {2, 0});
  Tensor selected = IndexSelect(GetContext(), input, -1, selected_indices);
  ExpectShape(selected, {2, 2});
  ExpectValues<int32_t>(GetContext(), selected, {3, 1, 6, 4});

  Tensor gather_indices = TensorFromValues<int32_t>(GetContext(), Shape{2, 2}, {2, 0, 1, 1});
  Tensor gathered = Gather(GetContext(), input, 1, gather_indices);
  ExpectValues<int32_t>(GetContext(), gathered, {3, 1, 5, 5});

  Tensor broadcast_indices = TensorFromValues<int64_t>(GetContext(), Shape{1, 2}, {2, 0});
  Tensor taken = TakeAlongDimension(GetContext(), input, broadcast_indices, 1);
  ExpectShape(taken, {2, 2});
  ExpectValues<int32_t>(GetContext(), taken, {3, 1, 6, 4});
}

TEST_F(ReductionIndexingTest, ScatterElementsCopiesInputAndOverwritesUniqueDestinations) {
  Tensor input = Zeros(GetContext(), Shape{2, 3}, DType::INT32);
  Tensor indices = TensorFromValues<int64_t>(GetContext(), Shape{2, 2}, {2, 0, 1, 2});
  Tensor source = TensorFromValues<int32_t>(GetContext(), Shape{2, 2}, {7, 8, 9, 10});
  Tensor output = ScatterElements(GetContext(), input, 1, indices, source);
  ExpectValues<int32_t>(GetContext(), output, {8, 0, 7, 0, 9, 10});

  ScatterElementsOut(GetContext(), input, input, 1, indices, source);
  ExpectValues<int32_t>(GetContext(), input, {8, 0, 7, 0, 9, 10});
}

TEST_F(ReductionIndexingTest, GatherRowsAndEmbeddingSupportArbitraryIndexShape) {
  Tensor table = TensorFromValues<int32_t>(GetContext(), Shape{3, 2}, {10, 11, 20, 21, 30, 31});
  Tensor indices = TensorFromValues<int32_t>(GetContext(), Shape{2, 2}, {2, 0, 1, 2});

  Tensor gathered = GatherRows(GetContext(), table, indices);
  ExpectShape(gathered, {2, 2, 2});
  ExpectValues<int32_t>(GetContext(), gathered, {30, 31, 10, 11, 20, 21, 30, 31});

  Tensor embedded = Embedding(GetContext(), table, indices);
  ExpectShape(embedded, {2, 2, 2});
  ExpectValues<int32_t>(GetContext(), embedded, {30, 31, 10, 11, 20, 21, 30, 31});
}

TEST_F(ReductionIndexingTest, OutOfBoundsIndexIsReportedAsynchronouslyAndShapeErrorsAreSynchronous) {
  Tensor input = TensorFromValues<int32_t>(GetContext(), Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor invalid = TensorFromValues<int64_t>(GetContext(), Shape{1}, {3});
  Tensor output = IndexSelect(GetContext(), input, 1, invalid);
  static_cast<void>(output);
  EXPECT_THROW(GetContext().CheckAsyncErrors(), DeviceError);
  EXPECT_NO_THROW(GetContext().Synchronize());

  Tensor wrong_dtype = FloatingTensorFromValues(GetContext(), Shape{1}, DType::FLOAT32, {0});
  EXPECT_THROW(static_cast<void>(IndexSelect(GetContext(), input, 1, wrong_dtype)), NotSupportedError);
  EXPECT_THROW(
      static_cast<void>(Gather(GetContext(), input, 1, TensorFromValues<int64_t>(GetContext(), Shape{1}, {0}))),
      InvalidArgumentError);
}

}  // namespace
}  // namespace ttl::test
