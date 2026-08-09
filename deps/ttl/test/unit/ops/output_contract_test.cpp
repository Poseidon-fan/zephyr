#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/attention.hpp"
#include "ttl/ops/cast.hpp"
#include "ttl/ops/composition.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/ops/indexing.hpp"
#include "ttl/ops/matmul.hpp"
#include "ttl/ops/normalization.hpp"
#include "ttl/ops/random.hpp"
#include "ttl/ops/reduction.hpp"
#include "ttl/ops/scan.hpp"
#include "ttl/ops/softmax.hpp"
#include "ttl/ops/topk.hpp"
#include "ttl/runtime/generator.hpp"
#include "ttl/tensor/layout.hpp"

namespace ttl::test {
namespace {

template <typename Function>
auto Transform(std::span<const float> values, Function &&function) -> std::vector<float> {
  std::vector<float> result;
  result.reserve(values.size());
  for (float value : values) {
    result.push_back(std::invoke(function, value));
  }
  return result;
}

class OutputContractTest : public SingleDeviceTest {};

TEST_F(OutputContractTest, CreationCopyCastAndCompositionOutFormsPreserveIndependentLogicalValues) {
  Tensor input = TensorFromValues<int32_t>(GetContext(), Shape{2, 2}, {1, 2, 3, 4});
  Tensor empty_like = EmptyLike(GetContext(), input);
  EXPECT_EQ(empty_like.GetShape(), input.GetShape());
  EXPECT_EQ(empty_like.GetDType(), input.GetDType());

  Tensor clone = Clone(GetContext(), input);
  ExpectValues<int32_t>(GetContext(), clone, {1, 2, 3, 4});

  Tensor transposed = Transpose(input, 0, 1);
  Tensor contiguous = Empty(GetContext(), Shape{2, 2}, DType::INT32);
  ContiguousOut(GetContext(), contiguous, transposed);
  ExpectValues<int32_t>(GetContext(), contiguous, {1, 3, 2, 4});

  Tensor cast = Empty(GetContext(), Shape{2, 2}, DType::FLOAT32);
  CastOut(GetContext(), cast, input);
  ExpectFloatValues(GetContext(), cast, {1, 2, 3, 4});

  const std::array inputs{TensorFromValues<int32_t>(GetContext(), Shape{2}, {1, 2}),
                          TensorFromValues<int32_t>(GetContext(), Shape{2}, {3, 4})};
  Tensor concatenated = Empty(GetContext(), Shape{4}, DType::INT32);
  ConcatOut(GetContext(), concatenated, inputs, 0);
  ExpectValues<int32_t>(GetContext(), concatenated, {1, 2, 3, 4});

  Tensor stacked = Empty(GetContext(), Shape{2, 2}, DType::INT32);
  StackOut(GetContext(), stacked, inputs, 0);
  ExpectValues<int32_t>(GetContext(), stacked, {1, 2, 3, 4});

  Tensor wrong_shape = Empty(GetContext(), Shape{3}, DType::FLOAT32);
  EXPECT_THROW(CastOut(GetContext(), wrong_shape, input), InvalidArgumentError);
  EXPECT_THROW(ConcatOut(GetContext(), input, inputs, 0), InvalidArgumentError);
}

TEST_F(OutputContractTest, ArithmeticComparisonAndLogicalOutFormsCoverTensorAndScalarBroadcasting) {
  Tensor lhs = TensorFromValues<int32_t>(GetContext(), Shape{2, 2}, {4, -6, 8, 3});
  Tensor rhs = TensorFromValues<int32_t>(GetContext(), Shape{2}, {2, 3});
  Tensor output = Empty(GetContext(), Shape{2, 2}, DType::INT32);

  SubtractOut(GetContext(), output, lhs, rhs);
  ExpectValues<int32_t>(GetContext(), output, {2, -9, 6, 0});
  MultiplyOut(GetContext(), output, lhs, Scalar{int64_t{2}});
  ExpectValues<int32_t>(GetContext(), output, {8, -12, 16, 6});
  DivideOut(GetContext(), output, lhs, Scalar{int64_t{2}});
  ExpectValues<int32_t>(GetContext(), output, {2, -3, 4, 1});
  MaximumOut(GetContext(), output, lhs, Scalar{int64_t{0}});
  ExpectValues<int32_t>(GetContext(), output, {4, 0, 8, 3});
  MinimumOut(GetContext(), output, lhs, Scalar{int64_t{0}});
  ExpectValues<int32_t>(GetContext(), output, {0, -6, 0, 0});

  Tensor bool_output = Empty(GetContext(), Shape{2, 2}, DType::BOOL);
  EqualOut(GetContext(), bool_output, lhs, Scalar{int64_t{4}});
  ExpectBoolValues(GetContext(), bool_output, {1, 0, 0, 0});
  NotEqualOut(GetContext(), bool_output, lhs, Scalar{int64_t{4}});
  ExpectBoolValues(GetContext(), bool_output, {0, 1, 1, 1});
  LessOut(GetContext(), bool_output, lhs, Scalar{int64_t{4}});
  ExpectBoolValues(GetContext(), bool_output, {0, 1, 0, 1});
  LessEqualOut(GetContext(), bool_output, lhs, Scalar{int64_t{4}});
  ExpectBoolValues(GetContext(), bool_output, {1, 1, 0, 1});
  GreaterOut(GetContext(), bool_output, lhs, rhs);
  ExpectBoolValues(GetContext(), bool_output, {1, 0, 1, 0});
  GreaterEqualOut(GetContext(), bool_output, lhs, rhs);
  ExpectBoolValues(GetContext(), bool_output, {1, 0, 1, 1});

  Tensor floating = FloatingTensorFromValues(GetContext(), Shape{4}, DType::FLOAT32, {-1.0F, 0.0F, 1.0F, 4.0F});
  Tensor unary = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  NegateOut(GetContext(), unary, floating);
  ExpectFloatValues(GetContext(), unary, {1, 0, -1, -4});
  AbsOut(GetContext(), unary, floating);
  ExpectFloatValues(GetContext(), unary, {1, 0, 1, 4});
  ReluOut(GetContext(), unary, floating);
  ExpectFloatValues(GetContext(), unary, {0, 0, 1, 4});
  ClampOut(GetContext(), unary, floating, Scalar{0.0}, Scalar{2.0});
  ExpectFloatValues(GetContext(), unary, {0, 0, 1, 2});

  Tensor condition = BoolTensorFromValues(GetContext(), Shape{4}, {1, 0, 1, 0});
  Tensor logical_rhs = BoolTensorFromValues(GetContext(), Shape{4}, {1, 1, 0, 0});
  Tensor logical_output = Empty(GetContext(), Shape{4}, DType::BOOL);
  LogicalAndOut(GetContext(), logical_output, condition, logical_rhs);
  ExpectBoolValues(GetContext(), logical_output, {1, 0, 0, 0});
  LogicalOrOut(GetContext(), logical_output, condition, logical_rhs);
  ExpectBoolValues(GetContext(), logical_output, {1, 1, 1, 0});
  LogicalNotOut(GetContext(), logical_output, condition);
  ExpectBoolValues(GetContext(), logical_output, {0, 1, 0, 1});
  WhereOut(GetContext(), unary, condition, floating, floating);
  ExpectFloatValues(GetContext(), unary, {-1, 0, 1, 4});
}

TEST_F(OutputContractTest, FloatingUnaryOutFormsMatchIndependentHostReferences) {
  const std::vector<float> values{0.25F, 1.0F, 2.0F, 4.0F};
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{4}, DType::FLOAT32, values);
  Tensor output = Empty(GetContext(), Shape{4}, DType::FLOAT32);

  ExpOut(GetContext(), output, input);
  ExpectFloatValues(GetContext(), output, Transform(values, [](float value) { return std::exp(value); }), 2.0e-5F);
  LogOut(GetContext(), output, input);
  ExpectFloatValues(GetContext(), output, Transform(values, [](float value) { return std::log(value); }), 2.0e-5F);
  SqrtOut(GetContext(), output, input);
  ExpectFloatValues(GetContext(), output, Transform(values, [](float value) { return std::sqrt(value); }), 2.0e-5F);
  RsqrtOut(GetContext(), output, input);
  ExpectFloatValues(GetContext(), output, Transform(values, [](float value) { return 1.0F / std::sqrt(value); }),
                    2.0e-5F);
  SinOut(GetContext(), output, input);
  ExpectFloatValues(GetContext(), output, Transform(values, [](float value) { return std::sin(value); }), 2.0e-5F);
  CosOut(GetContext(), output, input);
  ExpectFloatValues(GetContext(), output, Transform(values, [](float value) { return std::cos(value); }), 2.0e-5F);
  TanhOut(GetContext(), output, input);
  ExpectFloatValues(GetContext(), output, Transform(values, [](float value) { return std::tanh(value); }), 2.0e-5F);
  SigmoidOut(GetContext(), output, input);
  ExpectFloatValues(GetContext(), output,
                    Transform(values, [](float value) { return 1.0F / (1.0F + std::exp(-value)); }), 2.0e-5F);
  SiluOut(GetContext(), output, input);
  ExpectFloatValues(GetContext(), output,
                    Transform(values, [](float value) { return value / (1.0F + std::exp(-value)); }), 2.0e-5F);
  GeluOut(GetContext(), output, input, GeluApproximation::NONE);
  ExpectFloatValues(
      GetContext(), output,
      Transform(values,
                [](float value) { return 0.5F * value * (1.0F + std::erf(value / std::numbers::sqrt2_v<float>)); }),
      2.0e-4F);
  GeluOut(GetContext(), output, input, GeluApproximation::TANH);
  ExpectFloatValues(
      GetContext(), output,
      Transform(values,
                [](float value) {
                  constexpr float k_sqrt_two_over_pi = 0.7978845608028654F;
                  return 0.5F * value *
                         (1.0F + std::tanh(k_sqrt_two_over_pi * (value + (0.044715F * value * value * value))));
                }),
      2.0e-4F);
}

TEST_F(OutputContractTest, ReductionAndScanOutFormsPreserveAxesAndIdentities) {
  Tensor input = TensorFromValues<int32_t>(GetContext(), Shape{2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor output = Empty(GetContext(), Shape{2}, DType::INT32);
  ReductionOptions options{.axes_ = {1}, .keep_dimensions_ = false};
  SumOut(GetContext(), output, input, options);
  ExpectValues<int32_t>(GetContext(), output, {6, 15});

  Tensor float_input = FloatingTensorFromValues(GetContext(), Shape{2, 3}, DType::FLOAT32, {1, 2, 3, 4, 5, 6});
  Tensor float_output = Empty(GetContext(), Shape{2}, DType::FLOAT32);
  MeanOut(GetContext(), float_output, float_input, options);
  ExpectFloatValues(GetContext(), float_output, {2, 5});
  MinimumOut(GetContext(), float_output, float_input, options);
  ExpectFloatValues(GetContext(), float_output, {1, 4});
  MaximumOut(GetContext(), float_output, float_input, options);
  ExpectFloatValues(GetContext(), float_output, {3, 6});

  Tensor arg_output = Empty(GetContext(), Shape{2}, DType::INT64);
  ArgMinOut(GetContext(), arg_output, float_input, 1);
  ExpectValues<int64_t>(GetContext(), arg_output, {0, 0});
  ArgMaxOut(GetContext(), arg_output, float_input, 1);
  ExpectValues<int64_t>(GetContext(), arg_output, {2, 2});

  Tensor bool_input = BoolTensorFromValues(GetContext(), Shape{2, 3}, {1, 1, 0, 0, 1, 0});
  Tensor bool_output = Empty(GetContext(), Shape{2}, DType::BOOL);
  AnyOut(GetContext(), bool_output, bool_input, options);
  ExpectBoolValues(GetContext(), bool_output, {1, 1});
  AllOut(GetContext(), bool_output, bool_input, options);
  ExpectBoolValues(GetContext(), bool_output, {0, 0});

  Tensor scan_output = Empty(GetContext(), Shape{2, 3}, DType::INT32);
  CumulativeSumOut(GetContext(), scan_output, input, -1);
  ExpectValues<int32_t>(GetContext(), scan_output, {1, 3, 6, 4, 9, 15});
}

TEST_F(OutputContractTest, IndexingOutFormsMatchCanonicalGatherReferences) {
  Tensor input = TensorFromValues<int32_t>(GetContext(), Shape{2, 3}, {10, 11, 12, 20, 21, 22});
  Tensor index = TensorFromValues<int32_t>(GetContext(), Shape{2}, {2, 0});
  Tensor selected = Empty(GetContext(), Shape{2, 2}, DType::INT32);
  IndexSelectOut(GetContext(), selected, input, 1, index);
  ExpectValues<int32_t>(GetContext(), selected, {12, 10, 22, 20});

  Tensor gather_index = TensorFromValues<int32_t>(GetContext(), Shape{2, 2}, {2, 0, 1, 1});
  Tensor gathered = Empty(GetContext(), Shape{2, 2}, DType::INT32);
  GatherOut(GetContext(), gathered, input, 1, gather_index);
  ExpectValues<int32_t>(GetContext(), gathered, {12, 10, 21, 21});

  Tensor taken = Empty(GetContext(), Shape{2, 2}, DType::INT32);
  TakeAlongDimensionOut(GetContext(), taken, input, gather_index, 1);
  ExpectValues<int32_t>(GetContext(), taken, {12, 10, 21, 21});

  Tensor scatter_index = TensorFromValues<int32_t>(GetContext(), Shape{2, 2}, {2, 0, 1, 2});
  Tensor scatter_source = TensorFromValues<int32_t>(GetContext(), Shape{2, 2}, {100, 101, 200, 201});
  Tensor scattered = Empty(GetContext(), Shape{2, 3}, DType::INT32);
  ScatterElementsOut(GetContext(), scattered, input, 1, scatter_index, scatter_source);
  ExpectValues<int32_t>(GetContext(), scattered, {101, 11, 100, 20, 200, 201});

  const Tensor &table = input;
  Tensor rows_index = TensorFromValues<int32_t>(GetContext(), Shape{2}, {1, 0});
  Tensor rows = Empty(GetContext(), Shape{2, 3}, DType::INT32);
  GatherRowsOut(GetContext(), rows, table, rows_index);
  ExpectValues<int32_t>(GetContext(), rows, {20, 21, 22, 10, 11, 12});

  Tensor weight = TensorFromValues<float>(GetContext(), Shape{3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor tokens = TensorFromValues<int32_t>(GetContext(), Shape{2}, {2, 0});
  Tensor embedding = Empty(GetContext(), Shape{2, 2}, DType::FLOAT32);
  EmbeddingOut(GetContext(), embedding, weight, tokens);
  ExpectFloatValues(GetContext(), embedding, {5, 6, 1, 2});
}

TEST_F(OutputContractTest, LinalgNormalizationSoftmaxAndAttentionOutForms) {
  Tensor lhs = TensorFromValues<float>(GetContext(), Shape{2, 2}, {1, 2, 3, 4});
  Tensor rhs = TensorFromValues<float>(GetContext(), Shape{2, 2}, {5, 6, 7, 8});
  Tensor matrix = Empty(GetContext(), Shape{2, 2}, DType::FLOAT32);
  MatmulOut(GetContext(), matrix, lhs, rhs);
  ExpectFloatValues(GetContext(), matrix, {19, 22, 43, 50});

  Tensor batched_lhs = View(lhs, Shape{1, 2, 2});
  Tensor batched_rhs = View(rhs, Shape{1, 2, 2});
  Tensor batched = Empty(GetContext(), Shape{1, 2, 2}, DType::FLOAT32);
  BatchedMatmulOut(GetContext(), batched, batched_lhs, batched_rhs);
  ExpectFloatValues(GetContext(), batched, {19, 22, 43, 50});

  Tensor weight = TensorFromValues<float>(GetContext(), Shape{2, 2}, {2, 0, 0, 3});
  Tensor bias = TensorFromValues<float>(GetContext(), Shape{2}, {1, -1});
  Tensor linear = Empty(GetContext(), Shape{2, 2}, DType::FLOAT32);
  LinearOptions linear_options;
  linear_options.activation_ = LinearActivation::RELU;
  LinearOut(GetContext(), linear, lhs, weight, bias, linear_options);
  ExpectFloatValues(GetContext(), linear, {3, 5, 7, 11});

  Tensor normalized = Empty(GetContext(), Shape{2, 2}, DType::FLOAT32);
  LayerNormOut(GetContext(), normalized, lhs, std::nullopt, std::nullopt, {.normalized_rank_ = 1, .epsilon_ = 0});
  ExpectFloatValues(GetContext(), normalized, {-1, 1, -1, 1}, 1.0e-4F);
  RmsNormOut(GetContext(), normalized, lhs, std::nullopt, {.normalized_rank_ = 1, .epsilon_ = 0});
  ExpectFloatValues(GetContext(), normalized, {0.6324555F, 1.264911F, 0.8485281F, 1.1313708F}, 1.0e-4F);

  Tensor softmax = Empty(GetContext(), Shape{2, 2}, DType::FLOAT32);
  SoftmaxOut(GetContext(), softmax, lhs, {.axes_ = {1}});
  ExpectFloatValues(GetContext(), softmax, {0.2689414F, 0.7310586F, 0.2689414F, 0.7310586F}, 1.0e-4F);
  Tensor log_softmax = Empty(GetContext(), Shape{2, 2}, DType::FLOAT32);
  LogSoftmaxOut(GetContext(), log_softmax, lhs, {.axes_ = {1}});
  ExpectFloatValues(GetContext(), log_softmax, {-1.3132617F, -0.3132617F, -1.3132617F, -0.3132617F}, 1.0e-4F);

  Tensor query = TensorFromValues<float>(GetContext(), Shape{1, 1, 1, 2}, {1, 0});
  Tensor key = TensorFromValues<float>(GetContext(), Shape{1, 1, 2, 2}, {1, 0, 0, 1});
  Tensor value = TensorFromValues<float>(GetContext(), Shape{1, 1, 2, 2}, {10, 0, 0, 20});
  Tensor attention = Empty(GetContext(), Shape{1, 1, 1, 2}, DType::FLOAT32);
  ScaledDotProductAttentionOut(GetContext(), attention, query, key, value, std::nullopt, {.scale_ = 1.0F});
  const float first = (std::numbers::e_v<float> * 10.0F) / (std::numbers::e_v<float> + 1.0F);
  const float second = 20.0F / (std::numbers::e_v<float> + 1.0F);
  ExpectFloatValues(GetContext(), attention, {first, second}, 1.0e-4F);
}

TEST_F(OutputContractTest, RandomAndTopKOutFormsHonorGeneratorAndSelectionContracts) {
  Generator generator{GetContext(), 123};
  Tensor uniform = Empty(GetContext(), Shape{16}, DType::FLOAT32);
  UniformOut(GetContext(), uniform, generator, {.low_ = -2.0, .high_ = 3.0});
  const auto uniform_values = FloatingTensorToValues(GetContext(), uniform);
  ASSERT_EQ(uniform_values.size(), 16U);
  for (float value : uniform_values) {
    EXPECT_GE(value, -2.0F);
    EXPECT_LT(value, 3.0F);
  }

  Tensor normal = Empty(GetContext(), Shape{8}, DType::FLOAT32);
  NormalOut(GetContext(), normal, generator, {.mean_ = 2.0, .standard_deviation_ = 0.0});
  ExpectFloatValues(GetContext(), normal, std::vector<float>(8, 2.0F));

  Tensor input = FloatingTensorFromValues(GetContext(), Shape{1, 4}, DType::FLOAT32, {4, 1, 3, 2});
  Tensor values = Empty(GetContext(), Shape{1, 2}, DType::FLOAT32);
  Tensor indices = Empty(GetContext(), Shape{1, 2}, DType::INT64);
  TopKOut(GetContext(), values, indices, input, {.axis_ = 1, .k_ = 2, .largest_ = true});
  ExpectFloatValues(GetContext(), values, {4, 3});
  ExpectValues<int64_t>(GetContext(), indices, {0, 2});
}

TEST_F(OutputContractTest, PinnedCopyOutFormsCompleteTheHostDeviceTransferMatrix) {
  Tensor source = TensorFromValues<int32_t>(GetContext(), Shape{4}, {4, 3, 2, 1});
  PinnedBuffer pinned = GetRuntime().AllocatePinned(sizeof(int32_t) * 4);
  CopyToPinnedAsync(GetContext(), pinned, source);
  GetContext().Synchronize();
  const auto *host_values = static_cast<const int32_t *>(pinned.GetData());
  ASSERT_NE(host_values, nullptr);
  EXPECT_EQ(std::vector<int32_t>(host_values, host_values + 4), (std::vector<int32_t>{4, 3, 2, 1}));

  Tensor destination = Empty(GetContext(), Shape{4}, DType::INT32);
  CopyFromPinnedAsync(GetContext(), destination, pinned);
  ExpectValues<int32_t>(GetContext(), destination, {4, 3, 2, 1});
}

}  // namespace
}  // namespace ttl::test
