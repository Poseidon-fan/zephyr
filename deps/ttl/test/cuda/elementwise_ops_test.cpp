#include "ttl/ops/elementwise.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/layout.hpp"
#include "ttl/ops/cast.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/runtime.hpp"
#include "ttl/scalar.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

constexpr std::array ARITHMETIC_DTYPES{
    DType::INT32, DType::INT64, DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32,
};

constexpr std::array ORDERED_DTYPES{
    DType::UINT8, DType::INT32, DType::INT64, DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32,
};

constexpr std::array ALL_DTYPES{
    DType::BOOL, DType::UINT8, DType::INT32, DType::INT64, DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32,
};

class CountingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord /*error*/) noexcept override { count_.fetch_add(1, std::memory_order_relaxed); }

  [[nodiscard]] auto GetCount() const noexcept -> size_t { return count_.load(std::memory_order_relaxed); }

 private:
  std::atomic<size_t> count_{0};
};

[[nodiscard]] auto MakeRuntimeOptions(const std::shared_ptr<ErrorSink> &error_sink) -> RuntimeOptions {
  auto options = RuntimeOptions{};
  options.devices_ = {Device{0}};
  options.device_memory_.enable_maintenance_thread_ = false;
  options.event_pool_capacity_per_device_ = 64;
  options.error_sink_ = error_sink;
  return options;
}

template <TensorStorageType T, size_t size>
void CopyFromHost(ExecutionContext &context, Tensor &tensor, const std::array<T, size> &host) {
  ASSERT_EQ(tensor.GetNumElements(), static_cast<int64_t>(size));
  context.Synchronize();
  if constexpr (size != 0) {
    ASSERT_EQ(cudaMemcpy(internal::TensorAccess::GetMutableData<T>(tensor), host.data(), sizeof(host),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
}

template <TensorStorageType T>
[[nodiscard]] auto CopyToHost(ExecutionContext &context, const Tensor &tensor) -> std::vector<T> {
  context.Synchronize();
  std::vector<T> host(static_cast<size_t>(tensor.GetNumElements()));
  if (!host.empty()) {
    EXPECT_EQ(cudaMemcpy(host.data(), tensor.GetData<T>(), host.size() * sizeof(T), cudaMemcpyDeviceToHost),
              cudaSuccess);
  }
  return host;
}

[[nodiscard]] auto CopyBoolToHost(ExecutionContext &context, const Tensor &tensor) -> std::vector<uint8_t> {
  context.Synchronize();
  std::vector<uint8_t> host(static_cast<size_t>(tensor.GetNumElements()));
  if (!host.empty()) {
    EXPECT_EQ(cudaMemcpy(host.data(), tensor.GetData<bool>(), host.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  }
  return host;
}

[[nodiscard]] auto CopyAsFloat(ExecutionContext &context, const Tensor &tensor) -> std::vector<float> {
  const auto floating = Cast(context, tensor, DType::FLOAT32);
  return CopyToHost<float>(context, floating);
}

void ExpectNear(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 1.0e-5F) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], tolerance) << "at index " << index;
  }
}

class ElementwiseOpsTest : public testing::Test {
 protected:
  void SetUp() override {
    error_sink_ = std::make_shared<CountingErrorSink>();
    runtime_ = std::make_unique<Runtime>(MakeRuntimeOptions(error_sink_));
    context_.emplace(runtime_->CreateExecutionContext(Device{0}));
  }

  void TearDown() override {
    context_.reset();
    runtime_->Shutdown();
    runtime_.reset();
    EXPECT_EQ(error_sink_->GetCount(), 0);
  }

  [[nodiscard]] auto GetContext() -> ExecutionContext & { return *context_; }

  std::shared_ptr<CountingErrorSink> error_sink_;
  std::unique_ptr<Runtime> runtime_;
  std::optional<ExecutionContext> context_;
};

TEST_F(ElementwiseOpsTest, DispatchesCompleteSupportedDTypeMatrices) {
  for (const auto dtype : ARITHMETIC_DTYPES) {
    const auto lhs = Ones(GetContext(), Shape{19}, dtype);
    const auto rhs = Ones(GetContext(), Shape{19}, dtype);
    const auto sum = Add(GetContext(), lhs, rhs);
    EXPECT_EQ(CopyAsFloat(GetContext(), sum), std::vector<float>(19, 2.0F));
    if (IsFloating(dtype)) {
      const auto absolute = Abs(GetContext(), Multiply(GetContext(), lhs, Scalar{-1.0}));
      EXPECT_EQ(CopyAsFloat(GetContext(), absolute), std::vector<float>(19, 1.0F));
    }
  }

  for (const auto dtype : ALL_DTYPES) {
    const auto lhs = Ones(GetContext(), Shape{19}, dtype);
    const auto rhs = Ones(GetContext(), Shape{19}, dtype);
    EXPECT_EQ(CopyBoolToHost(GetContext(), Equal(GetContext(), lhs, rhs)), std::vector<uint8_t>(19, 1));

    const auto condition = Ones(GetContext(), Shape{19}, DType::BOOL);
    const auto selected = Where(GetContext(), condition, lhs, Zeros(GetContext(), Shape{19}, dtype));
    EXPECT_EQ(CopyAsFloat(GetContext(), selected), std::vector<float>(19, 1.0F));
  }

  for (const auto dtype : ORDERED_DTYPES) {
    const auto lhs = Zeros(GetContext(), Shape{19}, dtype);
    const auto rhs = Ones(GetContext(), Shape{19}, dtype);
    EXPECT_EQ(CopyBoolToHost(GetContext(), Less(GetContext(), lhs, rhs)), std::vector<uint8_t>(19, 1));
  }
}

TEST_F(ElementwiseOpsTest, ComputesTensorAndScalarBinaryOperations) {
  auto lhs = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  auto rhs = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  CopyFromHost(GetContext(), lhs, std::array<float, 4>{1.0F, 2.0F, -3.0F, 4.0F});
  CopyFromHost(GetContext(), rhs, std::array<float, 4>{2.0F, -1.0F, 2.0F, 0.5F});

  EXPECT_EQ(CopyToHost<float>(GetContext(), Add(GetContext(), lhs, rhs)),
            (std::vector<float>{3.0F, 1.0F, -1.0F, 4.5F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Subtract(GetContext(), lhs, rhs)),
            (std::vector<float>{-1.0F, 3.0F, -5.0F, 3.5F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Multiply(GetContext(), lhs, rhs)),
            (std::vector<float>{2.0F, -2.0F, -6.0F, 2.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Divide(GetContext(), lhs, rhs)),
            (std::vector<float>{0.5F, -2.0F, -1.5F, 8.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Maximum(GetContext(), lhs, rhs)),
            (std::vector<float>{2.0F, 2.0F, 2.0F, 4.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Minimum(GetContext(), lhs, rhs)),
            (std::vector<float>{1.0F, -1.0F, -3.0F, 0.5F}));

  EXPECT_EQ(CopyToHost<float>(GetContext(), Add(GetContext(), lhs, Scalar{2.0})),
            (std::vector<float>{3.0F, 4.0F, -1.0F, 6.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Subtract(GetContext(), lhs, Scalar{2.0})),
            (std::vector<float>{-1.0F, 0.0F, -5.0F, 2.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Multiply(GetContext(), lhs, Scalar{-2.0})),
            (std::vector<float>{-2.0F, -4.0F, 6.0F, -8.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Divide(GetContext(), lhs, Scalar{2.0})),
            (std::vector<float>{0.5F, 1.0F, -1.5F, 2.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Maximum(GetContext(), lhs, Scalar{0.0})),
            (std::vector<float>{1.0F, 2.0F, 0.0F, 4.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Minimum(GetContext(), lhs, Scalar{0.0})),
            (std::vector<float>{0.0F, 0.0F, -3.0F, 0.0F}));
}

TEST_F(ElementwiseOpsTest, DefinesIntegerModuloAndCheckedDivision) {
  auto lhs = Empty(GetContext(), Shape{4}, DType::INT32);
  auto rhs = Empty(GetContext(), Shape{4}, DType::INT32);
  CopyFromHost(GetContext(), lhs,
               std::array<int32_t, 4>{std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(), 7,
                                      std::numeric_limits<int32_t>::min()});
  CopyFromHost(GetContext(), rhs, std::array<int32_t, 4>{1, -1, 2, -1});

  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), Add(GetContext(), lhs, rhs)),
            (std::vector<int32_t>{std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), 9,
                                  std::numeric_limits<int32_t>::max()}));
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), Divide(GetContext(), lhs, rhs)),
            (std::vector<int32_t>{std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(), 3,
                                  std::numeric_limits<int32_t>::min()}));
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), Multiply(GetContext(), lhs, Scalar{int64_t{2}})),
            (std::vector<int32_t>{-2, 0, 14, 0}));

  CopyFromHost(GetContext(), rhs, std::array<int32_t, 4>{1, 0, 1, 1});
  auto invalid = Empty(GetContext(), Shape{4}, DType::INT32);
  DivideOut(GetContext(), invalid, lhs, rhs);
  try {
    GetContext().CheckAsyncErrors();
    FAIL() << "expected integer division by zero";
  } catch (const DeviceError &error) {
    EXPECT_NE(error.GetMessage().find("integer division by zero"), std::string::npos);
  }

  DivideOut(GetContext(), invalid, lhs, Scalar{int64_t{0}});
  EXPECT_THROW(GetContext().CheckAsyncErrors(), DeviceError);
  EXPECT_NO_THROW(GetContext().Synchronize());
}

TEST_F(ElementwiseOpsTest, ImplementsAllComparisonSemantics) {
  auto lhs = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  auto rhs = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  CopyFromHost(GetContext(), lhs, std::array<float, 4>{1.0F, 2.0F, std::numeric_limits<float>::quiet_NaN(), -0.0F});
  CopyFromHost(GetContext(), rhs, std::array<float, 4>{1.0F, 3.0F, std::numeric_limits<float>::quiet_NaN(), 0.0F});

  EXPECT_EQ(CopyBoolToHost(GetContext(), Equal(GetContext(), lhs, rhs)), (std::vector<uint8_t>{1, 0, 0, 1}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), NotEqual(GetContext(), lhs, rhs)), (std::vector<uint8_t>{0, 1, 1, 0}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), Less(GetContext(), lhs, rhs)), (std::vector<uint8_t>{0, 1, 0, 0}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), LessEqual(GetContext(), lhs, rhs)), (std::vector<uint8_t>{1, 1, 0, 1}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), Greater(GetContext(), lhs, rhs)), (std::vector<uint8_t>{0, 0, 0, 0}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), GreaterEqual(GetContext(), lhs, rhs)), (std::vector<uint8_t>{1, 0, 0, 1}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), Greater(GetContext(), lhs, Scalar{1.5})), (std::vector<uint8_t>{0, 1, 0, 0}));

  auto boolean = Ones(GetContext(), Shape{4}, DType::BOOL);
  EXPECT_EQ(CopyBoolToHost(GetContext(), Equal(GetContext(), boolean, Scalar{true})), std::vector<uint8_t>(4, 1));
}

TEST_F(ElementwiseOpsTest, ImplementsUnaryMathAndActivations) {
  auto input = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  CopyFromHost(GetContext(), input, std::array<float, 4>{0.25F, 0.5F, 1.0F, 2.0F});

  ExpectNear(CopyToHost<float>(GetContext(), Negate(GetContext(), input)), {-0.25F, -0.5F, -1.0F, -2.0F});
  ExpectNear(CopyToHost<float>(GetContext(), Abs(GetContext(), Negate(GetContext(), input))),
             {0.25F, 0.5F, 1.0F, 2.0F});
  ExpectNear(CopyToHost<float>(GetContext(), Exp(GetContext(), input)),
             {std::exp(0.25F), std::exp(0.5F), std::numbers::e_v<float>, std::exp(2.0F)});
  ExpectNear(CopyToHost<float>(GetContext(), Log(GetContext(), input)),
             {std::log(0.25F), std::log(0.5F), 0.0F, std::numbers::ln2_v<float>});
  ExpectNear(CopyToHost<float>(GetContext(), Sqrt(GetContext(), input)),
             {0.5F, std::numbers::sqrt2_v<float> / 2.0F, 1.0F, std::numbers::sqrt2_v<float>});
  ExpectNear(CopyToHost<float>(GetContext(), Rsqrt(GetContext(), input)),
             {2.0F, std::numbers::sqrt2_v<float>, 1.0F, 1.0F / std::numbers::sqrt2_v<float>});
  ExpectNear(CopyToHost<float>(GetContext(), Sin(GetContext(), input)),
             {std::sin(0.25F), std::sin(0.5F), std::sin(1.0F), std::sin(2.0F)});
  ExpectNear(CopyToHost<float>(GetContext(), Cos(GetContext(), input)),
             {std::cos(0.25F), std::cos(0.5F), std::cos(1.0F), std::cos(2.0F)});
  ExpectNear(CopyToHost<float>(GetContext(), Tanh(GetContext(), input)),
             {std::tanh(0.25F), std::tanh(0.5F), std::tanh(1.0F), std::tanh(2.0F)});

  auto activation_input = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  CopyFromHost(GetContext(), activation_input, std::array<float, 4>{-100.0F, -1.0F, 0.0F, 2.0F});
  const auto sigmoid = CopyToHost<float>(GetContext(), Sigmoid(GetContext(), activation_input));
  const auto exp_negative_100 = std::exp(-100.0F);
  EXPECT_FLOAT_EQ(sigmoid[0], exp_negative_100 / (1.0F + exp_negative_100));
  EXPECT_NEAR(sigmoid[1], 1.0F / (1.0F + std::exp(1.0F)), 1.0e-6F);
  EXPECT_FLOAT_EQ(sigmoid[2], 0.5F);
  EXPECT_NEAR(sigmoid[3], 1.0F / (1.0F + std::exp(-2.0F)), 1.0e-6F);
  ExpectNear(CopyToHost<float>(GetContext(), Relu(GetContext(), activation_input)), {0.0F, 0.0F, 0.0F, 2.0F});

  const auto silu = CopyToHost<float>(GetContext(), Silu(GetContext(), activation_input));
  EXPECT_NEAR(silu[1], -1.0F / (1.0F + std::exp(1.0F)), 1.0e-6F);
  EXPECT_NEAR(silu[3], 2.0F / (1.0F + std::exp(-2.0F)), 1.0e-6F);

  const auto gelu_exact =
      CopyToHost<float>(GetContext(), Gelu(GetContext(), activation_input, GeluApproximation::NONE));
  const auto gelu_tanh = CopyToHost<float>(GetContext(), Gelu(GetContext(), activation_input, GeluApproximation::TANH));
  EXPECT_FLOAT_EQ(gelu_exact[2], 0.0F);
  EXPECT_FLOAT_EQ(gelu_tanh[2], 0.0F);
  EXPECT_NEAR(gelu_exact[3], 1.9544997F, 1.0e-5F);
  EXPECT_NEAR(gelu_tanh[3], 1.9545977F, 1.0e-5F);

  ExpectNear(CopyToHost<float>(GetContext(), Clamp(GetContext(), activation_input, Scalar{-0.5}, Scalar{1.0})),
             {-0.5F, -0.5F, 0.0F, 1.0F});
  ExpectNear(CopyToHost<float>(GetContext(), Clamp(GetContext(), activation_input, std::nullopt, Scalar{0.0})),
             {-100.0F, -1.0F, 0.0F, 0.0F});
}

TEST_F(ElementwiseOpsTest, ComputesHalfAndBFloatMathInFloat32) {
  auto half = Full(GetContext(), Shape{4}, Scalar{0.5}, DType::FLOAT16);
  auto bfloat = Full(GetContext(), Shape{4}, Scalar{-1.0}, DType::BFLOAT16);
  ExpectNear(CopyAsFloat(GetContext(), Exp(GetContext(), half)), std::vector<float>(4, std::exp(0.5F)), 2.0e-3F);
  ExpectNear(CopyAsFloat(GetContext(), Silu(GetContext(), bfloat)),
             std::vector<float>(4, -1.0F / (1.0F + std::numbers::e_v<float>)), 5.0e-3F);
}

TEST_F(ElementwiseOpsTest, ImplementsLogicalAndWhereOperations) {
  auto lhs = Empty(GetContext(), Shape{4}, DType::BOOL);
  auto rhs = Empty(GetContext(), Shape{4}, DType::BOOL);
  CopyFromHost(GetContext(), lhs, std::array<bool, 4>{false, false, true, true});
  CopyFromHost(GetContext(), rhs, std::array<bool, 4>{false, true, false, true});
  EXPECT_EQ(CopyBoolToHost(GetContext(), LogicalAnd(GetContext(), lhs, rhs)), (std::vector<uint8_t>{0, 0, 0, 1}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), LogicalOr(GetContext(), lhs, rhs)), (std::vector<uint8_t>{0, 1, 1, 1}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), LogicalNot(GetContext(), lhs)), (std::vector<uint8_t>{1, 1, 0, 0}));

  auto true_value = Empty(GetContext(), Shape{4}, DType::INT64);
  auto false_value = Empty(GetContext(), Shape{4}, DType::INT64);
  CopyFromHost(GetContext(), true_value, std::array<int64_t, 4>{1, 2, 3, 4});
  CopyFromHost(GetContext(), false_value, std::array<int64_t, 4>{-1, -2, -3, -4});
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), Where(GetContext(), rhs, true_value, false_value)),
            (std::vector<int64_t>{-1, 2, -3, 4}));
}

TEST_F(ElementwiseOpsTest, SupportsBroadcastStridedOutputsAndExactAlias) {
  auto lhs = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  auto rhs = Empty(GetContext(), Shape{3}, DType::FLOAT32);
  CopyFromHost(GetContext(), lhs, std::array<float, 6>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  CopyFromHost(GetContext(), rhs, std::array<float, 3>{10.0F, 20.0F, 30.0F});

  auto strided_output = EmptyStrided(GetContext(), Shape{2, 3}, Strides{1, 2}, DType::FLOAT32);
  AddOut(GetContext(), strided_output, lhs, rhs);
  EXPECT_EQ(CopyToHost<float>(GetContext(), Contiguous(GetContext(), strided_output)),
            (std::vector<float>{10.0F, 21.0F, 32.0F, 13.0F, 24.0F, 35.0F}));

  AddOut(GetContext(), lhs, lhs, rhs);
  EXPECT_EQ(CopyToHost<float>(GetContext(), lhs), (std::vector<float>{10.0F, 21.0F, 32.0F, 13.0F, 24.0F, 35.0F}));
  ReluOut(GetContext(), lhs, lhs);
  EXPECT_NO_THROW(GetContext().Synchronize());

  auto condition = Greater(GetContext(), lhs, Scalar{20.0});
  auto false_value = Zeros(GetContext(), Shape{2, 3}, DType::FLOAT32);
  WhereOut(GetContext(), lhs, condition, lhs, false_value);
  EXPECT_EQ(CopyToHost<float>(GetContext(), lhs), (std::vector<float>{0.0F, 21.0F, 32.0F, 0.0F, 24.0F, 35.0F}));

  auto base = Empty(GetContext(), Shape{7}, DType::FLOAT32);
  auto overlapping_output = Narrow(base, 0, 0, 6);
  const auto overlapping_input = Narrow(base, 0, 1, 6);
  EXPECT_THROW(AddOut(GetContext(), overlapping_output, overlapping_input, Scalar{1.0}), InvalidArgumentError);
}

TEST_F(ElementwiseOpsTest, PreservesNanInfinityAndSignedZeroSemantics) {
  auto input = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  auto other = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  CopyFromHost(GetContext(), input, std::array<float, 4>{std::numeric_limits<float>::quiet_NaN(), -0.0F, 1.0F, -1.0F});
  CopyFromHost(GetContext(), other, std::array<float, 4>{2.0F, 0.0F, 0.0F, 0.0F});

  const auto maximum = CopyToHost<float>(GetContext(), Maximum(GetContext(), input, other));
  const auto minimum = CopyToHost<float>(GetContext(), Minimum(GetContext(), input, other));
  const auto relu = CopyToHost<float>(GetContext(), Relu(GetContext(), input));
  const auto clamped = CopyToHost<float>(GetContext(), Clamp(GetContext(), input, Scalar{-0.5}, Scalar{0.5}));
  EXPECT_TRUE(std::isnan(maximum[0]));
  EXPECT_TRUE(std::isnan(minimum[0]));
  EXPECT_TRUE(std::isnan(relu[0]));
  EXPECT_TRUE(std::isnan(clamped[0]));

  const auto absolute = CopyToHost<float>(GetContext(), Abs(GetContext(), input));
  EXPECT_FALSE(std::signbit(absolute[1]));
  const auto divided = CopyToHost<float>(GetContext(), Divide(GetContext(), input, other));
  EXPECT_TRUE(std::isinf(divided[2]));
  EXPECT_TRUE(std::isinf(divided[3]));
}

TEST_F(ElementwiseOpsTest, EnforcesSchemasOptionsEmptyAndAliasContracts) {
  auto unsigned_input = Ones(GetContext(), Shape{2}, DType::UINT8);
  EXPECT_THROW([[maybe_unused]] const auto output = Add(GetContext(), unsigned_input, unsigned_input),
               NotSupportedError);
  EXPECT_THROW([[maybe_unused]] const auto output = Negate(GetContext(), unsigned_input), NotSupportedError);

  auto boolean = Ones(GetContext(), Shape{2}, DType::BOOL);
  EXPECT_THROW([[maybe_unused]] const auto output = Less(GetContext(), boolean, boolean), NotSupportedError);
  EXPECT_THROW([[maybe_unused]] const auto output = LogicalAnd(GetContext(), boolean, unsigned_input),
               NotSupportedError);

  auto floating = Ones(GetContext(), Shape{2}, DType::FLOAT32);
  auto integer = Ones(GetContext(), Shape{2}, DType::INT32);
  EXPECT_THROW([[maybe_unused]] const auto output = Add(GetContext(), floating, integer), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto output = Where(GetContext(), floating, floating, floating),
               NotSupportedError);
  EXPECT_THROW([[maybe_unused]] const auto output = Where(GetContext(), boolean, floating, integer),
               InvalidArgumentError);

  auto wrong_shape = Empty(GetContext(), Shape{3}, DType::FLOAT32);
  EXPECT_THROW(AddOut(GetContext(), wrong_shape, floating, floating), InvalidArgumentError);
  auto wrong_dtype = Empty(GetContext(), Shape{2}, DType::INT32);
  EXPECT_THROW(AddOut(GetContext(), wrong_dtype, floating, floating), InvalidArgumentError);

  constexpr auto invalid_approximation = std::bit_cast<GeluApproximation>(uint8_t{UINT8_MAX});
  EXPECT_THROW([[maybe_unused]] const auto output = Gelu(GetContext(), floating, invalid_approximation),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto output = Clamp(GetContext(), floating, std::nullopt, std::nullopt),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto output = Clamp(GetContext(), floating, Scalar{2.0}, Scalar{1.0}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto output =
                   Clamp(GetContext(), floating, Scalar{std::numeric_limits<double>::quiet_NaN()}, std::nullopt),
               InvalidArgumentError);

  auto empty = Empty(GetContext(), Shape{0}, DType::INT32);
  EXPECT_THROW([[maybe_unused]] const auto output = Add(GetContext(), empty, Scalar{1.5}), InvalidArgumentError);
  EXPECT_NO_THROW([[maybe_unused]] const auto output = Add(GetContext(), empty, Scalar{int64_t{1}}));

  auto condition = Ones(GetContext(), Shape{2}, DType::BOOL);
  auto true_value = Zeros(GetContext(), Shape{2}, DType::BOOL);
  EXPECT_THROW(WhereOut(GetContext(), condition, condition, true_value, boolean), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl
