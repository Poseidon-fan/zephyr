#include "ttl/ops/attention.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/layout.hpp"
#include "ttl/ops/cast.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/runtime.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

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

template <TensorStorageType T>
void CopyFromHost(ExecutionContext &context, Tensor &tensor, const std::vector<T> &host) {
  ASSERT_EQ(tensor.GetNumElements(), static_cast<int64_t>(host.size()));
  context.Synchronize();
  if (!host.empty()) {
    ASSERT_EQ(cudaMemcpy(internal::TensorAccess::GetMutableData<T>(tensor), host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
}

void CopyBoolFromHost(ExecutionContext &context, Tensor &tensor, const std::vector<uint8_t> &host) {
  ASSERT_EQ(tensor.GetDType(), DType::BOOL);
  ASSERT_EQ(tensor.GetNumElements(), static_cast<int64_t>(host.size()));
  context.Synchronize();
  if (!host.empty()) {
    ASSERT_EQ(cudaMemcpy(internal::TensorAccess::GetMutableData<bool>(tensor), host.data(), host.size(),
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

[[nodiscard]] auto CopyAsFloat(ExecutionContext &context, const Tensor &tensor) -> std::vector<float> {
  return CopyToHost<float>(context, Cast(context, tensor, DType::FLOAT32));
}

class AttentionTest : public testing::Test {
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

TEST_F(AttentionTest, ComputesStableDenseAttention) {
  auto query = Empty(GetContext(), Shape{1, 1, 2, 1}, DType::FLOAT32);
  auto key = Empty(GetContext(), Shape{1, 1, 3, 1}, DType::FLOAT32);
  auto value = Empty(GetContext(), Shape{1, 1, 3, 2}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), query, {1.0F, 2.0F});
  CopyFromHost<float>(GetContext(), key, {1.0F, 2.0F, 3.0F});
  CopyFromHost<float>(GetContext(), value, {10.0F, 0.0F, 0.0F, 10.0F, 5.0F, 5.0F});

  const auto actual = CopyToHost<float>(
      GetContext(),
      ScaledDotProductAttention(GetContext(), query, key, value, std::nullopt, SdpaOptions{.scale_ = 1.0F}));
  for (size_t row = 0; row < 2; ++row) {
    const auto factor = static_cast<float>(row + 1);
    const auto maximum = 3.0F * factor;
    const std::array weights{std::exp((1.0F * factor) - maximum), std::exp((2.0F * factor) - maximum), 1.0F};
    const auto normalizer = weights[0] + weights[1] + weights[2];
    EXPECT_NEAR(actual[row * 2], ((10.0F * weights[0]) + (5.0F * weights[2])) / normalizer, 1.0e-5F);
    EXPECT_NEAR(actual[(row * 2) + 1], ((10.0F * weights[1]) + (5.0F * weights[2])) / normalizer, 1.0e-5F);
  }
}

TEST_F(AttentionTest, AppliesBooleanAndAdditiveBroadcastMasks) {
  const auto query = Zeros(GetContext(), Shape{1, 1, 2, 1}, DType::FLOAT32);
  const auto key = Zeros(GetContext(), Shape{1, 1, 3, 1}, DType::FLOAT32);
  auto value = Empty(GetContext(), Shape{1, 1, 3, 1}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), value, {1.0F, 3.0F, 5.0F});
  auto boolean_mask = Empty(GetContext(), Shape{2, 3}, DType::BOOL);
  CopyBoolFromHost(GetContext(), boolean_mask, {1, 0, 1, 0, 0, 0});
  EXPECT_EQ(CopyToHost<float>(GetContext(), ScaledDotProductAttention(GetContext(), query, key, value, boolean_mask)),
            (std::vector<float>{3.0F, 0.0F}));

  auto additive_mask = Empty(GetContext(), Shape{1, 1, 1, 3}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), additive_mask, {0.0F, -std::numeric_limits<float>::infinity(), 0.0F});
  EXPECT_EQ(CopyToHost<float>(GetContext(), ScaledDotProductAttention(GetContext(), query, key, value, additive_mask)),
            (std::vector<float>{3.0F, 3.0F}));

  CopyFromHost<float>(GetContext(), additive_mask, {0.0F, std::numeric_limits<float>::quiet_NaN(), 0.0F});
  const auto nan_output =
      CopyToHost<float>(GetContext(), ScaledDotProductAttention(GetContext(), query, key, value, additive_mask));
  EXPECT_TRUE(std::isnan(nan_output[0]));
  EXPECT_TRUE(std::isnan(nan_output[1]));
}

TEST_F(AttentionTest, DistinguishesBooleanMaskingFromAdditiveIeeeArithmetic) {
  auto query = Empty(GetContext(), Shape{1, 1, 1, 1}, DType::FLOAT32);
  const auto key = Zeros(GetContext(), Shape{1, 1, 1, 1}, DType::FLOAT32);
  const auto value = Ones(GetContext(), Shape{1, 1, 1, 1}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), query, {std::numeric_limits<float>::quiet_NaN()});

  auto additive_mask = Empty(GetContext(), Shape{1}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), additive_mask, {-std::numeric_limits<float>::infinity()});
  const auto additive =
      CopyToHost<float>(GetContext(), ScaledDotProductAttention(GetContext(), query, key, value, additive_mask));
  ASSERT_EQ(additive.size(), 1);
  EXPECT_TRUE(std::isnan(additive[0]));

  auto boolean_mask = Empty(GetContext(), Shape{1}, DType::BOOL);
  CopyBoolFromHost(GetContext(), boolean_mask, {0});
  EXPECT_EQ(CopyToHost<float>(GetContext(), ScaledDotProductAttention(GetContext(), query, key, value, boolean_mask)),
            (std::vector<float>{0.0F}));
}

TEST_F(AttentionTest, ImplementsBothCausalAlignmentsForUnequalLengths) {
  const auto query = Zeros(GetContext(), Shape{1, 1, 2, 1}, DType::FLOAT32);
  const auto key = Zeros(GetContext(), Shape{1, 1, 4, 1}, DType::FLOAT32);
  auto value = Empty(GetContext(), Shape{1, 1, 4, 1}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), value, {1.0F, 2.0F, 3.0F, 4.0F});

  const auto upper = ScaledDotProductAttention(
      GetContext(), query, key, value, std::nullopt,
      SdpaOptions{.scale_ = std::nullopt, .causal_ = true, .causal_alignment_ = CausalAlignment::UPPER_LEFT});
  EXPECT_EQ(CopyToHost<float>(GetContext(), upper), (std::vector<float>{1.0F, 1.5F}));

  const auto lower = ScaledDotProductAttention(
      GetContext(), query, key, value, std::nullopt,
      SdpaOptions{.scale_ = std::nullopt, .causal_ = true, .causal_alignment_ = CausalAlignment::LOWER_RIGHT});
  EXPECT_EQ(CopyToHost<float>(GetContext(), lower), (std::vector<float>{2.0F, 2.5F}));
}

TEST_F(AttentionTest, SupportsGroupedQueryAttentionAndEveryFloatingDType) {
  for (const auto dtype : {DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto query = Zeros(GetContext(), Shape{1, 4, 1, 2}, dtype);
    const auto key = Zeros(GetContext(), Shape{1, 2, 2, 2}, dtype);
    auto source_value = Empty(GetContext(), Shape{1, 2, 2, 1}, DType::FLOAT32);
    CopyFromHost<float>(GetContext(), source_value, {1.0F, 3.0F, 10.0F, 14.0F});
    const auto value = Cast(GetContext(), source_value, dtype);
    const auto output = ScaledDotProductAttention(GetContext(), query, key, value);
    EXPECT_EQ(CopyAsFloat(GetContext(), output), (std::vector<float>{2.0F, 2.0F, 12.0F, 12.0F}));
  }
}

TEST_F(AttentionTest, SupportsStridedInputsOutputAndLargeValueDimension) {
  auto query_base = Zeros(GetContext(), Shape{1, 1, 1, 2}, DType::FLOAT32);
  const auto query = Slice(query_base, 3, 0, 2, 2);
  auto key_base = Zeros(GetContext(), Shape{1, 1, 3, 2}, DType::FLOAT32);
  const auto key = Slice(key_base, 3, 0, 2, 2);
  auto value_base = Empty(GetContext(), Shape{1, 1, 300, 3}, DType::FLOAT32);
  auto value_host = std::vector<float>(900);
  for (size_t value_index = 0; value_index < 300; ++value_index) {
    value_host[value_index * 3] = 1.0F;
    value_host[(value_index * 3) + 1] = -1.0F;
    value_host[(value_index * 3) + 2] = 9.0F;
  }
  CopyFromHost<float>(GetContext(), value_base, value_host);
  const auto value = Transpose(value_base, 2, 3);
  auto output = EmptyStrided(GetContext(), Shape{1, 1, 1, 300}, Strides{300, 300, 300, 1}, DType::FLOAT32);
  ScaledDotProductAttentionOut(GetContext(), output, query, key, value);
  const auto actual = CopyToHost<float>(GetContext(), output);
  ASSERT_EQ(actual.size(), 300);
  for (const auto element : actual) {
    EXPECT_FLOAT_EQ(element, 3.0F);
  }
}

TEST_F(AttentionTest, ReturnsZeroForEmptyKeySequence) {
  const auto query = Ones(GetContext(), Shape{1, 1, 2, 3}, DType::FLOAT32);
  const auto key = Empty(GetContext(), Shape{1, 1, 0, 3}, DType::FLOAT32);
  const auto value = Empty(GetContext(), Shape{1, 1, 0, 4}, DType::FLOAT32);
  EXPECT_EQ(CopyToHost<float>(GetContext(), ScaledDotProductAttention(GetContext(), query, key, value)),
            (std::vector<float>(8, 0.0F)));
}

TEST_F(AttentionTest, EnforcesShapeMaskScaleAndAliasContracts) {
  const auto query = Ones(GetContext(), Shape{1, 2, 3, 4}, DType::FLOAT32);
  const auto key = Ones(GetContext(), Shape{1, 1, 5, 4}, DType::FLOAT32);
  const auto value = Ones(GetContext(), Shape{1, 1, 5, 2}, DType::FLOAT32);
  EXPECT_THROW(
      [[maybe_unused]] const auto output = ScaledDotProductAttention(
          GetContext(), query, key, value, std::nullopt, SdpaOptions{.scale_ = std::numeric_limits<float>::infinity()}),
      InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto output =
                   ScaledDotProductAttention(GetContext(), query, Ones(GetContext(), Shape{1, 3, 5, 4}, DType::FLOAT32),
                                             Ones(GetContext(), Shape{1, 3, 5, 2}, DType::FLOAT32)),
               InvalidArgumentError);
  const auto wrong_mask = Ones(GetContext(), Shape{2, 4}, DType::BOOL);
  EXPECT_THROW(
      [[maybe_unused]] const auto output = ScaledDotProductAttention(GetContext(), query, key, value, wrong_mask),
      InvalidArgumentError);

  const auto alias_value = Ones(GetContext(), Shape{1, 1, 5, 4}, DType::FLOAT32);
  auto alias = query;
  EXPECT_THROW(ScaledDotProductAttentionOut(GetContext(), alias, query, key, alias_value), InvalidArgumentError);
  auto wrong_output = Empty(GetContext(), Shape{1, 2, 3, 3}, DType::FLOAT32);
  EXPECT_THROW(ScaledDotProductAttentionOut(GetContext(), wrong_output, query, key, value), InvalidArgumentError);

  const auto zero_dimension_query = Empty(GetContext(), Shape{1, 1, 2, 0}, DType::FLOAT32);
  const auto zero_dimension_key = Empty(GetContext(), Shape{1, 1, 3, 0}, DType::FLOAT32);
  const auto zero_dimension_value = Ones(GetContext(), Shape{1, 1, 3, 1}, DType::FLOAT32);
  EXPECT_THROW([[maybe_unused]] const auto output =
                   ScaledDotProductAttention(GetContext(), zero_dimension_query, zero_dimension_key,
                                             zero_dimension_value, std::nullopt, SdpaOptions{.scale_ = 1.0F}),
               InvalidArgumentError);
}

}  // namespace
}  // namespace ttl
