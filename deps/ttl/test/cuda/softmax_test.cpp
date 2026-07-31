#include "ttl/ops/softmax.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
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
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/reduction.hpp"
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

class SoftmaxTest : public testing::Test {
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

TEST_F(SoftmaxTest, ComputesStableSoftmaxAndLogSoftmax) {
  auto input = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), input, {1.0F, 2.0F, 3.0F, 1001.0F, 1002.0F, 1003.0F});
  const auto options = SoftmaxOptions{.axes_ = {-1}};
  const auto softmax = CopyToHost<float>(GetContext(), Softmax(GetContext(), input, options));
  const auto log_softmax = CopyToHost<float>(GetContext(), LogSoftmax(GetContext(), input, options));

  constexpr auto e = std::numbers::e_v<float>;
  constexpr auto e_squared = e * e;
  constexpr float denominator = 1.0F + e + e_squared;
  for (size_t row = 0; row < 2; ++row) {
    EXPECT_NEAR(softmax[row * 3], 1.0F / denominator, 1.0e-6F);
    EXPECT_NEAR(softmax[(row * 3) + 1], e / denominator, 1.0e-6F);
    EXPECT_NEAR(softmax[(row * 3) + 2], e_squared / denominator, 1.0e-6F);
    for (size_t column = 0; column < 3; ++column) {
      EXPECT_NEAR(std::exp(log_softmax[(row * 3) + column]), softmax[(row * 3) + column], 1.0e-6F);
    }
  }
}

TEST_F(SoftmaxTest, SupportsEveryFloatingDTypeAndArbitraryAxes) {
  for (const auto dtype : {DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto input = Ones(GetContext(), Shape{2, 3, 4}, dtype);
    const auto output = Softmax(GetContext(), input, SoftmaxOptions{.axes_ = {0, -1}});
    const auto values = CopyAsFloat(GetContext(), output);
    for (const auto value : values) {
      EXPECT_NEAR(value, 0.125F, 5.0e-4F);
    }
  }
}

TEST_F(SoftmaxTest, SupportsStridedInputsAndOutputs) {
  auto base = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), base, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const auto input = Transpose(base, 0, 1);
  auto output = EmptyStrided(GetContext(), Shape{3, 2}, Strides{1, 3}, DType::FLOAT32);
  SoftmaxOut(GetContext(), output, input, SoftmaxOptions{.axes_ = {0}});
  const auto values = CopyToHost<float>(GetContext(), Contiguous(GetContext(), output));
  for (size_t column = 0; column < 2; ++column) {
    EXPECT_NEAR(values[column] + values[2 + column] + values[4 + column], 1.0F, 1.0e-6F);
  }
}

TEST_F(SoftmaxTest, DefinesNanInfinityAndEmptySemantics) {
  auto input = Empty(GetContext(), Shape{3, 2}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), input, {NAN, 1.0F, -INFINITY, -INFINITY, INFINITY, 1.0F});
  const auto output = CopyToHost<float>(GetContext(), Softmax(GetContext(), input, SoftmaxOptions{.axes_ = {1}}));
  EXPECT_TRUE(std::isnan(output[0]));
  EXPECT_TRUE(std::isnan(output[1]));
  EXPECT_TRUE(std::isnan(output[2]));
  EXPECT_TRUE(std::isnan(output[3]));
  EXPECT_TRUE(std::isnan(output[4]));
  EXPECT_TRUE(std::isnan(output[5]));

  const auto empty = Empty(GetContext(), Shape{2, 0, 3}, DType::FLOAT32);
  const auto empty_output = LogSoftmax(GetContext(), empty, SoftmaxOptions{.axes_ = {1}});
  EXPECT_EQ(empty_output.GetShape(), empty.GetShape());
  EXPECT_EQ(empty_output.GetNumElements(), 0);
}

TEST_F(SoftmaxTest, UsesTwoStageScratchForLargeGroups) {
  constexpr auto element_count = int64_t{1} << 20;
  const auto input = Zeros(GetContext(), Shape{element_count}, DType::FLOAT32);
  const auto output = Softmax(GetContext(), input, SoftmaxOptions{.axes_ = {0}});
  const auto total = CopyToHost<float>(GetContext(), Sum(GetContext(), output));
  ASSERT_EQ(total.size(), 1);
  EXPECT_NEAR(total[0], 1.0F, 1.0e-5F);
}

TEST_F(SoftmaxTest, EnforcesSchemaAndAliasContracts) {
  const auto input = Ones(GetContext(), Shape{2, 3}, DType::FLOAT32);
  EXPECT_THROW([[maybe_unused]] const auto result = Softmax(GetContext(), input, SoftmaxOptions{}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = Softmax(GetContext(), input, SoftmaxOptions{.axes_ = {0, -2}}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result =
                   Softmax(GetContext(), Ones(GetContext(), Shape{2, 3}, DType::INT32), SoftmaxOptions{.axes_ = {1}}),
               NotSupportedError);

  auto wrong_shape = Empty(GetContext(), Shape{6}, DType::FLOAT32);
  EXPECT_THROW(SoftmaxOut(GetContext(), wrong_shape, input, SoftmaxOptions{.axes_ = {1}}), InvalidArgumentError);
  auto wrong_dtype = Empty(GetContext(), Shape{2, 3}, DType::FLOAT16);
  EXPECT_THROW(SoftmaxOut(GetContext(), wrong_dtype, input, SoftmaxOptions{.axes_ = {1}}), InvalidArgumentError);
  auto alias = input;
  EXPECT_THROW(SoftmaxOut(GetContext(), alias, input, SoftmaxOptions{.axes_ = {1}}), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl
