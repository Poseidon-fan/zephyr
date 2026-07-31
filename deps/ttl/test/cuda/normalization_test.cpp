#include "ttl/ops/normalization.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <vector>

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/internal/execution_lane.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/layout.hpp"
#include "ttl/ops/cast.hpp"
#include "ttl/ops/copy.hpp"
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

class NormalizationTest : public testing::Test {
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

TEST_F(NormalizationTest, ComputesLayerNormWithAffineParameters) {
  auto input = Empty(GetContext(), Shape{2, 4}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), input, {1.0F, 2.0F, 3.0F, 4.0F, 2.0F, 2.0F, 2.0F, 2.0F});
  auto weight = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  auto bias = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), weight, {1.0F, 2.0F, 3.0F, 4.0F});
  CopyFromHost<float>(GetContext(), bias, {0.5F, 0.5F, 0.5F, 0.5F});

  const auto output =
      CopyToHost<float>(GetContext(), LayerNorm(GetContext(), input, weight, bias, NormOptions{.normalized_rank_ = 1}));
  const auto inverse_standard_deviation = 1.0F / std::sqrt(1.25F + 1.0e-5F);
  EXPECT_NEAR(output[0], (-1.5F * inverse_standard_deviation) + 0.5F, 1.0e-5F);
  EXPECT_NEAR(output[1], (-0.5F * inverse_standard_deviation * 2.0F) + 0.5F, 1.0e-5F);
  EXPECT_NEAR(output[2], (0.5F * inverse_standard_deviation * 3.0F) + 0.5F, 1.0e-5F);
  EXPECT_NEAR(output[3], (1.5F * inverse_standard_deviation * 4.0F) + 0.5F, 1.0e-5F);
  for (size_t index = 4; index < output.size(); ++index) {
    EXPECT_FLOAT_EQ(output[index], 0.5F);
  }
}

TEST_F(NormalizationTest, ComputesStableRmsNormForLargeValues) {
  auto input = Empty(GetContext(), Shape{2, 2}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), input, {3.0F, 4.0F, 1.0e30F, 1.0e30F});
  const auto output = CopyToHost<float>(
      GetContext(), RmsNorm(GetContext(), input, std::nullopt, NormOptions{.normalized_rank_ = 1, .epsilon_ = 0.0F}));
  const auto rms = std::sqrt(12.5F);
  EXPECT_NEAR(output[0], 3.0F / rms, 1.0e-6F);
  EXPECT_NEAR(output[1], 4.0F / rms, 1.0e-6F);
  EXPECT_TRUE(std::isfinite(output[2]));
  EXPECT_TRUE(std::isfinite(output[3]));
  EXPECT_NEAR(output[2], 1.0F, 1.0e-6F);
  EXPECT_NEAR(output[3], 1.0F, 1.0e-6F);
}

TEST_F(NormalizationTest, SupportsEveryFloatingDTypeAndMultipleNormalizedDimensions) {
  for (const auto dtype : {DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto input = Ones(GetContext(), Shape{2, 2, 3}, dtype);
    const auto layer = CopyAsFloat(
        GetContext(), LayerNorm(GetContext(), input, std::nullopt, std::nullopt, NormOptions{.normalized_rank_ = 2}));
    const auto rms =
        CopyAsFloat(GetContext(), RmsNorm(GetContext(), input, std::nullopt, NormOptions{.normalized_rank_ = 2}));
    for (size_t index = 0; index < layer.size(); ++index) {
      EXPECT_NEAR(layer[index], 0.0F, 1.0e-5F);
      EXPECT_NEAR(rms[index], 1.0F, 1.0e-3F);
    }
  }
}

TEST_F(NormalizationTest, SupportsStridedInputOutputAndAffineLayouts) {
  auto base = Empty(GetContext(), Shape{3, 2}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), base, {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F});
  const auto input = Transpose(base, 0, 1);
  auto output = EmptyStrided(GetContext(), Shape{2, 3}, Strides{1, 2}, DType::FLOAT32);
  LayerNormOut(GetContext(), output, input, std::nullopt, std::nullopt, NormOptions{.normalized_rank_ = 1});
  const auto values = CopyToHost<float>(GetContext(), Contiguous(GetContext(), output));
  for (size_t row = 0; row < 2; ++row) {
    const auto mean = (values[row * 3] + values[(row * 3) + 1] + values[(row * 3) + 2]) / 3.0F;
    EXPECT_NEAR(mean, 0.0F, 1.0e-6F);
  }
}

TEST_F(NormalizationTest, EnforcesEmptyAndSchemaContracts) {
  const auto outer_empty = Empty(GetContext(), Shape{0, 3}, DType::FLOAT32);
  EXPECT_EQ(LayerNorm(GetContext(), outer_empty, std::nullopt, std::nullopt, NormOptions{.normalized_rank_ = 1})
                .GetNumElements(),
            0);
  const auto group_empty = Empty(GetContext(), Shape{2, 0}, DType::FLOAT32);
  EXPECT_THROW([[maybe_unused]] const auto result =
                   RmsNorm(GetContext(), group_empty, std::nullopt, NormOptions{.normalized_rank_ = 1}),
               InvalidArgumentError);

  const auto input = Ones(GetContext(), Shape{2, 3}, DType::FLOAT32);
  EXPECT_THROW([[maybe_unused]] const auto result =
                   LayerNorm(GetContext(), input, std::nullopt, std::nullopt, NormOptions{.normalized_rank_ = 0}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result =
                   LayerNorm(GetContext(), input, std::nullopt, std::nullopt, NormOptions{.normalized_rank_ = 3}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = LayerNorm(GetContext(), input, std::nullopt, std::nullopt,
                                                              NormOptions{.normalized_rank_ = 1, .epsilon_ = -1.0F}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = RmsNorm(GetContext(), Ones(GetContext(), Shape{2, 3}, DType::INT32),
                                                            std::nullopt, NormOptions{.normalized_rank_ = 1}),
               NotSupportedError);

  const auto wrong_weight = Ones(GetContext(), Shape{2}, DType::FLOAT32);
  EXPECT_THROW([[maybe_unused]] const auto result =
                   LayerNorm(GetContext(), input, wrong_weight, std::nullopt, NormOptions{.normalized_rank_ = 1}),
               InvalidArgumentError);
  auto alias = input;
  EXPECT_THROW(LayerNormOut(GetContext(), alias, input, std::nullopt, std::nullopt, NormOptions{.normalized_rank_ = 1}),
               InvalidArgumentError);
}

TEST_F(NormalizationTest, UsesTwoStageScratchForLargeGroups) {
  constexpr auto element_count = int64_t{1} << 20;
  const auto input = Ones(GetContext(), Shape{element_count}, DType::FLOAT32);
  const auto output = RmsNorm(GetContext(), input, std::nullopt, NormOptions{.normalized_rank_ = 1});
  const auto first = CopyToHost<float>(GetContext(), Narrow(output, 0, 0, 1));
  ASSERT_EQ(first.size(), 1);
  EXPECT_NEAR(first[0], 1.0F, 1.0e-5F);
  const auto &lane = internal::ContextAccess::GetPrimaryLane(GetContext(), std::source_location::current());
  EXPECT_GT(lane.GetScratchHighWaterBytes(), 0);
}

}  // namespace
}  // namespace ttl
