#include "ttl/ops/random.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
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
#include "ttl/generator.hpp"
#include "ttl/ops/cast.hpp"
#include "ttl/ops/copy.hpp"
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

class RandomTest : public testing::Test {
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

TEST_F(RandomTest, ReproducesSequencesAndResetsSeed) {
  Generator first{GetContext(), 123456789};
  Generator second{GetContext(), 123456789};
  const auto first_values = CopyToHost<float>(GetContext(), Uniform(GetContext(), Shape{17}, DType::FLOAT32, first));
  const auto second_values = CopyToHost<float>(GetContext(), Uniform(GetContext(), Shape{17}, DType::FLOAT32, second));
  EXPECT_EQ(first_values, second_values);

  const auto continued = CopyToHost<float>(GetContext(), Uniform(GetContext(), Shape{17}, DType::FLOAT32, first));
  EXPECT_NE(first_values, continued);
  first.SetSeed(GetContext(), 123456789);
  EXPECT_EQ(first.GetSeed(), 123456789);
  EXPECT_EQ(CopyToHost<float>(GetContext(), Uniform(GetContext(), Shape{17}, DType::FLOAT32, first)), first_values);
}

TEST_F(RandomTest, GeneratesBoundedUniformForAllFloatingDTypesAndStridedOutput) {
  for (const auto dtype : {DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    Generator generator{GetContext(), 42};
    auto output = EmptyStrided(GetContext(), Shape{31, 7}, Strides{1, 31}, dtype);
    UniformOut(GetContext(), output, generator, UniformOptions{.low_ = -3.0, .high_ = 2.0});
    const auto values = CopyAsFloat(GetContext(), Contiguous(GetContext(), output));
    for (const auto value : values) {
      EXPECT_GE(value, -3.0F);
      EXPECT_LT(value, 2.0F);
    }
  }
}

TEST_F(RandomTest, GeneratesNormalDistributionWithFixedBoxMullerMapping) {
  Generator generator{GetContext(), 987654321};
  const auto values = CopyToHost<float>(GetContext(), Normal(GetContext(), Shape{100000}, DType::FLOAT32, generator,
                                                             NormalOptions{.mean_ = 2.0, .standard_deviation_ = 3.0}));
  const auto sum = std::accumulate(values.begin(), values.end(), 0.0);
  const auto mean = sum / static_cast<double>(values.size());
  auto squared_deviation = 0.0;
  for (const auto value : values) {
    const auto deviation = static_cast<double>(value) - mean;
    squared_deviation += deviation * deviation;
  }
  const auto standard_deviation = std::sqrt(squared_deviation / static_cast<double>(values.size()));
  EXPECT_NEAR(mean, 2.0, 0.04);
  EXPECT_NEAR(standard_deviation, 3.0, 0.04);

  const auto constant =
      CopyToHost<float>(GetContext(), Normal(GetContext(), Shape{11}, DType::FLOAT32, generator,
                                             NormalOptions{.mean_ = -4.0, .standard_deviation_ = 0.0}));
  for (const auto value : constant) {
    EXPECT_FLOAT_EQ(value, -4.0F);
  }
}

TEST_F(RandomTest, EmptyOutputDoesNotConsumeCounter) {
  Generator first{GetContext(), 7};
  Generator second{GetContext(), 7};
  EXPECT_EQ(Uniform(GetContext(), Shape{0, 3}, DType::FLOAT32, first).GetNumElements(), 0);
  EXPECT_EQ(CopyToHost<float>(GetContext(), Uniform(GetContext(), Shape{9}, DType::FLOAT32, first)),
            CopyToHost<float>(GetContext(), Uniform(GetContext(), Shape{9}, DType::FLOAT32, second)));
}

TEST_F(RandomTest, EnforcesGeneratorDTypeAndDistributionContracts) {
  Generator generator{GetContext(), 1};
  EXPECT_THROW([[maybe_unused]] const auto output = Uniform(GetContext(), Shape{4}, DType::INT32, generator),
               NotSupportedError);
  EXPECT_THROW([[maybe_unused]] const auto output = Uniform(GetContext(), Shape{4}, DType::FLOAT32, generator,
                                                            UniformOptions{.low_ = 1.0, .high_ = 1.0}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto output =
                   Uniform(GetContext(), Shape{4}, DType::FLOAT32, generator,
                           UniformOptions{.low_ = 0.0, .high_ = std::numeric_limits<double>::infinity()}),
               InvalidArgumentError);
  EXPECT_THROW(
      [[maybe_unused]] const auto output = Uniform(
          GetContext(), Shape{4}, DType::FLOAT32, generator,
          UniformOptions{.low_ = -std::numeric_limits<double>::max(), .high_ = std::numeric_limits<double>::max()}),
      InvalidArgumentError);
  EXPECT_THROW(
      [[maybe_unused]] const auto output = Uniform(GetContext(), Shape{4}, DType::FLOAT32, generator,
                                                   UniformOptions{.low_ = 1.0, .high_ = std::nextafter(1.0, 2.0)}),
      InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto output = Normal(GetContext(), Shape{4}, DType::FLOAT32, generator,
                                                           NormalOptions{.mean_ = 0.0, .standard_deviation_ = -1.0}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto output =
                   Normal(GetContext(), Shape{4}, DType::FLOAT32, generator,
                          NormalOptions{.mean_ = std::numeric_limits<double>::max(), .standard_deviation_ = 1.0}),
               InvalidArgumentError);

  auto other_context = runtime_->CreateExecutionContext(Device{0});
  EXPECT_THROW([[maybe_unused]] const auto output = Uniform(other_context, Shape{4}, DType::FLOAT32, generator),
               InvalidArgumentError);
}

}  // namespace
}  // namespace ttl
