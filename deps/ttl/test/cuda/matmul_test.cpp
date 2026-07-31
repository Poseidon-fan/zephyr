#include "ttl/ops/matmul.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/elementwise.hpp"
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
  options.blas_workspace_bytes_ = 4U * 1024U * 1024U;
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

class MatmulTest : public testing::Test {
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

TEST_F(MatmulTest, ComputesTwoDimensionalMatmul) {
  auto lhs = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  auto rhs = Empty(GetContext(), Shape{3, 2}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), lhs, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  CopyFromHost<float>(GetContext(), rhs, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

  const auto output = Matmul(GetContext(), lhs, rhs, MatmulOptions{.allow_tf32_ = false});
  EXPECT_EQ(output.GetShape(), (Shape{2, 2}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), output), (std::vector<float>{58.0F, 64.0F, 139.0F, 154.0F}));
}

TEST_F(MatmulTest, SupportsEveryFloatingDTypeWithFloatAccumulation) {
  auto source_lhs = Empty(GetContext(), Shape{2, 4}, DType::FLOAT32);
  auto source_rhs = Empty(GetContext(), Shape{4, 3}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), source_lhs, {1.0F, 2.0F, 3.0F, 4.0F, 2.0F, 3.0F, 4.0F, 5.0F});
  CopyFromHost<float>(GetContext(), source_rhs,
                      {1.0F, 0.0F, 2.0F, 0.0F, 1.0F, 2.0F, 1.0F, 1.0F, 0.0F, 2.0F, 0.0F, 1.0F});
  const std::vector expected{12.0F, 5.0F, 10.0F, 16.0F, 7.0F, 15.0F};

  for (const auto dtype : {DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto lhs = Cast(GetContext(), source_lhs, dtype);
    const auto rhs = Cast(GetContext(), source_rhs, dtype);
    const auto actual = CopyAsFloat(GetContext(), Matmul(GetContext(), lhs, rhs));
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
      EXPECT_NEAR(actual[index], expected[index], 1.0e-2F);
    }
  }
}

TEST_F(MatmulTest, SupportsDirectTransposeViewsMaterializationAndStridedOutput) {
  auto lhs_base = Empty(GetContext(), Shape{3, 2}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), lhs_base, {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F});
  const auto lhs = Transpose(lhs_base, 0, 1);

  auto rhs_base = Empty(GetContext(), Shape{3, 4}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), rhs_base,
                      {1.0F, -1.0F, 2.0F, -1.0F, 3.0F, -1.0F, 4.0F, -1.0F, 5.0F, -1.0F, 6.0F, -1.0F});
  const auto rhs = Slice(rhs_base, 1, 0, 4, 2);
  auto output = EmptyStrided(GetContext(), Shape{2, 2}, Strides{1, 2}, DType::FLOAT32);
  MatmulOut(GetContext(), output, lhs, rhs, MatmulOptions{.allow_tf32_ = false});

  EXPECT_EQ(CopyToHost<float>(GetContext(), Contiguous(GetContext(), output)),
            (std::vector<float>{22.0F, 28.0F, 49.0F, 64.0F}));
}

TEST_F(MatmulTest, BroadcastsMultipleBatchDimensions) {
  const auto lhs = Ones(GetContext(), Shape{2, 1, 2, 3}, DType::FLOAT32);
  const auto rhs = Ones(GetContext(), Shape{1, 3, 3, 2}, DType::FLOAT32);
  const auto output = BatchedMatmul(GetContext(), lhs, rhs);
  EXPECT_EQ(output.GetShape(), (Shape{2, 3, 2, 2}));
  for (const auto value : CopyToHost<float>(GetContext(), output)) {
    EXPECT_FLOAT_EQ(value, 3.0F);
  }
}

TEST_F(MatmulTest, ComputesLinearWithBiasAndActivations) {
  auto input = Empty(GetContext(), Shape{2, 2, 3}, DType::FLOAT32);
  auto weight = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  auto bias = Empty(GetContext(), Shape{2}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), input, {1.0F, 2.0F, 3.0F, -1.0F, -2.0F, -3.0F, 2.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F});
  CopyFromHost<float>(GetContext(), weight, {1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F});
  CopyFromHost<float>(GetContext(), bias, {-5.0F, 1.0F});

  const auto relu =
      Linear(GetContext(), input, weight, bias,
             LinearOptions{.activation_ = LinearActivation::RELU, .matmul_ = MatmulOptions{.allow_tf32_ = false}});
  EXPECT_EQ(relu.GetShape(), (Shape{2, 2, 2}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), relu),
            (std::vector<float>{0.0F, 3.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 2.0F}));

  const auto gelu = Linear(GetContext(), input, weight, std::nullopt,
                           LinearOptions{.activation_ = LinearActivation::GELU,
                                         .gelu_approximation_ = GeluApproximation::NONE,
                                         .matmul_ = MatmulOptions{.allow_tf32_ = false}});
  const auto values = CopyToHost<float>(GetContext(), gelu);
  EXPECT_NEAR(values[0], 4.0F, 1.0e-3F);
  EXPECT_LT(values[2], 0.0F);
  EXPECT_NEAR(values[6], 0.0F, 1.0e-6F);
}

TEST_F(MatmulTest, HandlesZeroReductionAndEmptyOutput) {
  const auto lhs = Empty(GetContext(), Shape{2, 0}, DType::FLOAT32);
  const auto rhs = Empty(GetContext(), Shape{0, 3}, DType::FLOAT32);
  EXPECT_EQ(CopyToHost<float>(GetContext(), Matmul(GetContext(), lhs, rhs)),
            (std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F}));

  const auto empty_lhs = Empty(GetContext(), Shape{0, 3}, DType::FLOAT32);
  const auto nonempty_rhs = Ones(GetContext(), Shape{3, 2}, DType::FLOAT32);
  EXPECT_EQ(Matmul(GetContext(), empty_lhs, nonempty_rhs).GetNumElements(), 0);

  const auto linear_input = Empty(GetContext(), Shape{2, 0}, DType::FLOAT32);
  const auto linear_weight = Empty(GetContext(), Shape{3, 0}, DType::FLOAT32);
  auto bias = Empty(GetContext(), Shape{3}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), bias, {-1.0F, 2.0F, 3.0F});
  EXPECT_EQ(
      CopyToHost<float>(GetContext(), Linear(GetContext(), linear_input, linear_weight, bias,
                                             LinearOptions{.activation_ = LinearActivation::RELU, .matmul_ = {}})),
      (std::vector<float>{0.0F, 2.0F, 3.0F, 0.0F, 2.0F, 3.0F}));
}

TEST_F(MatmulTest, EnforcesSchemaAndAliasContracts) {
  const auto lhs = Ones(GetContext(), Shape{2, 3}, DType::FLOAT32);
  const auto rhs = Ones(GetContext(), Shape{3, 2}, DType::FLOAT32);
  EXPECT_THROW(
      [[maybe_unused]] const auto result = Matmul(GetContext(), Ones(GetContext(), Shape{2}, DType::FLOAT32), rhs),
      InvalidArgumentError);
  EXPECT_THROW(
      [[maybe_unused]] const auto result = Matmul(GetContext(), lhs, Ones(GetContext(), Shape{4, 2}, DType::FLOAT32)),
      InvalidArgumentError);
  EXPECT_THROW(
      [[maybe_unused]] const auto result = Matmul(GetContext(), lhs, Ones(GetContext(), Shape{3, 2}, DType::INT32)),
      InvalidArgumentError);

  auto wrong_output = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  EXPECT_THROW(MatmulOut(GetContext(), wrong_output, lhs, rhs), InvalidArgumentError);
  auto alias = lhs;
  EXPECT_THROW(MatmulOut(GetContext(), alias, lhs, rhs), InvalidArgumentError);

  const auto weight = Ones(GetContext(), Shape{4, 3}, DType::FLOAT32);
  const auto wrong_bias = Ones(GetContext(), Shape{3}, DType::FLOAT32);
  EXPECT_THROW([[maybe_unused]] const auto result = Linear(GetContext(), lhs, weight, wrong_bias),
               InvalidArgumentError);
}

}  // namespace
}  // namespace ttl
