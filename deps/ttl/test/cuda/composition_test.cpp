#include "ttl/ops/composition.hpp"

#include <array>
#include <atomic>
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
#include "ttl/ops/copy.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/runtime.hpp"
#include "ttl/scalar.hpp"
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

class CompositionTest : public testing::Test {
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

TEST_F(CompositionTest, ConcatenatesAlongArbitraryAxes) {
  auto first = Empty(GetContext(), Shape{2, 2}, DType::INT32);
  auto second = Empty(GetContext(), Shape{2, 1}, DType::INT32);
  CopyFromHost<int32_t>(GetContext(), first, {1, 2, 3, 4});
  CopyFromHost<int32_t>(GetContext(), second, {5, 6});
  const std::array inputs{first, second};
  const auto output = Concat(GetContext(), inputs, -1);
  EXPECT_EQ(output.GetShape(), (Shape{2, 3}));
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), output), (std::vector<int32_t>{1, 2, 5, 3, 4, 6}));

  const std::array vertical_inputs{first, first};
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), Concat(GetContext(), vertical_inputs, 0)),
            (std::vector<int32_t>{1, 2, 3, 4, 1, 2, 3, 4}));
}

TEST_F(CompositionTest, StacksTensorsAndScalars) {
  auto first = Empty(GetContext(), Shape{2}, DType::FLOAT32);
  auto second = Empty(GetContext(), Shape{2}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), first, {1.0F, 2.0F});
  CopyFromHost<float>(GetContext(), second, {3.0F, 4.0F});
  const std::array inputs{first, second};
  const auto output = Stack(GetContext(), inputs, 1);
  EXPECT_EQ(output.GetShape(), (Shape{2, 2}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), output), (std::vector<float>{1.0F, 3.0F, 2.0F, 4.0F}));

  const auto scalar_a = Full(GetContext(), Shape{}, Scalar{int64_t{5}}, DType::INT64);
  const auto scalar_b = Full(GetContext(), Shape{}, Scalar{int64_t{7}}, DType::INT64);
  const std::array scalars{scalar_a, scalar_b};
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), Stack(GetContext(), scalars, -1)), (std::vector<int64_t>{5, 7}));
}

TEST_F(CompositionTest, SupportsStridedInputsAndOutputs) {
  auto base = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), base, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const auto first = Transpose(base, 0, 1);
  const auto second = Transpose(base, 0, 1);
  const std::array inputs{first, second};
  auto output = EmptyStrided(GetContext(), Shape{3, 4}, Strides{1, 3}, DType::FLOAT32);
  ConcatOut(GetContext(), output, inputs, 1);
  EXPECT_EQ(CopyToHost<float>(GetContext(), Contiguous(GetContext(), output)),
            (std::vector<float>{1.0F, 4.0F, 1.0F, 4.0F, 2.0F, 5.0F, 2.0F, 5.0F, 3.0F, 6.0F, 3.0F, 6.0F}));
}

TEST_F(CompositionTest, HandlesEmptyInputsWithoutLaunchingCopies) {
  const auto first = Empty(GetContext(), Shape{2, 0}, DType::FLOAT32);
  const auto second = Empty(GetContext(), Shape{2, 0}, DType::FLOAT32);
  const std::array inputs{first, second};
  const auto concatenated = Concat(GetContext(), inputs, 1);
  EXPECT_EQ(concatenated.GetShape(), (Shape{2, 0}));
  EXPECT_EQ(concatenated.GetNumElements(), 0);

  const auto stacked = Stack(GetContext(), inputs, 0);
  EXPECT_EQ(stacked.GetShape(), (Shape{2, 2, 0}));
  EXPECT_EQ(stacked.GetNumElements(), 0);
}

TEST_F(CompositionTest, EnforcesInputOutputAndAliasContractsBeforeCopying) {
  const std::vector<Tensor> empty_inputs;
  EXPECT_THROW([[maybe_unused]] const auto result = Concat(GetContext(), empty_inputs, 0), InvalidArgumentError);

  const auto first = Ones(GetContext(), Shape{2, 2}, DType::FLOAT32);
  const auto wrong_shape = Ones(GetContext(), Shape{3, 1}, DType::FLOAT32);
  const std::array shape_mismatch{first, wrong_shape};
  EXPECT_THROW([[maybe_unused]] const auto result = Concat(GetContext(), shape_mismatch, 1), InvalidArgumentError);

  const auto wrong_dtype = Ones(GetContext(), Shape{2, 2}, DType::INT32);
  const std::array dtype_mismatch{first, wrong_dtype};
  EXPECT_THROW([[maybe_unused]] const auto result = Stack(GetContext(), dtype_mismatch, 0), InvalidArgumentError);

  const std::array valid{first, first};
  auto wrong_output = Empty(GetContext(), Shape{4}, DType::FLOAT32);
  EXPECT_THROW(ConcatOut(GetContext(), wrong_output, valid, 0), InvalidArgumentError);
  auto alias = first;
  EXPECT_THROW(ConcatOut(GetContext(), alias, valid, 0), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = Stack(GetContext(), valid, 3), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl
