#include "ttl/ops/topk.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
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

class TopKTest : public testing::Test {
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

TEST_F(TopKTest, SelectsLargestAndSmallestWithStableTies) {
  auto input = Empty(GetContext(), Shape{2, 6}, DType::INT32);
  CopyFromHost<int32_t>(GetContext(), input, {3, 5, 5, -1, 2, 5, 7, 1, 7, 2, -3, 4});

  const auto [largest_values, largest_indices] =
      TopK(GetContext(), input, TopKOptions{.axis_ = -1, .k_ = 3, .largest_ = true, .sorted_ = true});
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), largest_values), (std::vector<int32_t>{5, 5, 5, 7, 7, 4}));
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), largest_indices), (std::vector<int64_t>{1, 2, 5, 0, 2, 5}));

  const auto [smallest_values, smallest_indices] =
      TopK(GetContext(), input, TopKOptions{.axis_ = 1, .k_ = 2, .largest_ = false, .sorted_ = true});
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), smallest_values), (std::vector<int32_t>{-1, 2, -3, 1}));
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), smallest_indices), (std::vector<int64_t>{3, 4, 4, 1}));
}

TEST_F(TopKTest, DefinesNanAndSignedZeroTotalOrder) {
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  auto input = Empty(GetContext(), Shape{6}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), input, {-0.0F, 1.0F, nan, 0.0F, nan, -2.0F});

  const auto [largest_values, largest_indices] =
      TopK(GetContext(), input, TopKOptions{.k_ = 4, .largest_ = true, .sorted_ = true});
  const auto values = CopyToHost<float>(GetContext(), largest_values);
  ASSERT_EQ(values.size(), 4);
  EXPECT_TRUE(std::isnan(values[0]));
  EXPECT_TRUE(std::isnan(values[1]));
  EXPECT_FLOAT_EQ(values[2], 1.0F);
  EXPECT_FLOAT_EQ(values[3], -0.0F);
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), largest_indices), (std::vector<int64_t>{2, 4, 1, 0}));

  const auto [smallest_values, smallest_indices] =
      TopK(GetContext(), input, TopKOptions{.k_ = 4, .largest_ = false, .sorted_ = true});
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), smallest_indices), (std::vector<int64_t>{5, 0, 3, 1}));
  const auto smallest = CopyToHost<float>(GetContext(), smallest_values);
  EXPECT_FLOAT_EQ(smallest[0], -2.0F);
  EXPECT_FLOAT_EQ(smallest[1], -0.0F);
  EXPECT_FLOAT_EQ(smallest[2], 0.0F);
  EXPECT_FLOAT_EQ(smallest[3], 1.0F);
}

TEST_F(TopKTest, SupportsEveryNumericDType) {
  auto source = Empty(GetContext(), Shape{2, 5}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), source, {1.0F, 5.0F, 3.0F, 4.0F, 2.0F, 2.0F, 0.0F, 4.0F, 3.0F, 1.0F});
  for (const auto dtype : {DType::UINT8, DType::INT32, DType::INT64, DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto input = Cast(GetContext(), source, dtype);
    const auto [values, indices] = TopK(GetContext(), input, TopKOptions{.k_ = 2});
    EXPECT_EQ(CopyAsFloat(GetContext(), values), (std::vector<float>{5.0F, 4.0F, 4.0F, 3.0F}));
    EXPECT_EQ(CopyToHost<int64_t>(GetContext(), indices), (std::vector<int64_t>{1, 3, 2, 3}));
  }
}

TEST_F(TopKTest, SupportsArbitraryAxisInputAndOutputLayouts) {
  auto base = Empty(GetContext(), Shape{3, 2}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), base, {1.0F, 6.0F, 5.0F, 2.0F, 3.0F, 4.0F});
  const auto input = Transpose(base, 0, 1);
  auto values = EmptyStrided(GetContext(), Shape{2, 2}, Strides{1, 2}, DType::FLOAT32);
  auto indices = EmptyStrided(GetContext(), Shape{2, 2}, Strides{1, 2}, DType::INT64);
  TopKOut(GetContext(), values, indices, input, TopKOptions{.axis_ = 1, .k_ = 2});

  EXPECT_EQ(CopyToHost<float>(GetContext(), Contiguous(GetContext(), values)),
            (std::vector<float>{5.0F, 3.0F, 6.0F, 4.0F}));
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), Contiguous(GetContext(), indices)), (std::vector<int64_t>{1, 2, 0, 2}));
}

TEST_F(TopKTest, UsesSegmentedRadixSortForLargeAxes) {
  constexpr auto axis_size = int64_t{2048};
  auto host = std::vector<int64_t>(static_cast<size_t>(axis_size));
  for (int64_t index = 0; index < axis_size; ++index) {
    host[static_cast<size_t>(index)] = (index * 17) % axis_size;
  }
  auto input = Empty(GetContext(), Shape{2, axis_size}, DType::INT64);
  auto doubled = host;
  doubled.insert(doubled.end(), host.begin(), host.end());
  CopyFromHost<int64_t>(GetContext(), input, doubled);

  const auto [values, indices] = TopK(GetContext(), input, TopKOptions{.k_ = 3});
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), values), (std::vector<int64_t>{2047, 2046, 2045, 2047, 2046, 2045}));
  const auto actual_indices = CopyToHost<int64_t>(GetContext(), indices);
  ASSERT_EQ(actual_indices.size(), 6);
  for (size_t row = 0; row < 2; ++row) {
    EXPECT_EQ(host[static_cast<size_t>(actual_indices[row * 3])], 2047);
    EXPECT_EQ(host[static_cast<size_t>(actual_indices[(row * 3) + 1])], 2046);
    EXPECT_EQ(host[static_cast<size_t>(actual_indices[(row * 3) + 2])], 2045);
  }
}

TEST_F(TopKTest, HandlesEmptyResultsAndEnforcesContracts) {
  const auto input = Ones(GetContext(), Shape{2, 3}, DType::FLOAT32);
  const auto [empty_values, empty_indices] = TopK(GetContext(), input, TopKOptions{.k_ = 0});
  EXPECT_EQ(empty_values.GetShape(), (Shape{2, 0}));
  EXPECT_EQ(empty_indices.GetNumElements(), 0);

  const auto outer_empty = Empty(GetContext(), Shape{0, 3}, DType::FLOAT32);
  EXPECT_EQ(TopK(GetContext(), outer_empty, TopKOptions{.k_ = 2}).first.GetNumElements(), 0);
  EXPECT_THROW([[maybe_unused]] const auto result = TopK(GetContext(), input, TopKOptions{.k_ = 4}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = TopK(GetContext(), input, TopKOptions{.axis_ = 2, .k_ = 1}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result =
                   TopK(GetContext(), Ones(GetContext(), Shape{3}, DType::BOOL), TopKOptions{.k_ = 1}),
               NotSupportedError);

  auto wrong_values = Empty(GetContext(), Shape{2, 1}, DType::FLOAT32);
  auto wrong_indices = Empty(GetContext(), Shape{2, 2}, DType::INT32);
  EXPECT_THROW(TopKOut(GetContext(), wrong_values, wrong_indices, input, TopKOptions{.k_ = 1}), InvalidArgumentError);
  auto alias = input;
  auto indices = Empty(GetContext(), Shape{2, 3}, DType::INT64);
  EXPECT_THROW(TopKOut(GetContext(), alias, indices, input, TopKOptions{.k_ = 3}), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl
