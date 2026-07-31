#include "ttl/ops/indexing.hpp"

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

class IndexingTest : public testing::Test {
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

TEST_F(IndexingTest, IndexSelectSupportsEveryValueAndIndexDType) {
  auto source = Empty(GetContext(), Shape{3, 2}, DType::INT32);
  CopyFromHost<int32_t>(GetContext(), source, {10, 11, 20, 21, 30, 31});
  for (const auto value_dtype :
       {DType::BOOL, DType::UINT8, DType::INT32, DType::INT64, DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto input = Cast(GetContext(), source, value_dtype);
    for (const auto index_dtype : {DType::INT32, DType::INT64}) {
      auto raw_index = Empty(GetContext(), Shape{2}, DType::INT64);
      CopyFromHost<int64_t>(GetContext(), raw_index, {2, 0});
      const auto index = Cast(GetContext(), raw_index, index_dtype);
      const auto output = Cast(GetContext(), IndexSelect(GetContext(), input, 0, index), DType::INT32);
      const auto values = CopyToHost<int32_t>(GetContext(), output);
      if (value_dtype == DType::BOOL) {
        EXPECT_EQ(values, (std::vector<int32_t>{1, 1, 1, 1}));
      } else {
        EXPECT_EQ(values, (std::vector<int32_t>{30, 31, 10, 11}));
      }
    }
  }
}

TEST_F(IndexingTest, ImplementsGatherAndTakeAlongBroadcasting) {
  auto input = Empty(GetContext(), Shape{2, 3}, DType::INT32);
  CopyFromHost<int32_t>(GetContext(), input, {10, 11, 12, 20, 21, 22});
  auto gather_index = Empty(GetContext(), Shape{2, 2}, DType::INT64);
  CopyFromHost<int64_t>(GetContext(), gather_index, {2, 0, 1, 1});
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), Gather(GetContext(), input, 1, gather_index)),
            (std::vector<int32_t>{12, 10, 21, 21}));

  auto take_index = Empty(GetContext(), Shape{1, 2}, DType::INT32);
  CopyFromHost<int32_t>(GetContext(), take_index, {2, 0});
  const auto taken = TakeAlongDimension(GetContext(), input, take_index, -1);
  EXPECT_EQ(taken.GetShape(), (Shape{2, 2}));
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), taken), (std::vector<int32_t>{12, 10, 22, 20}));
}

TEST_F(IndexingTest, ImplementsGatherRowsAndEmbedding) {
  auto table = Empty(GetContext(), Shape{4, 3}, DType::FLOAT32);
  CopyFromHost<float>(GetContext(), table,
                      {0.0F, 1.0F, 2.0F, 10.0F, 11.0F, 12.0F, 20.0F, 21.0F, 22.0F, 30.0F, 31.0F, 32.0F});
  auto indices = Empty(GetContext(), Shape{2, 2}, DType::INT64);
  CopyFromHost<int64_t>(GetContext(), indices, {3, 1, 0, 2});
  const auto gathered = GatherRows(GetContext(), table, indices);
  EXPECT_EQ(gathered.GetShape(), (Shape{2, 2, 3}));
  const auto expected =
      std::vector<float>{30.0F, 31.0F, 32.0F, 10.0F, 11.0F, 12.0F, 0.0F, 1.0F, 2.0F, 20.0F, 21.0F, 22.0F};
  EXPECT_EQ(CopyToHost<float>(GetContext(), gathered), expected);
  EXPECT_EQ(CopyToHost<float>(GetContext(), Embedding(GetContext(), table, indices)), expected);
}

TEST_F(IndexingTest, SupportsStridedInputsIndicesAndOutputs) {
  auto base = Empty(GetContext(), Shape{3, 2}, DType::INT32);
  CopyFromHost<int32_t>(GetContext(), base, {10, 20, 11, 21, 12, 22});
  const auto input = Transpose(base, 0, 1);
  auto index_base = Empty(GetContext(), Shape{2, 2}, DType::INT64);
  CopyFromHost<int64_t>(GetContext(), index_base, {0, 1, 2, 0});
  const auto index = Transpose(index_base, 0, 1);
  auto output = EmptyStrided(GetContext(), Shape{2, 2}, Strides{1, 2}, DType::INT32);
  GatherOut(GetContext(), output, input, 1, index);
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), Contiguous(GetContext(), output)),
            (std::vector<int32_t>{10, 12, 21, 20}));
}

TEST_F(IndexingTest, ReportsBoundsErrorsWithoutIllegalAccessAndRecovers) {
  const auto input = Ones(GetContext(), Shape{2, 3}, DType::FLOAT32);
  auto negative = Empty(GetContext(), Shape{1}, DType::INT64);
  CopyFromHost<int64_t>(GetContext(), negative, {-1});
  [[maybe_unused]] const auto invalid_negative = IndexSelect(GetContext(), input, 1, negative);
  EXPECT_THROW(GetContext().Synchronize(), DeviceError);

  auto upper_bound = Empty(GetContext(), Shape{1}, DType::INT32);
  CopyFromHost<int32_t>(GetContext(), upper_bound, {3});
  [[maybe_unused]] const auto invalid_upper = IndexSelect(GetContext(), input, 1, upper_bound);
  EXPECT_THROW(GetContext().Synchronize(), DeviceError);

  auto valid = Empty(GetContext(), Shape{1}, DType::INT32);
  CopyFromHost<int32_t>(GetContext(), valid, {2});
  EXPECT_EQ(CopyToHost<float>(GetContext(), IndexSelect(GetContext(), input, 1, valid)),
            (std::vector<float>{1.0F, 1.0F}));
}

TEST_F(IndexingTest, EnforcesShapeDTypeEmptyAndAliasContracts) {
  const auto input = Ones(GetContext(), Shape{2, 3}, DType::FLOAT32);
  const auto empty_index = Empty(GetContext(), Shape{0}, DType::INT64);
  EXPECT_EQ(IndexSelect(GetContext(), input, 1, empty_index).GetShape(), (Shape{2, 0}));

  const auto wrong_index_dtype = Ones(GetContext(), Shape{1}, DType::FLOAT32);
  EXPECT_THROW([[maybe_unused]] const auto result = IndexSelect(GetContext(), input, 1, wrong_index_dtype),
               NotSupportedError);
  const auto rank_two_index = Empty(GetContext(), Shape{1, 1}, DType::INT64);
  EXPECT_THROW([[maybe_unused]] const auto result = IndexSelect(GetContext(), input, 1, rank_two_index),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = Gather(GetContext(), input, 2, rank_two_index),
               InvalidArgumentError);

  auto wrong_output = Empty(GetContext(), Shape{1}, DType::FLOAT32);
  EXPECT_THROW(IndexSelectOut(GetContext(), wrong_output, input, 1, empty_index), InvalidArgumentError);
  auto alias = input;
  auto valid = Empty(GetContext(), Shape{3}, DType::INT64);
  CopyFromHost<int64_t>(GetContext(), valid, {0, 1, 2});
  EXPECT_THROW(GatherOut(GetContext(), alias, input, 1, valid), InvalidArgumentError);

  const auto bool_table = Ones(GetContext(), Shape{2, 3}, DType::BOOL);
  EXPECT_THROW([[maybe_unused]] const auto result = GatherRows(GetContext(), bool_table, valid), NotSupportedError);
}

}  // namespace
}  // namespace ttl
