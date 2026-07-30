#include "ttl/ops/copy.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/layout.hpp"
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

template <TensorStorageType T, size_t size>
void CopyFromHost(ExecutionContext &context, Tensor &tensor, const std::array<T, size> &host) {
  ASSERT_EQ(tensor.GetNumElements(), static_cast<int64_t>(size));
  context.Synchronize();
  if constexpr (size != 0) {
    ASSERT_EQ(cudaMemcpy(internal::TensorAccess::GetMutableData<T>(tensor), host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
  }
}

auto FailMemset(void * /*pointer*/, int /*value*/, size_t /*bytes*/, cudaStream_t /*stream*/) -> cudaError_t {
  return cudaErrorInvalidValue;
}

auto FailMemcpy(void * /*destination*/, const void * /*source*/, size_t /*bytes*/, cudaMemcpyKind /*kind*/,
                cudaStream_t /*stream*/) -> cudaError_t {
  return cudaErrorInvalidValue;
}

class FoundationOpsTest : public testing::Test {
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

TEST_F(FoundationOpsTest, FillsEverySupportedDType) {
  auto boolean = Empty(GetContext(), Shape{5}, DType::BOOL);
  auto unsigned_integer = Empty(GetContext(), Shape{5}, DType::UINT8);
  auto integer32 = Empty(GetContext(), Shape{5}, DType::INT32);
  auto integer64 = Empty(GetContext(), Shape{5}, DType::INT64);
  auto half = Empty(GetContext(), Shape{5}, DType::FLOAT16);
  auto bfloat = Empty(GetContext(), Shape{5}, DType::BFLOAT16);
  auto floating = Empty(GetContext(), Shape{5}, DType::FLOAT32);

  FillOut(GetContext(), boolean, Scalar{true});
  FillOut(GetContext(), unsigned_integer, Scalar{int64_t{7}});
  FillOut(GetContext(), integer32, Scalar{int64_t{-19}});
  FillOut(GetContext(), integer64, Scalar{int64_t{1} << 40});
  FillOut(GetContext(), half, Scalar{1.5});
  FillOut(GetContext(), bfloat, Scalar{-2.25});
  FillOut(GetContext(), floating, Scalar{3.5});

  EXPECT_EQ(CopyBoolToHost(GetContext(), boolean), std::vector<uint8_t>(5, 1));
  EXPECT_EQ(CopyToHost<uint8_t>(GetContext(), unsigned_integer), std::vector<uint8_t>(5, 7));
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), integer32), std::vector<int32_t>(5, -19));
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), integer64), std::vector<int64_t>(5, int64_t{1} << 40));
  for (const auto value : CopyToHost<Float16>(GetContext(), half)) {
    EXPECT_FLOAT_EQ(Float16ToFloat(value), 1.5F);
  }
  for (const auto value : CopyToHost<BFloat16>(GetContext(), bfloat)) {
    EXPECT_FLOAT_EQ(BFloat16ToFloat(value), -2.25F);
  }
  EXPECT_EQ(CopyToHost<float>(GetContext(), floating), std::vector<float>(5, 3.5F));
}

TEST_F(FoundationOpsTest, FillsDenseStridedOutputsAndValidatesScalarBeforeEmptyReturn) {
  auto transposed_storage = EmptyStrided(GetContext(), Shape{3, 2}, Strides{1, 3}, DType::INT32);
  FillOut(GetContext(), transposed_storage, Scalar{int64_t{9}});
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), transposed_storage), std::vector<int32_t>(6, 9));

  auto empty = EmptyStrided(GetContext(), Shape{0, 3}, Strides{1, 1}, DType::UINT8);
  EXPECT_THROW(FillOut(GetContext(), empty, Scalar{int64_t{256}}), OverflowError);

  auto scalar = Empty(GetContext(), Shape{1}, DType::FLOAT32);
  auto broadcast_output = Expand(scalar, Shape{4});
  EXPECT_THROW(FillOut(GetContext(), broadcast_output, Scalar{1.0}), InvalidArgumentError);

  auto negative_zero = Empty(GetContext(), Shape{1}, DType::FLOAT32);
  FillOut(GetContext(), negative_zero, Scalar{-0.0});
  const auto negative_zero_host = CopyToHost<float>(GetContext(), negative_zero);
  ASSERT_EQ(negative_zero_host.size(), 1);
  EXPECT_TRUE(std::signbit(negative_zero_host[0]));
}

TEST_F(FoundationOpsTest, CopiesContiguousStridedAndBroadcastInputs) {
  auto matrix = Empty(GetContext(), Shape{2, 3}, DType::INT32);
  constexpr std::array<int32_t, 6> values{0, 1, 2, 3, 4, 5};
  CopyFromHost(GetContext(), matrix, values);

  const auto transposed = Transpose(matrix, 0, 1);
  auto contiguous_output = Empty(GetContext(), Shape{3, 2}, DType::INT32);
  CopyOut(GetContext(), contiguous_output, transposed);
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), contiguous_output), (std::vector<int32_t>{0, 3, 1, 4, 2, 5}));

  auto strided_output = EmptyStrided(GetContext(), Shape{3, 2}, Strides{1, 3}, DType::INT32);
  CopyOut(GetContext(), strided_output, contiguous_output);
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), strided_output), (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));

  const auto row = Narrow(matrix, 0, 0, 1);
  const auto expanded = Expand(row, Shape{2, 3});
  auto broadcast_output = Empty(GetContext(), Shape{2, 3}, DType::INT32);
  CopyOut(GetContext(), broadcast_output, expanded);
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), broadcast_output), (std::vector<int32_t>{0, 1, 2, 0, 1, 2}));
}

TEST_F(FoundationOpsTest, ImplementsCloneAndContiguousOwnershipSemantics) {
  auto input = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  constexpr std::array<float, 6> values{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
  CopyFromHost(GetContext(), input, values);

  const auto contiguous_alias = Contiguous(GetContext(), input);
  EXPECT_EQ(ClassifyAlias(contiguous_alias, input), AliasKind::EXACT);

  const auto cloned = Clone(GetContext(), input);
  EXPECT_EQ(ClassifyAlias(cloned, input), AliasKind::DISJOINT);
  EXPECT_TRUE(cloned.IsContiguous());
  EXPECT_EQ(CopyToHost<float>(GetContext(), cloned), std::vector<float>(values.begin(), values.end()));

  const auto transposed = Transpose(input, 0, 1);
  const auto materialized = Contiguous(GetContext(), transposed);
  EXPECT_TRUE(materialized.IsContiguous());
  EXPECT_EQ(ClassifyAlias(materialized, transposed), AliasKind::DISJOINT);
  EXPECT_EQ(CopyToHost<float>(GetContext(), materialized), (std::vector<float>{0.0F, 3.0F, 1.0F, 4.0F, 2.0F, 5.0F}));
}

TEST_F(FoundationOpsTest, EnforcesCopyShapeDTypeLayoutAndAliasContracts) {
  auto base = Empty(GetContext(), Shape{5}, DType::INT32);
  auto output = Narrow(base, 0, 0, 4);
  const auto overlapping_input = Narrow(base, 0, 1, 4);
  EXPECT_THROW(CopyOut(GetContext(), output, overlapping_input), InvalidArgumentError);

  EXPECT_NO_THROW(CopyOut(GetContext(), base, base));

  auto wrong_shape = Empty(GetContext(), Shape{4}, DType::INT32);
  EXPECT_THROW(CopyOut(GetContext(), wrong_shape, base), InvalidArgumentError);

  auto wrong_dtype = Empty(GetContext(), Shape{5}, DType::FLOAT32);
  EXPECT_THROW(CopyOut(GetContext(), wrong_dtype, base), InvalidArgumentError);

  auto noncontiguous = EmptyStrided(GetContext(), Shape{2, 3}, Strides{1, 2}, DType::INT32);
  auto matching_input = Empty(GetContext(), Shape{2, 3}, DType::INT32);
  EXPECT_THROW(ContiguousOut(GetContext(), noncontiguous, matching_input), InvalidArgumentError);

  auto empty_output = Empty(GetContext(), Shape{0}, DType::INT32);
  const auto empty_input = Empty(GetContext(), Shape{0}, DType::INT32);
  EXPECT_NO_THROW(CopyOut(GetContext(), empty_output, empty_input));
}

TEST_F(FoundationOpsTest, PropagatesAsynchronousCudaSubmissionFailures) {
  auto output = Empty(GetContext(), Shape{4}, DType::INT32);
  auto input = Empty(GetContext(), Shape{4}, DType::INT32);

  auto cuda_api = internal::GetCudaApi();
  cuda_api.memset_async_ = FailMemset;
  {
    internal::ScopedCudaApiOverride override{cuda_api};
    EXPECT_THROW(FillOut(GetContext(), output, Scalar{int64_t{0}}), CudaError);
  }

  cuda_api = internal::GetCudaApi();
  cuda_api.memcpy_async_ = FailMemcpy;
  {
    internal::ScopedCudaApiOverride override{cuda_api};
    EXPECT_THROW(CopyOut(GetContext(), output, input), CudaError);
  }
}

}  // namespace
}  // namespace ttl
