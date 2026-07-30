#include "ttl/ops/reduction.hpp"

#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <vector>

#include <cuda_runtime_api.h>
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

[[nodiscard]] auto CopyBoolToHost(ExecutionContext &context, const Tensor &tensor) -> std::vector<uint8_t> {
  context.Synchronize();
  std::vector<uint8_t> host(static_cast<size_t>(tensor.GetNumElements()));
  if (!host.empty()) {
    EXPECT_EQ(cudaMemcpy(host.data(), tensor.GetData<bool>(), host.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  }
  return host;
}

[[nodiscard]] auto CopyAsFloat(ExecutionContext &context, const Tensor &tensor) -> std::vector<float> {
  const auto converted = Cast(context, tensor, DType::FLOAT32);
  return CopyToHost<float>(context, converted);
}

class ReductionTest : public testing::Test {
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

TEST_F(ReductionTest, DispatchesCompleteSupportedDTypeMatrices) {
  for (const auto dtype : {DType::UINT8, DType::INT32, DType::INT64, DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto input = Ones(GetContext(), Shape{2, 3}, dtype);
    const auto rows = ReductionOptions{.axes_ = {1}};
    EXPECT_EQ(CopyAsFloat(GetContext(), Minimum(GetContext(), input, rows)), (std::vector<float>{1.0F, 1.0F}));
    EXPECT_EQ(CopyAsFloat(GetContext(), Maximum(GetContext(), input, rows)), (std::vector<float>{1.0F, 1.0F}));
    EXPECT_EQ(CopyToHost<int64_t>(GetContext(), ArgMin(GetContext(), input, 1)), (std::vector<int64_t>{0, 0}));
    EXPECT_EQ(CopyToHost<int64_t>(GetContext(), ArgMax(GetContext(), input, 1)), (std::vector<int64_t>{0, 0}));
  }

  for (const auto dtype : {DType::INT32, DType::INT64, DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto input = Ones(GetContext(), Shape{2, 3}, dtype);
    EXPECT_EQ(CopyAsFloat(GetContext(), Sum(GetContext(), input)), (std::vector<float>{6.0F}));
  }

  for (const auto dtype : {DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto input = Ones(GetContext(), Shape{2, 3}, dtype);
    EXPECT_EQ(CopyAsFloat(GetContext(), Mean(GetContext(), input)), (std::vector<float>{1.0F}));
  }
}

TEST_F(ReductionTest, ReducesArbitraryAxesAndStridedLayouts) {
  auto input = Empty(GetContext(), Shape{2, 3, 4}, DType::FLOAT32);
  std::vector<float> values(24);
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>(index + 1);
  }
  CopyFromHost(GetContext(), input, values);

  const auto across_middle = ReductionOptions{.axes_ = {1}};
  EXPECT_EQ(CopyToHost<float>(GetContext(), Sum(GetContext(), input, across_middle)),
            (std::vector<float>{15.0F, 18.0F, 21.0F, 24.0F, 51.0F, 54.0F, 57.0F, 60.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Mean(GetContext(), input, across_middle)),
            (std::vector<float>{5.0F, 6.0F, 7.0F, 8.0F, 17.0F, 18.0F, 19.0F, 20.0F}));

  const auto keep_multiple = ReductionOptions{.axes_ = {-1, 0}, .keep_dimensions_ = true};
  const auto multiple = Sum(GetContext(), input, keep_multiple);
  EXPECT_EQ(multiple.GetShape(), (Shape{1, 3, 1}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), multiple), (std::vector<float>{68.0F, 100.0F, 132.0F}));

  const auto transposed = Transpose(input, 1, 2);
  auto strided_output = EmptyStrided(GetContext(), Shape{2, 4}, Strides{1, 2}, DType::FLOAT32);
  SumOut(GetContext(), strided_output, transposed, ReductionOptions{.axes_ = {2}});
  EXPECT_EQ(CopyToHost<float>(GetContext(), Contiguous(GetContext(), strided_output)),
            (std::vector<float>{15.0F, 18.0F, 21.0F, 24.0F, 51.0F, 54.0F, 57.0F, 60.0F}));
}

TEST_F(ReductionTest, UsesSpecifiedAccumulatorsAndIntegerModulo) {
  for (const auto dtype : {DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32}) {
    const auto input = Ones(GetContext(), Shape{4096}, dtype);
    EXPECT_EQ(CopyAsFloat(GetContext(), Sum(GetContext(), input)), (std::vector<float>{4096.0F}));
    EXPECT_EQ(CopyAsFloat(GetContext(), Mean(GetContext(), input)), (std::vector<float>{1.0F}));
  }

  auto int32_input = Empty(GetContext(), Shape{3}, DType::INT32);
  CopyFromHost<int32_t>(GetContext(), int32_input,
                        {std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max(), 2});
  EXPECT_EQ(CopyToHost<int32_t>(GetContext(), Sum(GetContext(), int32_input)), (std::vector<int32_t>{0}));

  auto int64_input = Empty(GetContext(), Shape{2}, DType::INT64);
  CopyFromHost<int64_t>(GetContext(), int64_input, {std::numeric_limits<int64_t>::max(), int64_t{1}});
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), Sum(GetContext(), int64_input)),
            (std::vector<int64_t>{std::numeric_limits<int64_t>::min()}));
}

TEST_F(ReductionTest, DefinesExtremaNanSignedZeroAndTieSemantics) {
  auto input = Empty(GetContext(), Shape{2, 5}, DType::FLOAT32);
  const auto nan = std::bit_cast<float>(uint32_t{0x7FC12345});
  CopyFromHost<float>(
      GetContext(), input,
      {2.0F, nan, std::numeric_limits<float>::quiet_NaN(), -1.0F, -1.0F, -0.0F, 0.0F, -0.0F, 3.0F, 3.0F});

  const auto options = ReductionOptions{.axes_ = {1}};
  const auto minimum = CopyToHost<float>(GetContext(), Minimum(GetContext(), input, options));
  const auto maximum = CopyToHost<float>(GetContext(), Maximum(GetContext(), input, options));
  EXPECT_TRUE(std::isnan(minimum[0]));
  EXPECT_TRUE(std::isnan(maximum[0]));
  EXPECT_EQ(std::bit_cast<uint32_t>(minimum[0]), std::bit_cast<uint32_t>(nan));
  EXPECT_EQ(std::bit_cast<uint32_t>(maximum[0]), std::bit_cast<uint32_t>(nan));
  EXPECT_TRUE(std::signbit(minimum[1]));
  EXPECT_FLOAT_EQ(maximum[1], 3.0F);

  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), ArgMin(GetContext(), input, 1)), (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), ArgMax(GetContext(), input, -1)), (std::vector<int64_t>{1, 3}));

  auto integers = Empty(GetContext(), Shape{2, 4}, DType::UINT8);
  CopyFromHost<uint8_t>(GetContext(), integers, {4, 1, 1, 9, 7, 7, 2, 2});
  EXPECT_EQ(CopyToHost<uint8_t>(GetContext(), Minimum(GetContext(), integers, options)), (std::vector<uint8_t>{1, 2}));
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), ArgMin(GetContext(), integers, 1)), (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(CopyToHost<int64_t>(GetContext(), ArgMax(GetContext(), integers, 1)), (std::vector<int64_t>{3, 0}));
}

TEST_F(ReductionTest, ImplementsLogicalAndEmptyGroupIdentities) {
  auto boolean = Empty(GetContext(), Shape{2, 3}, DType::BOOL);
  CopyBoolFromHost(GetContext(), boolean, {0, 0, 1, 1, 1, 1});
  const auto rows = ReductionOptions{.axes_ = {1}, .keep_dimensions_ = true};
  EXPECT_EQ(CopyBoolToHost(GetContext(), Any(GetContext(), boolean, rows)), (std::vector<uint8_t>{1, 1}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), All(GetContext(), boolean, rows)), (std::vector<uint8_t>{0, 1}));

  const auto empty_float = Empty(GetContext(), Shape{0, 3}, DType::FLOAT32);
  const auto empty_bool = Empty(GetContext(), Shape{0, 3}, DType::BOOL);
  const auto reduce_empty_axis = ReductionOptions{.axes_ = {0}};
  EXPECT_EQ(CopyToHost<float>(GetContext(), Sum(GetContext(), empty_float, reduce_empty_axis)),
            (std::vector<float>{0.0F, 0.0F, 0.0F}));
  const auto means = CopyToHost<float>(GetContext(), Mean(GetContext(), empty_float, reduce_empty_axis));
  EXPECT_TRUE(std::isnan(means[0]));
  EXPECT_TRUE(std::isnan(means[1]));
  EXPECT_TRUE(std::isnan(means[2]));
  EXPECT_EQ(CopyBoolToHost(GetContext(), Any(GetContext(), empty_bool, reduce_empty_axis)),
            (std::vector<uint8_t>{0, 0, 0}));
  EXPECT_EQ(CopyBoolToHost(GetContext(), All(GetContext(), empty_bool, reduce_empty_axis)),
            (std::vector<uint8_t>{1, 1, 1}));

  EXPECT_THROW([[maybe_unused]] const auto result = Minimum(GetContext(), empty_float, reduce_empty_axis),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = ArgMax(GetContext(), empty_float, 0), InvalidArgumentError);
  const auto empty_output = Minimum(GetContext(), empty_float, ReductionOptions{.axes_ = {1}});
  EXPECT_EQ(empty_output.GetShape(), (Shape{0}));
}

TEST_F(ReductionTest, HandlesScalarsAndEnforcesSchemaContracts) {
  const auto scalar = Full(GetContext(), Shape{}, Scalar{3.0}, DType::FLOAT32);
  EXPECT_EQ(CopyToHost<float>(GetContext(), Sum(GetContext(), scalar)), (std::vector<float>{3.0F}));
  EXPECT_EQ(CopyToHost<float>(GetContext(), Mean(GetContext(), scalar)), (std::vector<float>{3.0F}));
  EXPECT_THROW([[maybe_unused]] const auto result = ArgMax(GetContext(), scalar, 0), InvalidArgumentError);

  const auto input = Ones(GetContext(), Shape{1, 3}, DType::FLOAT32);
  EXPECT_THROW([[maybe_unused]] const auto result = Sum(GetContext(), input, ReductionOptions{.axes_ = {0, -2}}),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto result = Sum(GetContext(), input, ReductionOptions{.axes_ = {2}}),
               InvalidArgumentError);
  const auto boolean = Ones(GetContext(), Shape{2}, DType::BOOL);
  EXPECT_THROW([[maybe_unused]] const auto result = Sum(GetContext(), boolean), NotSupportedError);
  EXPECT_THROW([[maybe_unused]] const auto result = Mean(GetContext(), Ones(GetContext(), Shape{2}, DType::INT32)),
               NotSupportedError);
  EXPECT_THROW([[maybe_unused]] const auto result = Any(GetContext(), input), NotSupportedError);

  auto wrong_shape = Empty(GetContext(), Shape{1}, DType::FLOAT32);
  EXPECT_THROW(SumOut(GetContext(), wrong_shape, input, ReductionOptions{.axes_ = {0}}), InvalidArgumentError);
  auto wrong_dtype = Empty(GetContext(), Shape{3}, DType::INT64);
  EXPECT_THROW(SumOut(GetContext(), wrong_dtype, input, ReductionOptions{.axes_ = {0}}), InvalidArgumentError);

  auto exact_alias = Squeeze(input, 0);
  EXPECT_THROW(SumOut(GetContext(), exact_alias, input, ReductionOptions{.axes_ = {0}}), InvalidArgumentError);
}

TEST_F(ReductionTest, UsesDeterministicTwoStageScratchForLargeGroups) {
  constexpr auto element_count = int64_t{1} << 20;
  const auto input = Ones(GetContext(), Shape{element_count}, DType::FLOAT32);
  const auto first = Sum(GetContext(), input);
  const auto second = Sum(GetContext(), input);
  const auto first_value = CopyToHost<float>(GetContext(), first);
  const auto second_value = CopyToHost<float>(GetContext(), second);
  EXPECT_EQ(first_value, (std::vector<float>{static_cast<float>(element_count)}));
  EXPECT_EQ(std::bit_cast<uint32_t>(first_value[0]), std::bit_cast<uint32_t>(second_value[0]));

  const auto &lane = internal::ContextAccess::GetPrimaryLane(GetContext(), std::source_location::current());
  EXPECT_GT(lane.GetScratchHighWaterBytes(), 0);
}

}  // namespace
}  // namespace ttl
