#include "ttl/ops/cast.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
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
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

constexpr std::array SUPPORTED_DTYPES{
    DType::BOOL, DType::UINT8, DType::INT32, DType::INT64, DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32,
};

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

template <TensorStorageType T, size_t size>
void CopyFromHost(ExecutionContext &context, Tensor &tensor, const std::array<T, size> &host) {
  ASSERT_EQ(tensor.GetNumElements(), static_cast<int64_t>(size));
  context.Synchronize();
  ASSERT_EQ(cudaMemcpy(internal::TensorAccess::GetMutableData<T>(tensor), host.data(), host.size() * sizeof(T),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
}

void UploadUnitValues(ExecutionContext &context, Tensor &tensor) {
  switch (tensor.GetDType()) {
    case DType::BOOL:
      CopyFromHost(context, tensor, std::array{false, true});
      return;
    case DType::UINT8:
      CopyFromHost(context, tensor, std::array<uint8_t, 2>{0, 1});
      return;
    case DType::INT32:
      CopyFromHost(context, tensor, std::array<int32_t, 2>{0, 1});
      return;
    case DType::INT64:
      CopyFromHost(context, tensor, std::array<int64_t, 2>{0, 1});
      return;
    case DType::FLOAT16:
      CopyFromHost(context, tensor, std::array{FloatToFloat16(0.0F), FloatToFloat16(1.0F)});
      return;
    case DType::BFLOAT16:
      CopyFromHost(context, tensor, std::array{FloatToBFloat16(0.0F), FloatToBFloat16(1.0F)});
      return;
    case DType::FLOAT32:
      CopyFromHost(context, tensor, std::array<float, 2>{0.0F, 1.0F});
      return;
  }
  FAIL() << "invalid test dtype";
}

template <TensorStorageType T>
[[nodiscard]] auto CopyTwoToHost(const Tensor &tensor) -> std::array<T, 2> {
  std::array<T, 2> host{};
  EXPECT_EQ(cudaMemcpy(host.data(), tensor.GetData<T>(), sizeof(host), cudaMemcpyDeviceToHost), cudaSuccess);
  return host;
}

[[nodiscard]] auto ReadAsDoubles(const Tensor &tensor) -> std::array<double, 2> {
  switch (tensor.GetDType()) {
    case DType::BOOL: {
      const auto host = CopyTwoToHost<bool>(tensor);
      return {host[0] ? 1.0 : 0.0, host[1] ? 1.0 : 0.0};
    }
    case DType::UINT8: {
      const auto host = CopyTwoToHost<uint8_t>(tensor);
      return {static_cast<double>(host[0]), static_cast<double>(host[1])};
    }
    case DType::INT32: {
      const auto host = CopyTwoToHost<int32_t>(tensor);
      return {static_cast<double>(host[0]), static_cast<double>(host[1])};
    }
    case DType::INT64: {
      const auto host = CopyTwoToHost<int64_t>(tensor);
      return {static_cast<double>(host[0]), static_cast<double>(host[1])};
    }
    case DType::FLOAT16: {
      const auto host = CopyTwoToHost<Float16>(tensor);
      return {Float16ToFloat(host[0]), Float16ToFloat(host[1])};
    }
    case DType::BFLOAT16: {
      const auto host = CopyTwoToHost<BFloat16>(tensor);
      return {BFloat16ToFloat(host[0]), BFloat16ToFloat(host[1])};
    }
    case DType::FLOAT32: {
      const auto host = CopyTwoToHost<float>(tensor);
      return {host[0], host[1]};
    }
  }
  ADD_FAILURE() << "invalid test dtype";
  return {};
}

class CastTest : public testing::Test {
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

TEST_F(CastTest, DispatchesEverySupportedDTypePair) {
  for (const auto source_dtype : SUPPORTED_DTYPES) {
    auto input = Empty(GetContext(), Shape{2}, source_dtype);
    UploadUnitValues(GetContext(), input);

    std::vector<Tensor> outputs;
    outputs.reserve(SUPPORTED_DTYPES.size());
    for (const auto target_dtype : SUPPORTED_DTYPES) {
      outputs.push_back(Cast(GetContext(), input, target_dtype));
    }
    GetContext().Synchronize();

    for (const auto &output : outputs) {
      EXPECT_EQ(ReadAsDoubles(output), (std::array<double, 2>{0.0, 1.0}));
    }
  }
}

TEST_F(CastTest, AppliesCheckedFloatingToIntegerSemantics) {
  auto input = Empty(GetContext(), Shape{2}, DType::FLOAT32);
  CopyFromHost(GetContext(), input, std::array<float, 2>{-0.75F, 255.75F});
  const auto output = Cast(GetContext(), input, DType::UINT8);
  GetContext().Synchronize();
  EXPECT_EQ(CopyTwoToHost<uint8_t>(output), (std::array<uint8_t, 2>{0, 255}));

  CopyFromHost(GetContext(), input, std::array<float, 2>{-1.0F, 256.0F});
  auto invalid_output = Empty(GetContext(), Shape{2}, DType::UINT8);
  EXPECT_NO_THROW(CastOut(GetContext(), invalid_output, input));
  EXPECT_THROW(GetContext().CheckAsyncErrors(), DeviceError);

  CopyFromHost(GetContext(), input, std::array<float, 2>{1.9F, -2.9F});
  const auto reusable_output = Cast(GetContext(), input, DType::INT32);
  EXPECT_NO_THROW(GetContext().Synchronize());
  EXPECT_EQ(CopyTwoToHost<int32_t>(reusable_output), (std::array<int32_t, 2>{1, -2}));
}

TEST_F(CastTest, ChecksIntegerNarrowingAndNonFiniteInputs) {
  constexpr auto int32_minimum = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
  constexpr auto int32_maximum = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
  auto integer = Empty(GetContext(), Shape{2}, DType::INT64);

  CopyFromHost(GetContext(), integer, std::array<int64_t, 2>{int32_minimum, int32_maximum});
  const auto exact = Cast(GetContext(), integer, DType::INT32);
  GetContext().Synchronize();
  EXPECT_EQ(CopyTwoToHost<int32_t>(exact),
            (std::array<int32_t, 2>{std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()}));

  CopyFromHost(GetContext(), integer, std::array<int64_t, 2>{int32_minimum - 1, int32_maximum + 1});
  auto narrowed = Empty(GetContext(), Shape{2}, DType::INT32);
  CastOut(GetContext(), narrowed, integer);
  EXPECT_THROW(GetContext().CheckAsyncErrors(), DeviceError);

  CopyFromHost(GetContext(), integer, std::array<int64_t, 2>{-1, 255});
  auto unsigned_output = Empty(GetContext(), Shape{2}, DType::UINT8);
  CastOut(GetContext(), unsigned_output, integer);
  EXPECT_THROW(GetContext().CheckAsyncErrors(), DeviceError);

  auto floating = Empty(GetContext(), Shape{2}, DType::FLOAT32);
  CopyFromHost(GetContext(), floating,
               std::array<float, 2>{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()});
  CastOut(GetContext(), narrowed, floating);
  EXPECT_THROW(GetContext().CheckAsyncErrors(), DeviceError);
}

TEST_F(CastTest, PreservesStickyFirstErrorUntilAnExplicitCheck) {
  auto floating = Empty(GetContext(), Shape{2}, DType::FLOAT32);
  CopyFromHost(GetContext(), floating, std::array<float, 2>{256.0F, 257.0F});
  auto integer = Empty(GetContext(), Shape{2}, DType::INT64);
  CopyFromHost(GetContext(), integer, std::array<int64_t, 2>{INT64_MAX, INT64_MIN});

  auto unsigned_output = Empty(GetContext(), Shape{2}, DType::UINT8);
  CastOut(GetContext(), unsigned_output, floating);
  auto integer_output = Empty(GetContext(), Shape{2}, DType::INT32);
  CastOut(GetContext(), integer_output, integer);

  try {
    GetContext().CheckAsyncErrors();
    FAIL() << "expected a sticky cast error";
  } catch (const DeviceError &error) {
    EXPECT_NE(error.GetMessage().find("float32 to uint8"), std::string::npos);
  }

  auto valid = Zeros(GetContext(), Shape{2}, DType::INT32);
  EXPECT_NO_THROW(GetContext().Synchronize());
  EXPECT_EQ(CopyTwoToHost<int32_t>(valid), (std::array<int32_t, 2>{0, 0}));
}

TEST_F(CastTest, HandlesSpecialBooleanAndFloatingValues) {
  auto input = Empty(GetContext(), Shape{2}, DType::FLOAT32);
  CopyFromHost(GetContext(), input, std::array<float, 2>{std::numeric_limits<float>::quiet_NaN(), -0.0F});
  const auto boolean = Cast(GetContext(), input, DType::BOOL);
  GetContext().Synchronize();
  EXPECT_EQ(CopyTwoToHost<bool>(boolean), (std::array<bool, 2>{true, false}));

  auto integer = Empty(GetContext(), Shape{2}, DType::INT64);
  CopyFromHost(GetContext(), integer, std::array<int64_t, 2>{INT64_MIN, INT64_MAX});
  const auto floating = Cast(GetContext(), integer, DType::FLOAT32);
  GetContext().Synchronize();
  const auto host = CopyTwoToHost<float>(floating);
  EXPECT_LT(host[0], 0.0F);
  EXPECT_GT(host[1], 0.0F);
}

TEST_F(CastTest, SupportsStridedInputAndOutputLayouts) {
  auto matrix = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  CopyFromHost(GetContext(), matrix, std::array<float, 6>{0.5F, 1.5F, 2.5F, 3.5F, 4.5F, 5.5F});
  const auto transposed = Transpose(matrix, 0, 1);
  auto strided_output = EmptyStrided(GetContext(), Shape{3, 2}, Strides{1, 3}, DType::INT32);
  CastOut(GetContext(), strided_output, transposed);
  const auto contiguous = Contiguous(GetContext(), strided_output);
  GetContext().Synchronize();

  std::array<int32_t, 6> host{};
  ASSERT_EQ(cudaMemcpy(host.data(), contiguous.GetData<int32_t>(), sizeof(host), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(host, (std::array<int32_t, 6>{0, 3, 1, 4, 2, 5}));
}

TEST_F(CastTest, EnforcesShapeAndAliasContracts) {
  auto base = Empty(GetContext(), Shape{8}, DType::UINT8);
  const auto input = Narrow(base, 0, 0, 2);
  auto overlapping_output =
      internal::TensorFactory::Create(internal::TensorAccess::GetStorage(base), DType::INT32, Shape{2}, Strides{1}, 0);
  EXPECT_THROW(CastOut(GetContext(), overlapping_output, input), InvalidArgumentError);

  auto wrong_shape = Empty(GetContext(), Shape{3}, DType::INT32);
  EXPECT_THROW(CastOut(GetContext(), wrong_shape, input), InvalidArgumentError);

  auto same_dtype = Empty(GetContext(), Shape{2}, DType::UINT8);
  EXPECT_NO_THROW(CastOut(GetContext(), same_dtype, input));
  EXPECT_NO_THROW(CastOut(GetContext(), same_dtype, same_dtype));

  const auto empty = Empty(GetContext(), Shape{0}, DType::FLOAT32);
  EXPECT_NO_THROW([[maybe_unused]] const auto converted = Cast(GetContext(), empty, DType::INT64));
}

}  // namespace
}  // namespace ttl
