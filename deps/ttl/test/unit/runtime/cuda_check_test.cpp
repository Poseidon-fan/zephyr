#include <atomic>
#include <cstdint>
#include <source_location>
#include <string_view>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "support/runtime_session.hpp"
#include "ttl/common/error.hpp"
#include "ttl/internal/distributed/nccl_api.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/error_report.hpp"
#include "ttl/internal/runtime/library/cublas_api.hpp"

namespace ttl::internal {
namespace {

std::atomic<int32_t> get_last_error_call_count{0};
std::atomic<int32_t> configured_last_error{static_cast<int32_t>(cudaSuccess)};

auto ReportEventReady(cudaEvent_t event) -> cudaError_t {
  static_cast<void>(event);
  return cudaSuccess;
}

auto ReportEventNotReady(cudaEvent_t event) -> cudaError_t {
  static_cast<void>(event);
  return cudaErrorNotReady;
}

auto ReportStreamNotReady(cudaStream_t stream) -> cudaError_t {
  static_cast<void>(stream);
  return cudaErrorNotReady;
}

auto GetConfiguredLastError() -> cudaError_t {
  get_last_error_call_count.fetch_add(1, std::memory_order_relaxed);
  return static_cast<cudaError_t>(configured_last_error.load(std::memory_order_relaxed));
}

void ConfigureLastError(cudaError_t status) {
  configured_last_error.store(static_cast<int32_t>(status), std::memory_order_relaxed);
  get_last_error_call_count.store(0, std::memory_order_relaxed);
}

[[nodiscard]] auto MakeErrorContext() -> ErrorReportContext {
  return ErrorReportContext{
      .location_ = std::source_location::current(),
      .device_ = Device{0},
      .stream_id_ = 17,
  };
}

TEST(CudaQueryTest, DoesNotConsumeLastErrorForReadyEvent) {
  ConfigureLastError(cudaErrorUnknown);
  auto cuda_api = GetCudaApi();
  cuda_api.query_event_ = ReportEventReady;
  cuda_api.get_last_error_ = GetConfiguredLastError;

  const ScopedCudaApiOverride override{cuda_api};
  EXPECT_TRUE(QueryCudaEvent(nullptr, "cudaEventQuery", std::source_location::current()));
  EXPECT_EQ(get_last_error_call_count.load(std::memory_order_relaxed), 0);
}

TEST(CudaQueryTest, ConsumesExpectedNotReadyStatusForThrowingEventQuery) {
  ConfigureLastError(cudaErrorNotReady);
  auto cuda_api = GetCudaApi();
  cuda_api.query_event_ = ReportEventNotReady;
  cuda_api.get_last_error_ = GetConfiguredLastError;

  const ScopedCudaApiOverride override{cuda_api};
  EXPECT_FALSE(QueryCudaEvent(nullptr, "cudaEventQuery", std::source_location::current()));
  EXPECT_EQ(get_last_error_call_count.load(std::memory_order_relaxed), 1);
}

TEST(CudaQueryTest, ThrowsWhenClearingNotReadyRevealsAnotherCudaError) {
  ConfigureLastError(cudaErrorUnknown);
  auto cuda_api = GetCudaApi();
  cuda_api.query_event_ = ReportEventNotReady;
  cuda_api.get_last_error_ = GetConfiguredLastError;

  const ScopedCudaApiOverride override{cuda_api};
  EXPECT_THROW(static_cast<void>(QueryCudaEvent(nullptr, "cudaEventQuery", std::source_location::current())),
               CudaError);
  EXPECT_EQ(get_last_error_call_count.load(std::memory_order_relaxed), 1);
}

TEST(CudaQueryTest, ConsumesExpectedNotReadyStatusForNoexceptStreamQuery) {
  ConfigureLastError(cudaSuccess);
  auto cuda_api = GetCudaApi();
  cuda_api.query_stream_ = ReportStreamNotReady;
  cuda_api.get_last_error_ = GetConfiguredLastError;
  test::RecordingErrorSink error_sink;

  const ScopedCudaApiOverride override{cuda_api};
  EXPECT_EQ(TryQueryCudaStream(nullptr, "cudaStreamQuery", error_sink, MakeErrorContext()), CudaReadiness::NOT_READY);
  EXPECT_EQ(get_last_error_call_count.load(std::memory_order_relaxed), 1);
  EXPECT_TRUE(error_sink.GetRecords().empty());
}

TEST(CudaQueryTest, ReportsUnexpectedErrorWhileClearingNoexceptEventQuery) {
  ConfigureLastError(cudaErrorUnknown);
  auto cuda_api = GetCudaApi();
  cuda_api.query_event_ = ReportEventNotReady;
  cuda_api.get_last_error_ = GetConfiguredLastError;
  test::RecordingErrorSink error_sink;

  const ScopedCudaApiOverride override{cuda_api};
  EXPECT_EQ(TryQueryCudaEvent(nullptr, "cudaEventQuery", error_sink, MakeErrorContext()), CudaReadiness::ERROR);
  EXPECT_EQ(get_last_error_call_count.load(std::memory_order_relaxed), 1);
  const auto records = error_sink.GetRecords();
  ASSERT_EQ(records.size(), 1);
  EXPECT_EQ(records[0].code_, ErrorCode::CUDA);
}

TEST(NativeLibraryCheckTest, ThrowsTypedErrorsAndPreservesOperationContext) {
  try {
    CheckCublas(CUBLAS_STATUS_NOT_SUPPORTED, "cublas operation");
    FAIL() << "CheckCublas did not throw";
  } catch (const CublasError &error) {
    EXPECT_EQ(error.GetCode(), ErrorCode::CUBLAS);
    EXPECT_NE(std::string_view{error.what()}.find("cublas operation"), std::string_view::npos);
  }

  try {
    CheckNccl(ncclInvalidArgument, "nccl operation");
    FAIL() << "CheckNccl did not throw";
  } catch (const NcclError &error) {
    EXPECT_EQ(error.GetCode(), ErrorCode::NCCL);
    EXPECT_NE(std::string_view{error.what()}.find("nccl operation"), std::string_view::npos);
  }
}

TEST(NativeLibraryCheckTest, NoexceptChecksReportExactlyOneTypedRecord) {
  test::RecordingErrorSink error_sink;
  const ErrorReportContext context = MakeErrorContext();
  EXPECT_TRUE(TryCublas(CUBLAS_STATUS_SUCCESS, "success", error_sink, context));
  EXPECT_TRUE(TryNccl(ncclSuccess, "success", error_sink, context));
  EXPECT_FALSE(TryCublas(CUBLAS_STATUS_EXECUTION_FAILED, "cublas failure", error_sink, context));
  EXPECT_FALSE(TryNccl(ncclSystemError, "nccl failure", error_sink, context));

  const auto records = error_sink.GetRecords();
  ASSERT_EQ(records.size(), 2U);
  EXPECT_EQ(records[0].code_, ErrorCode::CUBLAS);
  EXPECT_EQ(records[1].code_, ErrorCode::NCCL);
  EXPECT_EQ(records[0].device_, context.device_);
  EXPECT_EQ(records[1].stream_id_, context.stream_id_);
}

TEST(NativeLibraryOverrideTest, RestoresCuBlasAndNcclTablesAfterScopedOverrides) {
  const auto &cublas_before = GetCublasApi();
  auto cublas_override = cublas_before;
  cublas_override.create_ = nullptr;
  {
    const ScopedCublasApiOverride scoped{cublas_override};
    EXPECT_EQ(GetCublasApi().create_, nullptr);
  }
  EXPECT_EQ(GetCublasApi().create_, cublas_before.create_);

  const auto &nccl_before = GetNcclApi();
  auto nccl_override = nccl_before;
  nccl_override.group_start_ = nullptr;
  {
    const ScopedNcclApiOverride scoped{nccl_override};
    EXPECT_EQ(GetNcclApi().group_start_, nullptr);
  }
  EXPECT_EQ(GetNcclApi().group_start_, nccl_before.group_start_);
}

}  // namespace
}  // namespace ttl::internal
