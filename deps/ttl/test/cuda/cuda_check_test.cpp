#include "ttl/internal/cuda_check.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <nccl.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"

namespace ttl::internal {
namespace {

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override {
    record_.emplace(std::move(error));
    report_count_++;
  }

  [[nodiscard]] auto GetRecord() const noexcept -> const std::optional<ErrorRecord> & { return record_; }
  [[nodiscard]] auto GetReportCount() const noexcept -> size_t { return report_count_; }

 private:
  std::optional<ErrorRecord> record_;
  size_t report_count_{0};
};

TEST(CudaCheckTest, AcceptsSuccessStatuses) {
  EXPECT_NO_THROW(CheckCuda(cudaSuccess, "cudaSuccess"));
  EXPECT_NO_THROW(CheckCublas(CUBLAS_STATUS_SUCCESS, "cublas success"));
  EXPECT_NO_THROW(CheckNccl(ncclSuccess, "nccl success"));
}

TEST(CudaCheckTest, ThrowsTypedCudaErrorWithNativeDiagnosticAndCallSite) {
  const auto location = std::source_location::current();

  try {
    CheckCuda(cudaErrorInvalidValue, "cuda operation", location);
    FAIL() << "CheckCuda did not throw";
  } catch (const CudaError &error) {
    EXPECT_EQ(error.GetCode(), ErrorCode::CUDA);
    EXPECT_NE(error.GetMessage().find("cuda operation"), std::string_view::npos);
    EXPECT_NE(error.GetMessage().find("cudaErrorInvalidValue"), std::string_view::npos);
    EXPECT_NE(error.GetMessage().find(std::to_string(static_cast<int64_t>(cudaErrorInvalidValue))),
              std::string_view::npos);
    EXPECT_EQ(error.GetLocation().line(), location.line());
  }
}

TEST(CudaCheckTest, ThrowsTypedCublasErrorWithNativeDiagnostic) {
  try {
    CheckCublas(CUBLAS_STATUS_INVALID_VALUE, "cublas operation");
    FAIL() << "CheckCublas did not throw";
  } catch (const CublasError &error) {
    EXPECT_EQ(error.GetCode(), ErrorCode::CUBLAS);
    EXPECT_NE(error.GetMessage().find("cublas operation"), std::string_view::npos);
    EXPECT_NE(error.GetMessage().find("CUBLAS_STATUS_INVALID_VALUE"), std::string_view::npos);
  }
}

TEST(CudaCheckTest, TreatsNcclInProgressAsANonSuccessState) {
  try {
    CheckNccl(ncclInProgress, "nonblocking NCCL operation");
    FAIL() << "CheckNccl did not throw";
  } catch (const NcclError &error) {
    EXPECT_EQ(error.GetCode(), ErrorCode::NCCL);
    EXPECT_NE(error.GetMessage().find("ncclInProgress"), std::string_view::npos);
  }
}

TEST(CudaCheckTest, TryFunctionsIgnoreSuccess) {
  RecordingErrorSink error_sink;
  const ErrorReportContext context{
      .location_ = std::source_location::current(),
      .device_ = Device{1},
      .stream_id_ = uint64_t{9},
  };

  EXPECT_TRUE(TryCuda(cudaSuccess, "cuda success", error_sink, context));
  EXPECT_TRUE(TryCublas(CUBLAS_STATUS_SUCCESS, "cublas success", error_sink, context));
  EXPECT_TRUE(TryNccl(ncclSuccess, "nccl success", error_sink, context));

  EXPECT_EQ(error_sink.GetReportCount(), 0);
  EXPECT_FALSE(error_sink.GetRecord().has_value());
}

TEST(CudaCheckTest, TryCudaReportsOwningContextWithoutThrowing) {
  RecordingErrorSink error_sink;
  const auto location = std::source_location::current();

  EXPECT_FALSE(TryCuda(cudaErrorInvalidDevice, "restore device", error_sink,
                       ErrorReportContext{
                           .location_ = location,
                           .device_ = Device{3},
                           .stream_id_ = uint64_t{42},
                       }));

  ASSERT_TRUE(error_sink.GetRecord().has_value());
  const auto &record = *error_sink.GetRecord();
  EXPECT_EQ(record.code_, ErrorCode::CUDA);
  EXPECT_NE(record.message_.find("restore device"), std::string::npos);
  EXPECT_NE(record.message_.find("cudaErrorInvalidDevice"), std::string::npos);
  EXPECT_EQ(record.device_, Device{3});
  EXPECT_EQ(record.stream_id_, uint64_t{42});
  EXPECT_EQ(record.location_.line(), location.line());
}

TEST(CudaCheckTest, TryCublasAndTryNcclReportCorrectCategories) {
  RecordingErrorSink cublas_error_sink;
  RecordingErrorSink nccl_error_sink;
  const ErrorReportContext context{
      .location_ = std::source_location::current(),
      .device_ = std::nullopt,
      .stream_id_ = std::nullopt,
  };

  EXPECT_FALSE(TryCublas(CUBLAS_STATUS_EXECUTION_FAILED, "cublas launch", cublas_error_sink, context));
  EXPECT_FALSE(TryNccl(ncclInvalidUsage, "nccl enqueue", nccl_error_sink, context));

  ASSERT_TRUE(cublas_error_sink.GetRecord().has_value());
  EXPECT_EQ(cublas_error_sink.GetRecord()->code_, ErrorCode::CUBLAS);
  ASSERT_TRUE(nccl_error_sink.GetRecord().has_value());
  EXPECT_EQ(nccl_error_sink.GetRecord()->code_, ErrorCode::NCCL);
}

}  // namespace
}  // namespace ttl::internal
