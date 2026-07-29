#include "ttl/internal/cuda_check.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"

namespace ttl::internal {
namespace {

[[nodiscard]] auto NullSafeString(const char *value, std::string_view fallback) -> std::string_view {
  return value == nullptr ? fallback : std::string_view{value};
}

[[nodiscard]] constexpr auto NcclStatusName(ncclResult_t status) noexcept -> std::string_view {
  switch (status) {
    case ncclSuccess:
      return "ncclSuccess";
    case ncclUnhandledCudaError:
      return "ncclUnhandledCudaError";
    case ncclSystemError:
      return "ncclSystemError";
    case ncclInternalError:
      return "ncclInternalError";
    case ncclInvalidArgument:
      return "ncclInvalidArgument";
    case ncclInvalidUsage:
      return "ncclInvalidUsage";
    case ncclRemoteError:
      return "ncclRemoteError";
    case ncclInProgress:
      return "ncclInProgress";
    case ncclNumResults:
      return "ncclNumResults";
  }
  return "unknown NCCL status";
}

template <typename Status>
[[nodiscard]] auto FormatNativeError(std::string_view operation, std::string_view status_name, Status status,
                                     std::string_view description) -> std::string {
  std::string message;
  message.reserve(operation.size() + status_name.size() + description.size() + 32);
  message.append(operation);
  message.append(" failed with ");
  message.append(status_name);
  message.append(" (");
  message.append(std::to_string(static_cast<int64_t>(status)));
  message.append("): ");
  message.append(description);
  return message;
}

[[nodiscard]] auto FormatCudaError(cudaError_t status, std::string_view operation) -> std::string {
  return FormatNativeError(operation, NullSafeString(cudaGetErrorName(status), "unknown CUDA status"), status,
                           NullSafeString(cudaGetErrorString(status), "no CUDA error description"));
}

[[nodiscard]] auto FormatCublasError(cublasStatus_t status, std::string_view operation) -> std::string {
  return FormatNativeError(operation, NullSafeString(cublasGetStatusName(status), "unknown cuBLAS status"), status,
                           NullSafeString(cublasGetStatusString(status), "no cuBLAS error description"));
}

[[nodiscard]] auto FormatNcclError(ncclResult_t status, std::string_view operation) -> std::string {
  return FormatNativeError(operation, NcclStatusName(status), status,
                           NullSafeString(ncclGetErrorString(status), "no NCCL error description"));
}

template <typename Format>
void TryReport(ErrorCode code, ErrorSink &error_sink, const ErrorReportContext &context, Format &&format) noexcept {
  try {
    error_sink.Report(ErrorRecord{
        .code_ = code,
        .message_ = std::forward<Format>(format)(),
        .device_ = context.device_,
        .stream_id_ = context.stream_id_,
        .location_ = context.location_,
    });
  } catch (...) {
    // A noexcept cleanup or callback boundary cannot propagate formatting/allocation failures.
    return;
  }
}

}  // namespace

void CheckCuda(cudaError_t status, std::string_view operation, std::source_location location) {
  if (status != cudaSuccess) {
    throw CudaError(FormatCudaError(status, operation), location);
  }
}

void CheckCublas(cublasStatus_t status, std::string_view operation, std::source_location location) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw CublasError(FormatCublasError(status, operation), location);
  }
}

void CheckNccl(ncclResult_t status, std::string_view operation, std::source_location location) {
  if (status != ncclSuccess) {
    throw NcclError(FormatNcclError(status, operation), location);
  }
}

void CheckCudaKernelLaunch(std::string_view kernel, std::source_location location) {
  CheckCuda(cudaPeekAtLastError(), kernel, location);
}

void TryCuda(cudaError_t status, std::string_view operation, ErrorSink &error_sink,
             const ErrorReportContext &context) noexcept {
  if (status != cudaSuccess) {
    TryReport(ErrorCode::CUDA, error_sink, context, [&] { return FormatCudaError(status, operation); });
  }
}

void TryCublas(cublasStatus_t status, std::string_view operation, ErrorSink &error_sink,
               const ErrorReportContext &context) noexcept {
  if (status != CUBLAS_STATUS_SUCCESS) {
    TryReport(ErrorCode::CUBLAS, error_sink, context, [&] { return FormatCublasError(status, operation); });
  }
}

void TryNccl(ncclResult_t status, std::string_view operation, ErrorSink &error_sink,
             const ErrorReportContext &context) noexcept {
  if (status != ncclSuccess) {
    TryReport(ErrorCode::NCCL, error_sink, context, [&] { return FormatNcclError(status, operation); });
  }
}

}  // namespace ttl::internal
