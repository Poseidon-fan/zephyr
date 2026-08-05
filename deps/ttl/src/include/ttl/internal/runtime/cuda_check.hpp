#pragma once

#include <source_location>
#include <string_view>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/error_report.hpp"

namespace ttl::internal {

/** Throw CudaError unless status is cudaSuccess. */
void CheckCuda(cudaError_t status, std::string_view operation,
               std::source_location location = std::source_location::current());

/** Throw CublasError unless status is CUBLAS_STATUS_SUCCESS. */
void CheckCublas(cublasStatus_t status, std::string_view operation,
                 std::source_location location = std::source_location::current());

/**
 * @brief Throw NcclError unless status is ncclSuccess.
 *
 * ncclInProgress is a valid state for selected nonblocking NCCL APIs, but is not success. Callers of those APIs must
 * handle it before using this function.
 */
void CheckNccl(ncclResult_t status, std::string_view operation,
               std::source_location location = std::source_location::current());

/** Return whether a CUDA call succeeded, reporting a failure without throwing. */
auto TryCuda(cudaError_t status, std::string_view operation, ErrorSink &error_sink,
             const ErrorReportContext &context) noexcept -> bool;

/** As above, appending a parenthesized detail to the operation name only on failure. */
auto TryCuda(cudaError_t status, std::string_view operation, std::string_view detail, ErrorSink &error_sink,
             const ErrorReportContext &context) noexcept -> bool;

/** Return whether a cuBLAS call succeeded, reporting a failure without throwing. */
auto TryCublas(cublasStatus_t status, std::string_view operation, ErrorSink &error_sink,
               const ErrorReportContext &context) noexcept -> bool;

/** Return whether an NCCL call succeeded, reporting a failure without throwing. */
auto TryNccl(ncclResult_t status, std::string_view operation, ErrorSink &error_sink,
             const ErrorReportContext &context) noexcept -> bool;

}  // namespace ttl::internal
