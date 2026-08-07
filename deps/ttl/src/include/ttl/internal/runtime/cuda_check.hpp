#pragma once

#include <cstdint>
#include <source_location>
#include <string_view>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/error_report.hpp"

namespace ttl::internal {

/** Result of a non-throwing CUDA readiness query; ERROR has already been reported to the supplied ErrorSink. */
enum class CudaReadiness : uint8_t {
  READY,
  NOT_READY,
  ERROR,
};

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

/**
 * @brief Query one CUDA event, returning false for the expected not-ready state and throwing on actual failure.
 *
 * A CUDA Runtime not-ready status is consumed with cudaGetLastError before returning so it cannot contaminate a
 * later CUDA operation on the calling host thread.
 */
[[nodiscard]] auto QueryCudaEvent(cudaEvent_t event, std::string_view operation,
                                  std::source_location location = std::source_location::current()) -> bool;

/**
 * @brief Non-throwing event readiness query with explicit not-ready and error results.
 *
 * Errors are reported through error_sink. NOT_READY is an expected transient state whose CUDA last-error status has
 * already been consumed.
 */
[[nodiscard]] auto TryQueryCudaEvent(cudaEvent_t event, std::string_view operation, ErrorSink &error_sink,
                                     const ErrorReportContext &context) noexcept -> CudaReadiness;

/** Non-throwing stream equivalent of TryQueryCudaEvent. */
[[nodiscard]] auto TryQueryCudaStream(cudaStream_t stream, std::string_view operation, ErrorSink &error_sink,
                                      const ErrorReportContext &context) noexcept -> CudaReadiness;

/** Return whether a cuBLAS call succeeded, reporting a failure without throwing. */
auto TryCublas(cublasStatus_t status, std::string_view operation, ErrorSink &error_sink,
               const ErrorReportContext &context) noexcept -> bool;

/** Return whether an NCCL call succeeded, reporting a failure without throwing. */
auto TryNccl(ncclResult_t status, std::string_view operation, ErrorSink &error_sink,
             const ErrorReportContext &context) noexcept -> bool;

}  // namespace ttl::internal
