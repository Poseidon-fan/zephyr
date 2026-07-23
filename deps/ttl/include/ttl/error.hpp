#pragma once

#include <cstdint>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ttl {

/**
 * ErrorCode is the stable source-level classification for errors reported by TTL.
 *
 * Native CUDA, cuBLAS, and NCCL status values are deliberately not used as ErrorCode values. Their numeric value and
 * textual description belong in the error message produced by the corresponding API wrapper.
 */
enum class ErrorCode : uint8_t {
  INVALID_ARGUMENT,
  NOT_SUPPORTED,
  OUT_OF_MEMORY,
  CUDA,
  CUBLAS,
  NCCL,
  CAPTURE,
  ASYNC_EXECUTION,
  INTERNAL,
};

/** Return the symbolic name of an ErrorCode without allocating. */
[[nodiscard]] constexpr auto ErrorCodeToString(ErrorCode code) noexcept -> std::string_view {
  switch (code) {
    case ErrorCode::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case ErrorCode::NOT_SUPPORTED:
      return "NOT_SUPPORTED";
    case ErrorCode::OUT_OF_MEMORY:
      return "OUT_OF_MEMORY";
    case ErrorCode::CUDA:
      return "CUDA";
    case ErrorCode::CUBLAS:
      return "CUBLAS";
    case ErrorCode::NCCL:
      return "NCCL";
    case ErrorCode::CAPTURE:
      return "CAPTURE";
    case ErrorCode::ASYNC_EXECUTION:
      return "ASYNC_EXECUTION";
    case ErrorCode::INTERNAL:
      return "INTERNAL";
  }
  return "UNKNOWN";
}

/**
 * Base class for every synchronous C++ exception reported by TTL.
 *
 * Constructing an Error never writes to stderr or a logger. The application boundary decides whether and where an
 * exception is logged. what() contains the symbolic error code, the original message, and the captured source
 * location; the individual fields remain available for structured error handling.
 */
class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, std::string message, std::source_location location = std::source_location::current());

  [[nodiscard]] auto GetCode() const noexcept -> ErrorCode;
  [[nodiscard]] auto GetMessage() const noexcept -> std::string_view;
  [[nodiscard]] auto GetLocation() const noexcept -> const std::source_location &;

 private:
  ErrorCode code_;
  std::string message_;
  std::source_location location_;
};

/** A public API argument violates its documented value, shape, dtype, device, or alias contract. */
class InvalidArgumentError final : public Error {
 public:
  explicit InvalidArgumentError(std::string message, std::source_location location = std::source_location::current())
      : Error(ErrorCode::INVALID_ARGUMENT, std::move(message), location) {}
};

/** The requested operation is outside TTL's documented hardware, dtype, layout, or API support boundary. */
class NotSupportedError final : public Error {
 public:
  explicit NotSupportedError(std::string message, std::source_location location = std::source_location::current())
      : Error(ErrorCode::NOT_SUPPORTED, std::move(message), location) {}
};

/** A device allocation cannot be satisfied after the allocator's documented recovery attempt. */
class OutOfMemoryError final : public Error {
 public:
  explicit OutOfMemoryError(std::string message, std::source_location location = std::source_location::current())
      : Error(ErrorCode::OUT_OF_MEMORY, std::move(message), location) {}
};

/** A CUDA Runtime or Driver API call returned an error. */
class CudaError final : public Error {
 public:
  explicit CudaError(std::string message, std::source_location location = std::source_location::current())
      : Error(ErrorCode::CUDA, std::move(message), location) {}
};

/** A cuBLAS or cuBLASLt API call returned an error. */
class CublasError final : public Error {
 public:
  explicit CublasError(std::string message, std::source_location location = std::source_location::current())
      : Error(ErrorCode::CUBLAS, std::move(message), location) {}
};

/** An NCCL API call or asynchronous communicator status reported an error. */
class NcclError final : public Error {
 public:
  explicit NcclError(std::string message, std::source_location location = std::source_location::current())
      : Error(ErrorCode::NCCL, std::move(message), location) {}
};

/** An operation violated CUDA Graph capture or replay requirements. */
class CaptureError final : public Error {
 public:
  explicit CaptureError(std::string message, std::source_location location = std::source_location::current())
      : Error(ErrorCode::CAPTURE, std::move(message), location) {}
};

/** A previously enqueued kernel reported a semantic error through the execution context's device error buffer. */
class DeviceError final : public Error {
 public:
  explicit DeviceError(std::string message, std::source_location location = std::source_location::current())
      : Error(ErrorCode::ASYNC_EXECUTION, std::move(message), location) {}
};

/** TTL detected a broken internal invariant that cannot be caused by a valid public API call. */
class InternalError final : public Error {
 public:
  explicit InternalError(std::string message, std::source_location location = std::source_location::current())
      : Error(ErrorCode::INTERNAL, std::move(message), location) {}
};

}  // namespace ttl
