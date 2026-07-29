#pragma once

#include <cstdint>
#include <optional>
#include <source_location>
#include <string>

#include "ttl/device.hpp"
#include "ttl/error.hpp"

namespace ttl {

/**
 * An owning description of an error that cannot be propagated as a C++ exception.
 *
 * Records may originate from asynchronous execution, resource cleanup, worker threads, or C callbacks. A missing
 * device or stream means the failure is process-wide or cannot be attributed to one execution resource.
 */
struct ErrorRecord final {
  ErrorCode code_;
  std::string message_;
  std::optional<Device> device_;
  std::optional<uint64_t> stream_id_;
  std::source_location location_;
};

/**
 * Receives errors from execution paths that are forbidden to throw.
 *
 * Report may be called concurrently and from destructors or C callbacks. Implementations must be thread-safe,
 * non-blocking, must not call TTL or NVIDIA APIs, and must internally handle allocation or logging failures.
 */
class ErrorSink {
 public:
  virtual ~ErrorSink() = default;

  virtual void Report(ErrorRecord error) noexcept = 0;
};

}  // namespace ttl
