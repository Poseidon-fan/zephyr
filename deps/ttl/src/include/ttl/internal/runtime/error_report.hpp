#pragma once

#include <cstdint>
#include <optional>
#include <source_location>
#include <string_view>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"

namespace ttl::internal {

struct ErrorReportContext final {
  std::source_location location_;
  std::optional<Device> device_;
  std::optional<uint64_t> stream_id_;
};

void ReportErrorNoexcept(ErrorCode code, std::string_view message, ErrorSink &error_sink,
                         const ErrorReportContext &context) noexcept;

}  // namespace ttl::internal
