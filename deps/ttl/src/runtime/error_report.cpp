#include "ttl/internal/runtime/error_report.hpp"

#include <string>
#include <string_view>

#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"

namespace ttl::internal {

void ReportErrorNoexcept(ErrorCode code, std::string_view message, ErrorSink &error_sink,
                         const ErrorReportContext &context) noexcept {
  try {
    error_sink.Report(ErrorRecord{
        .code_ = code,
        .message_ = std::string{message},
        .device_ = context.device_,
        .stream_id_ = context.stream_id_,
        .location_ = context.location_,
    });
  } catch (...) {
    return;
  }
}

}  // namespace ttl::internal
