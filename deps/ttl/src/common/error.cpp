#include "ttl/common/error.hpp"

#include <string>
#include <string_view>

namespace ttl {
namespace {

[[nodiscard]] auto FormatError(ErrorCode code, std::string_view message, const std::source_location &location)
    -> std::string {
  std::string result;
  result.reserve(message.size() + 96);
  result.push_back('[');
  result.append(ErrorCodeToString(code));
  result.append("] ");
  result.append(message);
  result.append(" (");
  result.append(location.file_name());
  result.push_back(':');
  result.append(std::to_string(location.line()));
  result.append(" in ");
  result.append(location.function_name());
  result.push_back(')');
  return result;
}

}  // namespace

Error::Error(ErrorCode code, std::string message, std::source_location location)
    : std::runtime_error(FormatError(code, message, location)),
      code_(code),
      message_(std::move(message)),
      location_(location) {}

auto Error::GetCode() const noexcept -> ErrorCode { return code_; }

auto Error::GetMessage() const noexcept -> std::string_view { return message_; }

auto Error::GetLocation() const noexcept -> const std::source_location & { return location_; }

}  // namespace ttl
