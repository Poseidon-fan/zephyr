#include "common/exception.hpp"

#include <source_location>
#include <string>
#include <string_view>

namespace zephyr {

Exception::Exception(ExceptionType type, const std::string &message, std::source_location location)
    : std::runtime_error(message), type_(type), location_(location) {}

auto Exception::GetType() const noexcept -> ExceptionType { return type_; }

auto Exception::GetLocation() const noexcept -> const std::source_location & { return location_; }

auto Exception::ExceptionTypeToString(ExceptionType type) noexcept -> std::string_view {
  switch (type) {
    case ExceptionType::INVALID_ARGUMENT:
      return "Invalid Argument";
    case ExceptionType::NOT_IMPLEMENTED:
      return "Not Implemented";
    case ExceptionType::CONFIGURATION:
      return "Configuration";
    case ExceptionType::OUT_OF_MEMORY:
      return "Out of Memory";
    case ExceptionType::INTERNAL:
      return "Internal";
  }
  return "Unknown";
}

}  // namespace zephyr
