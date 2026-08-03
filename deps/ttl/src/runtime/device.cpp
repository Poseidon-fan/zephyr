#include "ttl/runtime/device.hpp"

#include <ostream>
#include <string>
#include <utility>

#include "ttl/common/error.hpp"

namespace ttl {

void Device::ThrowInvalidOrdinal(int32_t ordinal, std::source_location location) {
  std::string message{"device ordinal must be non-negative, got "};
  message.append(std::to_string(ordinal));
  throw InvalidArgumentError(std::move(message), location);
}

auto Device::ToString() const -> std::string {
  std::string result{"cuda:"};
  result.append(std::to_string(ordinal_));
  return result;
}

auto operator<<(std::ostream &stream, const Device &device) -> std::ostream & { return stream << device.ToString(); }

}  // namespace ttl
