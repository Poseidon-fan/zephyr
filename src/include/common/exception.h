#pragma once

#include <cstdint>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace zephyr {

/** ExceptionType identifies the stable category of an exception reported by Zephyr. */
enum class ExceptionType : uint8_t {
  INVALID_ARGUMENT = 0,
  NOT_IMPLEMENTED,
  CONFIGURATION,
  OUT_OF_MEMORY,
  INTERNAL,
};

/** Base class for all exceptions reported by Zephyr. */
class Exception : public std::runtime_error {
 public:
  Exception(ExceptionType type, std::string message, std::source_location location = std::source_location::current());

  /** @return the category of this exception */
  [[nodiscard]] auto GetType() const noexcept -> ExceptionType;

  /** @return the source location at which this exception was created */
  [[nodiscard]] auto GetLocation() const noexcept -> const std::source_location &;

  /** @return the symbolic name of an exception category */
  [[nodiscard]] static auto ExceptionTypeToString(ExceptionType type) noexcept -> std::string_view;

 private:
  ExceptionType type_;
  std::source_location location_;
};

/** A public API argument violates its documented contract. */
class InvalidArgumentException final : public Exception {
 public:
  explicit InvalidArgumentException(std::string message,
                                    std::source_location location = std::source_location::current())
      : Exception(ExceptionType::INVALID_ARGUMENT, std::move(message), location) {}
};

/** The requested operation is intentionally unsupported. */
class NotImplementedException final : public Exception {
 public:
  explicit NotImplementedException(std::string message, std::source_location location = std::source_location::current())
      : Exception(ExceptionType::NOT_IMPLEMENTED, std::move(message), location) {}
};

/** Engine or model configuration is invalid or inconsistent. */
class ConfigurationException final : public Exception {
 public:
  explicit ConfigurationException(std::string message, std::source_location location = std::source_location::current())
      : Exception(ExceptionType::CONFIGURATION, std::move(message), location) {}
};

/** A required host or device allocation cannot be satisfied. */
class OutOfMemoryException final : public Exception {
 public:
  explicit OutOfMemoryException(std::string message, std::source_location location = std::source_location::current())
      : Exception(ExceptionType::OUT_OF_MEMORY, std::move(message), location) {}
};

/** Zephyr detected a broken internal invariant. */
class InternalException final : public Exception {
 public:
  explicit InternalException(std::string message, std::source_location location = std::source_location::current())
      : Exception(ExceptionType::INTERNAL, std::move(message), location) {}
};

}  // namespace zephyr
