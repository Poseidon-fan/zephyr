#pragma once

#include <cstdint>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace zephyr {

/** ExceptionType identifies the stable category of an exception reported by Zephyr. */
enum class ExceptionType : uint8_t {
  /** A public API argument violates its documented contract. */
  INVALID_ARGUMENT = 0,

  /** The requested operation is valid but unsupported. */
  NOT_IMPLEMENTED,

  /** Model, engine, or device configuration is inconsistent. */
  CONFIGURATION,

  /** A required host or device allocation cannot be satisfied. */
  OUT_OF_MEMORY,

  /** An external file operation failed. */
  IO,

  /** A schedulable KV-cache request temporarily exceeds capacity. */
  KV_CACHE_CAPACITY,

  /** A device, kernel, or communication operation failed. */
  EXECUTION,

  /** Zephyr detected a broken internal invariant. */
  INTERNAL,
};

/** Base class for all exceptions reported by Zephyr. */
class Exception : public std::runtime_error {
 public:
  /**
   * Constructs a Zephyr exception.
   * @param type stable exception category
   * @param message human-readable failure description
   * @param location source location at which the exception was created
   */
  Exception(ExceptionType type, const std::string &message,
            std::source_location location = std::source_location::current());

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
  explicit InvalidArgumentException(const std::string &message,
                                    std::source_location location = std::source_location::current())
      : Exception(ExceptionType::INVALID_ARGUMENT, message, location) {}
};

/** The requested operation is intentionally unsupported. */
class NotImplementedException final : public Exception {
 public:
  explicit NotImplementedException(const std::string &message,
                                   std::source_location location = std::source_location::current())
      : Exception(ExceptionType::NOT_IMPLEMENTED, message, location) {}
};

/** Engine or model configuration is invalid or inconsistent. */
class ConfigurationException final : public Exception {
 public:
  explicit ConfigurationException(const std::string &message,
                                  std::source_location location = std::source_location::current())
      : Exception(ExceptionType::CONFIGURATION, message, location) {}
};

/** A required host or device allocation cannot be satisfied. */
class OutOfMemoryException final : public Exception {
 public:
  explicit OutOfMemoryException(const std::string &message,
                                std::source_location location = std::source_location::current())
      : Exception(ExceptionType::OUT_OF_MEMORY, message, location) {}
};

/** A checkpoint or other external file operation failed. */
class IOException final : public Exception {
 public:
  explicit IOException(const std::string &message, std::source_location location = std::source_location::current())
      : Exception(ExceptionType::IO, message, location) {}
};

/** A KV-cache request temporarily exceeds available cache capacity. */
class KVCacheCapacityException final : public Exception {
 public:
  explicit KVCacheCapacityException(const std::string &message,
                                    std::source_location location = std::source_location::current())
      : Exception(ExceptionType::KV_CACHE_CAPACITY, message, location) {}
};

/** A device, kernel, or communication operation failed. */
class ExecutionException final : public Exception {
 public:
  explicit ExecutionException(const std::string &message,
                              std::source_location location = std::source_location::current())
      : Exception(ExceptionType::EXECUTION, message, location) {}
};

/** Zephyr detected a broken internal invariant. */
class InternalException final : public Exception {
 public:
  explicit InternalException(const std::string &message,
                             std::source_location location = std::source_location::current())
      : Exception(ExceptionType::INTERNAL, message, location) {}
};

}  // namespace zephyr
