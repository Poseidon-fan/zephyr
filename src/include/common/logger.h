#pragma once

#include <cstdint>
#include <source_location>
#include <utility>

#include <fmt/base.h>

namespace zephyr {

/** LogLevel controls the severity threshold of the Zephyr logger. */
enum class LogLevel : uint8_t {
  TRACE = 0,
  DEBUG,
  INFO,
  WARN,
  ERROR,
  OFF,
};

/** Sets the runtime logging threshold for the Zephyr logger. */
void SetLogLevel(LogLevel level);

/** @return the runtime logging threshold of the Zephyr logger */
[[nodiscard]] auto GetLogLevel() -> LogLevel;

namespace internal {

[[nodiscard]] auto ShouldLog(LogLevel level) noexcept -> bool;

void LogMessage(LogLevel level, std::source_location location, fmt::string_view format,
                fmt::format_args arguments) noexcept;

template <typename... Args>
void Log(LogLevel level, std::source_location location, fmt::format_string<Args...> format, Args &&...args) noexcept {
  LogMessage(level, location, format.get(), fmt::make_format_args(args...));
}

}  // namespace internal

}  // namespace zephyr

#define ZEPHYR_LOG_LEVEL_TRACE 0
#define ZEPHYR_LOG_LEVEL_DEBUG 1
#define ZEPHYR_LOG_LEVEL_INFO 2
#define ZEPHYR_LOG_LEVEL_WARN 3
#define ZEPHYR_LOG_LEVEL_ERROR 4
#define ZEPHYR_LOG_LEVEL_OFF 5

#ifndef ZEPHYR_ACTIVE_LOG_LEVEL
#ifdef NDEBUG
#define ZEPHYR_ACTIVE_LOG_LEVEL ZEPHYR_LOG_LEVEL_INFO
#else
#define ZEPHYR_ACTIVE_LOG_LEVEL ZEPHYR_LOG_LEVEL_DEBUG
#endif
#endif

#define ZEPHYR_LOG_INTERNAL(level, ...)                                               \
  do {                                                                                \
    if (::zephyr::internal::ShouldLog((level))) {                                     \
      ::zephyr::internal::Log((level), std::source_location::current(), __VA_ARGS__); \
    }                                                                                 \
  } while (false)

#if ZEPHYR_ACTIVE_LOG_LEVEL <= ZEPHYR_LOG_LEVEL_TRACE
#define ZEPHYR_LOG_TRACE(...) ZEPHYR_LOG_INTERNAL(::zephyr::LogLevel::TRACE, __VA_ARGS__)
#else
#define ZEPHYR_LOG_TRACE(...) ((void)0)
#endif

#if ZEPHYR_ACTIVE_LOG_LEVEL <= ZEPHYR_LOG_LEVEL_DEBUG
#define ZEPHYR_LOG_DEBUG(...) ZEPHYR_LOG_INTERNAL(::zephyr::LogLevel::DEBUG, __VA_ARGS__)
#else
#define ZEPHYR_LOG_DEBUG(...) ((void)0)
#endif

#if ZEPHYR_ACTIVE_LOG_LEVEL <= ZEPHYR_LOG_LEVEL_INFO
#define ZEPHYR_LOG_INFO(...) ZEPHYR_LOG_INTERNAL(::zephyr::LogLevel::INFO, __VA_ARGS__)
#else
#define ZEPHYR_LOG_INFO(...) ((void)0)
#endif

#if ZEPHYR_ACTIVE_LOG_LEVEL <= ZEPHYR_LOG_LEVEL_WARN
#define ZEPHYR_LOG_WARN(...) ZEPHYR_LOG_INTERNAL(::zephyr::LogLevel::WARN, __VA_ARGS__)
#else
#define ZEPHYR_LOG_WARN(...) ((void)0)
#endif

#if ZEPHYR_ACTIVE_LOG_LEVEL <= ZEPHYR_LOG_LEVEL_ERROR
#define ZEPHYR_LOG_ERROR(...) ZEPHYR_LOG_INTERNAL(::zephyr::LogLevel::ERROR, __VA_ARGS__)
#else
#define ZEPHYR_LOG_ERROR(...) ((void)0)
#endif
