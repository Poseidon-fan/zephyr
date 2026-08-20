#include "common/logger.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <source_location>

#include <fmt/format.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_sinks.h>

namespace zephyr {
namespace {

/** Converts the public Zephyr severity into the backing spdlog severity. */
auto ToSpdlogLevel(LogLevel level) noexcept -> spdlog::level::level_enum {
  switch (level) {
    case LogLevel::TRACE:
      return spdlog::level::trace;
    case LogLevel::DEBUG:
      return spdlog::level::debug;
    case LogLevel::INFO:
      return spdlog::level::info;
    case LogLevel::WARN:
      return spdlog::level::warn;
    case LogLevel::ERROR:
      return spdlog::level::err;
    case LogLevel::OFF:
      return spdlog::level::off;
  }
  return spdlog::level::off;
}

/** Converts a backing spdlog severity into the public Zephyr severity. */
auto FromSpdlogLevel(spdlog::level::level_enum level) noexcept -> LogLevel {
  switch (level) {
    case spdlog::level::trace:
      return LogLevel::TRACE;
    case spdlog::level::debug:
      return LogLevel::DEBUG;
    case spdlog::level::info:
      return LogLevel::INFO;
    case spdlog::level::warn:
      return LogLevel::WARN;
    case spdlog::level::err:
    case spdlog::level::critical:
      return LogLevel::ERROR;
    case spdlog::level::off:
    case spdlog::level::n_levels:
      return LogLevel::OFF;
  }
  return LogLevel::OFF;
}

/** Returns the lazily constructed process-wide logger. */
auto Logger() -> spdlog::logger & {
  static auto logger = [] {
    auto sink = std::make_shared<spdlog::sinks::stderr_sink_mt>();
    auto result = spdlog::logger{"zephyr", std::move(sink)};
    result.set_pattern("%Y-%m-%d %H:%M:%S.%e [%n] [%l] [%t] [%s:%# %!] %v");
#ifdef NDEBUG
    result.set_level(spdlog::level::info);
#else
    result.set_level(spdlog::level::debug);
#endif
    result.flush_on(spdlog::level::err);
    return result;
  }();
  return logger;
}

/** Reports a logging-infrastructure failure without recursively using the logger. */
void ReportLogFailure(std::source_location location, const char *message) noexcept {
  std::fprintf(stderr, "%s:%u in %s: logging failed: %s\n", location.file_name(),
               static_cast<unsigned int>(location.line()), location.function_name(), message);
  std::fflush(stderr);
}

}  // namespace

void SetLogLevel(LogLevel level) { Logger().set_level(ToSpdlogLevel(level)); }

auto GetLogLevel() -> LogLevel { return FromSpdlogLevel(Logger().level()); }

namespace internal {

auto ShouldLog(LogLevel level) noexcept -> bool {
  try {
    return Logger().should_log(ToSpdlogLevel(level));
  } catch (...) {
    return true;
  }
}

void LogMessage(LogLevel level, std::source_location location, fmt::string_view format,
                fmt::format_args arguments) noexcept {
  try {
    auto &logger = Logger();
    const auto spdlog_level = ToSpdlogLevel(level);
    if (!logger.should_log(spdlog_level)) {
      return;
    }

    auto buffer = fmt::memory_buffer{};
    fmt::vformat_to(fmt::appender{buffer}, format, arguments);
    logger.log(spdlog::source_loc{location.file_name(), static_cast<int>(location.line()), location.function_name()},
               spdlog_level, spdlog::string_view_t{buffer.data(), buffer.size()});
  } catch (const std::exception &exception) {
    ReportLogFailure(location, exception.what());
  } catch (...) {
    ReportLogFailure(location, "unknown error");
  }
}

}  // namespace internal

}  // namespace zephyr
