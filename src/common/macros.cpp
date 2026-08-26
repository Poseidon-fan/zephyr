#include "common/macros.hpp"

#include <cstdio>
#include <cstdlib>
#include <source_location>
#include <string_view>

namespace zephyr {

namespace {

/** Writes a string view without requiring a null terminator. */
void PrintString(std::string_view value) noexcept {
  if (!value.empty()) {
    static_cast<void>(std::fwrite(value.data(), sizeof(char), value.size(), stderr));
  }
}

/** Writes an optional diagnostic suffix and flushes stderr before termination. */
void PrintMessage(std::string_view message) noexcept {
  if (!message.empty()) {
    std::fputs(": ", stderr);
    PrintString(message);
  }
  std::fputc('\n', stderr);
  std::fflush(stderr);
}

}  // namespace

namespace internal {

void AssertionFailure(std::string_view expression, std::string_view message, std::source_location location) noexcept {
  std::fprintf(stderr, "%s:%u in %s: assertion `", location.file_name(), static_cast<unsigned int>(location.line()),
               location.function_name());
  PrintString(expression);
  std::fputs("` failed", stderr);
  PrintMessage(message);
  std::abort();
}

void EnsureFailure(std::string_view expression, std::string_view message, std::source_location location) noexcept {
  std::fprintf(stderr, "%s:%u in %s: ensure `", location.file_name(), static_cast<unsigned int>(location.line()),
               location.function_name());
  PrintString(expression);
  std::fputs("` failed", stderr);
  PrintMessage(message);
  std::abort();
}

void UnreachableFailure(std::string_view message, std::source_location location) noexcept {
  std::fprintf(stderr, "%s:%u in %s: unreachable code reached", location.file_name(),
               static_cast<unsigned int>(location.line()), location.function_name());
  PrintMessage(message);
  std::abort();
}

}  // namespace internal

}  // namespace zephyr
