#pragma once

#include <source_location>
#include <string_view>

namespace zephyr::internal {

/** Reports a failed debug assertion and terminates the process. */
[[noreturn]] void AssertionFailure(std::string_view expression, std::string_view message,
                                   std::source_location location) noexcept;

/** Reports a failed invariant required in every build and terminates the process. */
[[noreturn]] void EnsureFailure(std::string_view expression, std::string_view message,
                                std::source_location location) noexcept;

/** Reports an unreachable control-flow path and terminates the process. */
[[noreturn]] void UnreachableFailure(std::string_view message, std::source_location location) noexcept;

}  // namespace zephyr::internal

#ifndef NDEBUG
#define ZEPHYR_ASSERT(expr, message)                                                           \
  do {                                                                                         \
    if (!(expr)) {                                                                             \
      ::zephyr::internal::AssertionFailure(#expr, (message), std::source_location::current()); \
    }                                                                                          \
  } while (false)
#else
#define ZEPHYR_ASSERT(expr, message) ((void)0)
#endif

#define ZEPHYR_ENSURE(expr, message)                                                        \
  do {                                                                                      \
    if (!(expr)) {                                                                          \
      ::zephyr::internal::EnsureFailure(#expr, (message), std::source_location::current()); \
    }                                                                                       \
  } while (false)

#define ZEPHYR_UNIMPLEMENTED(message)                                                    \
  do {                                                                                   \
    throw ::zephyr::NotImplementedException((message), std::source_location::current()); \
  } while (false)

#define ZEPHYR_UNREACHABLE(message)                                                     \
  do {                                                                                  \
    ::zephyr::internal::UnreachableFailure((message), std::source_location::current()); \
  } while (false)
