#include <array>
#include <cstdint>
#include <source_location>
#include <string_view>
#include <utility>

#include "common/exception.hpp"
#include "gtest/gtest.h"

namespace zephyr {
namespace {

TEST(ExceptionTest, BaseException) {
  const auto location = std::source_location::current();
  Exception exception{ExceptionType::INTERNAL, "internal failure", location};

  EXPECT_EQ(exception.GetType(), ExceptionType::INTERNAL);
  EXPECT_STREQ(exception.what(), "internal failure");
  EXPECT_STREQ(exception.GetLocation().file_name(), location.file_name());
  EXPECT_EQ(exception.GetLocation().line(), location.line());
  EXPECT_EQ(exception.GetLocation().column(), location.column());
}

TEST(ExceptionTest, CapturesConstructionLocation) {
  const auto expected_line = __LINE__ + 1;
  const ConfigurationException exception{"invalid configuration"};

  EXPECT_STREQ(exception.GetLocation().file_name(), __FILE__);
  EXPECT_EQ(exception.GetLocation().line(), expected_line);
}

TEST(ExceptionTest, SpecializedExceptionForwardsLocation) {
  const auto location = std::source_location::current();
  const InternalException exception{"internal failure", location};

  EXPECT_STREQ(exception.GetLocation().file_name(), location.file_name());
  EXPECT_STREQ(exception.GetLocation().function_name(), location.function_name());
  EXPECT_EQ(exception.GetLocation().line(), location.line());
  EXPECT_EQ(exception.GetLocation().column(), location.column());
}

TEST(ExceptionTest, ExceptionTypeToString) {
  constexpr std::array<std::pair<ExceptionType, std::string_view>, 5> cases{{
      {ExceptionType::INVALID_ARGUMENT, "Invalid Argument"},
      {ExceptionType::NOT_IMPLEMENTED, "Not Implemented"},
      {ExceptionType::CONFIGURATION, "Configuration"},
      {ExceptionType::OUT_OF_MEMORY, "Out of Memory"},
      {ExceptionType::INTERNAL, "Internal"},
  }};

  for (const auto &[type, name] : cases) {
    EXPECT_EQ(Exception::ExceptionTypeToString(type), name);
  }
  // Exercise the defensive fallback with an unnamed value of the fixed underlying type.
  constexpr auto unknown_type =
      static_cast<ExceptionType>(UINT8_MAX);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(Exception::ExceptionTypeToString(unknown_type), "Unknown");
}

TEST(ExceptionTest, SpecializedExceptions) {
  const InvalidArgumentException invalid_argument{"invalid argument"};
  const NotImplementedException not_implemented{"not implemented"};
  const ConfigurationException configuration{"invalid configuration"};
  const OutOfMemoryException out_of_memory{"out of memory"};
  const InternalException internal{"internal failure"};

  EXPECT_EQ(invalid_argument.GetType(), ExceptionType::INVALID_ARGUMENT);
  EXPECT_EQ(not_implemented.GetType(), ExceptionType::NOT_IMPLEMENTED);
  EXPECT_EQ(configuration.GetType(), ExceptionType::CONFIGURATION);
  EXPECT_EQ(out_of_memory.GetType(), ExceptionType::OUT_OF_MEMORY);
  EXPECT_EQ(internal.GetType(), ExceptionType::INTERNAL);
}

TEST(ExceptionTest, CatchByBaseClass) {
  EXPECT_THROW(throw ConfigurationException{"invalid configuration"}, Exception);
  EXPECT_THROW(throw InternalException{"internal failure"}, Exception);
}

}  // namespace
}  // namespace zephyr
