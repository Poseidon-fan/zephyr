#include <array>
#include <cstdint>
#include <string>

#include "common/logger.hpp"
#include "gtest/gtest.h"

namespace zephyr {
namespace {

class LoggerTest : public ::testing::Test {
 protected:
  void SetUp() override { previous_level_ = GetLogLevel(); }

  void TearDown() override { SetLogLevel(previous_level_); }

 private:
  LogLevel previous_level_{LogLevel::OFF};
};

TEST_F(LoggerTest, SetsRuntimeLevel) {
  constexpr std::array levels{
      LogLevel::TRACE, LogLevel::DEBUG, LogLevel::INFO, LogLevel::WARN, LogLevel::ERROR, LogLevel::OFF,
  };

  for (const auto level : levels) {
    SetLogLevel(level);
    EXPECT_EQ(GetLogLevel(), level);
  }
}

TEST_F(LoggerTest, FormatsMessageAndPreservesSourceLocation) {
  SetLogLevel(LogLevel::INFO);
  testing::internal::CaptureStderr();
  const uint_least32_t expected_line = __LINE__ + 1;
  ZEPHYR_LOG_INFO("loaded {} on cuda:{}", "model", 2);
  const auto output = testing::internal::GetCapturedStderr();

  EXPECT_NE(output.find("[zephyr] [info]"), std::string::npos);
  EXPECT_NE(output.find("logger_test.cpp:" + std::to_string(expected_line)), std::string::npos);
  EXPECT_NE(output.find("loaded model on cuda:2"), std::string::npos);
}

TEST_F(LoggerTest, RuntimeFilterDoesNotEvaluateArguments) {
  SetLogLevel(LogLevel::ERROR);
  int evaluations = 0;

  testing::internal::CaptureStderr();
  ZEPHYR_LOG_INFO("evaluation {}", ++evaluations);
  const auto output = testing::internal::GetCapturedStderr();

  EXPECT_EQ(evaluations, 0);
  EXPECT_TRUE(output.empty());
}

TEST_F(LoggerTest, LogsEnabledLevels) {
  SetLogLevel(LogLevel::TRACE);
  testing::internal::CaptureStderr();
  ZEPHYR_LOG_TRACE("trace-message");
  ZEPHYR_LOG_DEBUG("debug-message");
  ZEPHYR_LOG_INFO("info-message");
  ZEPHYR_LOG_WARN("warn-message");
  ZEPHYR_LOG_ERROR("error-message");
  const auto output = testing::internal::GetCapturedStderr();

#if ZEPHYR_ACTIVE_LOG_LEVEL <= ZEPHYR_LOG_LEVEL_TRACE
  EXPECT_NE(output.find("trace-message"), std::string::npos);
#else
  EXPECT_EQ(output.find("trace-message"), std::string::npos);
#endif

#if ZEPHYR_ACTIVE_LOG_LEVEL <= ZEPHYR_LOG_LEVEL_DEBUG
  EXPECT_NE(output.find("debug-message"), std::string::npos);
#else
  EXPECT_EQ(output.find("debug-message"), std::string::npos);
#endif

  EXPECT_NE(output.find("info-message"), std::string::npos);
  EXPECT_NE(output.find("warn-message"), std::string::npos);
  EXPECT_NE(output.find("error-message"), std::string::npos);
}

TEST_F(LoggerTest, OffLevelDisablesLogging) {
  SetLogLevel(LogLevel::OFF);
  testing::internal::CaptureStderr();
  ZEPHYR_LOG_ERROR("disabled-message");
  const auto output = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(output.empty());
}

}  // namespace
}  // namespace zephyr
