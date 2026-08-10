#include <source_location>
#include <string>

#include <gtest/gtest.h>

#include "ttl/common/error.hpp"

namespace ttl::test {

TEST(ErrorTest, PreservesStructuredCodeMessageAndLocation) {
  const auto location = std::source_location::current();
  const InvalidArgumentError error{"bad tensor", location};
  EXPECT_EQ(error.GetCode(), ErrorCode::INVALID_ARGUMENT);
  EXPECT_EQ(error.GetMessage(), "bad tensor");
  EXPECT_EQ(error.GetLocation().line(), location.line());
  EXPECT_NE(std::string{error.what()}.find("INVALID_ARGUMENT"), std::string::npos);
  EXPECT_NE(std::string{error.what()}.find("bad tensor"), std::string::npos);
}

TEST(ErrorTest, NamesEveryStableErrorCode) {
  EXPECT_EQ(ErrorCodeToString(ErrorCode::INVALID_ARGUMENT), "INVALID_ARGUMENT");
  EXPECT_EQ(ErrorCodeToString(ErrorCode::OVERFLOW), "OVERFLOW");
  EXPECT_EQ(ErrorCodeToString(ErrorCode::NOT_SUPPORTED), "NOT_SUPPORTED");
  EXPECT_EQ(ErrorCodeToString(ErrorCode::OUT_OF_MEMORY), "OUT_OF_MEMORY");
  EXPECT_EQ(ErrorCodeToString(ErrorCode::CUDA), "CUDA");
  EXPECT_EQ(ErrorCodeToString(ErrorCode::CUBLAS), "CUBLAS");
  EXPECT_EQ(ErrorCodeToString(ErrorCode::NCCL), "NCCL");
  EXPECT_EQ(ErrorCodeToString(ErrorCode::CAPTURE), "CAPTURE");
  EXPECT_EQ(ErrorCodeToString(ErrorCode::ASYNC_EXECUTION), "ASYNC_EXECUTION");
  EXPECT_EQ(ErrorCodeToString(ErrorCode::INTERNAL), "INTERNAL");
  EXPECT_EQ(ErrorCodeToString(static_cast<ErrorCode>(255)), "UNKNOWN");
}

}  // namespace ttl::test
