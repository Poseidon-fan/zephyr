#include "ttl/error.hpp"

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ttl {

using testing::HasSubstr;

TEST(ErrorCodeTest, ConvertsEveryCodeToString) {
  EXPECT_EQ(ErrorCodeToString(ErrorCode::INVALID_ARGUMENT), "INVALID_ARGUMENT");
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

TEST(ErrorTest, PreservesStructuredFieldsAndFormatsWhat) {
  const auto expected_line = __LINE__ + 1;
  const InvalidArgumentError error("shape contains a negative dimension");

  EXPECT_EQ(error.GetCode(), ErrorCode::INVALID_ARGUMENT);
  EXPECT_EQ(error.GetMessage(), "shape contains a negative dimension");
  EXPECT_EQ(error.GetLocation().line(), expected_line);
  EXPECT_THAT(std::string_view(error.GetLocation().file_name()), HasSubstr("error_test.cpp"));
  EXPECT_THAT(std::string_view(error.what()), HasSubstr("[INVALID_ARGUMENT]"));
  EXPECT_THAT(std::string_view(error.what()), HasSubstr("shape contains a negative dimension"));
  EXPECT_THAT(std::string_view(error.what()), HasSubstr("error_test.cpp"));
}

TEST(ErrorTest, DerivedErrorsAreCatchableAsError) {
  EXPECT_THROW({ throw CudaError("cudaSetDevice failed"); }, Error);
}

}  // namespace ttl
