#include <cstdint>

#include "common/exception.h"
#include "common/tensor_type.h"
#include "gtest/gtest.h"

namespace zephyr {
namespace {

TEST(TensorTypeTest, SupportsStaticAndDynamicDimensions) {
  const auto type = TensorType{
      .dtype_ = ttl::DType::BFLOAT16,
      .shape_ = {DynamicDimension{"tokens"}, int64_t{4096}},
  };

  EXPECT_EQ(type.dtype_, ttl::DType::BFLOAT16);
  ASSERT_EQ(type.shape_.size(), 2U);
  EXPECT_EQ(std::get<DynamicDimension>(type.shape_[0]).name_, "tokens");
  EXPECT_EQ(std::get<int64_t>(type.shape_[1]), 4096);
  EXPECT_EQ(type.ToString(), "tensor<[tokens, 4096], bf16>");
}

TEST(TensorTypeTest, FormatsScalarAndStaticTypes) {
  EXPECT_EQ((TensorType{.dtype_ = ttl::DType::FLOAT32, .shape_ = {}}).ToString(), "tensor<[], f32>");
  EXPECT_EQ((TensorType{.dtype_ = ttl::DType::INT64, .shape_ = {int64_t{2}, int64_t{3}}}).ToString(),
            "tensor<[2, 3], i64>");
}

TEST(TensorTypeTest, RejectsInvalidDTypeWhenFormatted) {
  const auto type = TensorType{.dtype_ = static_cast<ttl::DType>(255), .shape_ = {}};
  EXPECT_THROW(type.ToString(), InvalidArgumentException);
}

TEST(TensorTypeTest, SupportsUint8) {
  const auto type = TensorType{.dtype_ = ttl::DType::UINT8, .shape_ = {int64_t{8}}};
  EXPECT_EQ(type.ToString(), "tensor<[8], u8>");
}

TEST(TensorTypeTest, EqualityIncludesShapeAndDtype) {
  const auto first = TensorType{.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{2}, int64_t{3}}};
  const auto same = TensorType{.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{2}, int64_t{3}}};
  const auto different_dtype = TensorType{.dtype_ = ttl::DType::FLOAT16, .shape_ = {int64_t{2}, int64_t{3}}};
  const auto different_shape = TensorType{.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{2}, int64_t{4}}};

  EXPECT_EQ(first, same);
  EXPECT_NE(first, different_dtype);
  EXPECT_NE(first, different_shape);
}

TEST(TensorTypeTest, TensorSliceUsesHalfOpenRanges) {
  const auto slice = TensorSlice{{.begin_ = 0, .end_ = 4}, {.begin_ = 2, .end_ = 8}};

  EXPECT_EQ(slice[0].begin_, 0);
  EXPECT_EQ(slice[0].end_, 4);
  EXPECT_EQ(slice[1].begin_, 2);
  EXPECT_EQ(slice[1].end_, 8);
}

}  // namespace
}  // namespace zephyr
