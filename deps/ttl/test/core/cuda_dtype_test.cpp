#include "ttl/internal/cuda_dtype.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ttl::internal {

using testing::HasSubstr;

inline constexpr auto DTYPE_VISITOR = []<CudaStorageType T>(std::type_identity<T>) { return CUDA_DTYPE_OF<T>; };

static_assert(std::same_as<CudaTypeForT<DType::BOOL>, bool>);
static_assert(std::same_as<CudaTypeForT<DType::UINT8>, uint8_t>);
static_assert(std::same_as<CudaTypeForT<DType::INT32>, int32_t>);
static_assert(std::same_as<CudaTypeForT<DType::INT64>, int64_t>);
static_assert(std::same_as<CudaTypeForT<DType::FLOAT16>, __half>);
static_assert(std::same_as<CudaTypeForT<DType::BFLOAT16>, __nv_bfloat16>);
static_assert(std::same_as<CudaTypeForT<DType::FLOAT32>, float>);

static_assert(CudaStorageType<bool>);
static_assert(CudaStorageType<__half>);
static_assert(CudaStorageType<__nv_bfloat16>);
static_assert(!CudaStorageType<Float16>);
static_assert(!CudaStorageType<BFloat16>);
static_assert(CUDA_DTYPE_OF<bool> == DType::BOOL);
static_assert(CUDA_DTYPE_OF<const __half> == DType::FLOAT16);
static_assert(CUDA_DTYPE_OF<__nv_bfloat16> == DType::BFLOAT16);

TEST(CudaDTypeDispatchTest, DispatchesEverySupportedType) {
  constexpr std::array dtypes{
      DType::BOOL, DType::UINT8, DType::INT32, DType::INT64, DType::FLOAT16, DType::BFLOAT16, DType::FLOAT32,
  };

  for (const auto dtype : dtypes) {
    const auto dispatched = DispatchCudaDType(dtype, "dispatch test", DTYPE_VISITOR);
    EXPECT_EQ(dispatched, dtype);
  }
}

TEST(CudaDTypeDispatchTest, DispatchesCategorySpecificTypes) {
  EXPECT_EQ(DispatchCudaFloatingDType(DType::FLOAT16, "floating dispatch", DTYPE_VISITOR), DType::FLOAT16);
  EXPECT_EQ(DispatchCudaFloatingDType(DType::BFLOAT16, "floating dispatch", DTYPE_VISITOR), DType::BFLOAT16);
  EXPECT_EQ(DispatchCudaIntegralDType(DType::INT64, "integral dispatch", DTYPE_VISITOR), DType::INT64);
  EXPECT_EQ(DispatchCudaNumericDType(DType::UINT8, "numeric dispatch", DTYPE_VISITOR), DType::UINT8);
}

TEST(CudaDTypeDispatchTest, RejectsUnsupportedCategories) {
  EXPECT_THROW(static_cast<void>(DispatchCudaFloatingDType(DType::INT32, "softmax", DTYPE_VISITOR)), NotSupportedError);
  EXPECT_THROW(static_cast<void>(DispatchCudaNumericDType(DType::BOOL, "add", DTYPE_VISITOR)), NotSupportedError);

  try {
    static_cast<void>(DispatchCudaFloatingDType(DType::INT32, "softmax", DTYPE_VISITOR));
  } catch (const NotSupportedError &error) {
    EXPECT_THAT(error.GetMessage(), HasSubstr("softmax"));
    EXPECT_THAT(error.GetMessage(), HasSubstr("int32"));
    EXPECT_THAT(error.GetMessage(), HasSubstr("floating dtype"));
    return;
  }
  FAIL() << "expected NotSupportedError";
}

TEST(CudaDTypeDispatchTest, RejectsInvalidEnumValues) {
  constexpr auto invalid = static_cast<DType>(0xFF);
  EXPECT_THROW(static_cast<void>(DispatchCudaDType(invalid, "copy", DTYPE_VISITOR)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(DispatchCudaIntegralDType(invalid, "index", DTYPE_VISITOR)), InvalidArgumentError);
}

}  // namespace ttl::internal
