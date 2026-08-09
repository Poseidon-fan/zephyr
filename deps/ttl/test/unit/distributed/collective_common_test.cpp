#include <bit>
#include <cstdint>
#include <source_location>

#include <gtest/gtest.h>
#include <nccl.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/distributed/collective_common.hpp"

namespace ttl::internal {
namespace {

TEST(CollectiveCommonTest, MapsEverySupportedDTypeToNccLType) {
  EXPECT_EQ(ToNcclDType(DType::BOOL, std::source_location::current()), ncclUint8);
  EXPECT_EQ(ToNcclDType(DType::UINT8, std::source_location::current()), ncclUint8);
  EXPECT_EQ(ToNcclDType(DType::INT32, std::source_location::current()), ncclInt32);
  EXPECT_EQ(ToNcclDType(DType::INT64, std::source_location::current()), ncclInt64);
  EXPECT_EQ(ToNcclDType(DType::FLOAT16, std::source_location::current()), ncclFloat16);
  EXPECT_EQ(ToNcclDType(DType::BFLOAT16, std::source_location::current()), ncclBfloat16);
  EXPECT_EQ(ToNcclDType(DType::FLOAT32, std::source_location::current()), ncclFloat32);
}

TEST(CollectiveCommonTest, RejectsUnknownDType) {
  const auto invalid_dtype = std::bit_cast<DType>(uint8_t{255});
  EXPECT_THROW(static_cast<void>(ToNcclDType(invalid_dtype, std::source_location::current())), InvalidArgumentError);
}

TEST(CollectiveCommonTest, ComputesByteOffsetsWithoutChangingPointerValue) {
  std::uint8_t bytes[32]{};
  const void *const_pointer = bytes;
  void *mutable_pointer = bytes;
  EXPECT_EQ(CollectiveByteOffset(const_pointer, 7), bytes + 7);
  EXPECT_EQ(MutableCollectiveByteOffset(mutable_pointer, 19), bytes + 19);
  EXPECT_EQ(CollectiveByteOffset(nullptr, 0), nullptr);
}

}  // namespace
}  // namespace ttl::internal
