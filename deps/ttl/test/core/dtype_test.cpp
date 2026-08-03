#include <gtest/gtest.h>

#include "ttl/dtype.hpp"

namespace ttl {

TEST(DTypeTest, ReportsFloat32Metadata) {
  EXPECT_EQ(GetDTypeName(DType::FLOAT32), "float32");
  EXPECT_EQ(GetDTypeSize(DType::FLOAT32), sizeof(float));
}

}  // namespace ttl
