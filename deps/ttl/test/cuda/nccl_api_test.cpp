#include "ttl/internal/nccl_api.hpp"

#include <gtest/gtest.h>
#include <nccl.h>

namespace ttl::internal {
namespace {

auto FakeGetVersionOne(int *version) -> ncclResult_t {
  *version = 1;
  return ncclSuccess;
}

auto FakeGetVersionTwo(int *version) -> ncclResult_t {
  *version = 2;
  return ncclSuccess;
}

TEST(NcclApiTest, RestoresNestedOverridesInLifoOrder) {
  const auto &production_api = GetNcclApi();
  auto first_api = production_api;
  auto second_api = production_api;
  first_api.get_version_ = FakeGetVersionOne;
  second_api.get_version_ = FakeGetVersionTwo;

  {
    const ScopedNcclApiOverride first_override{first_api};
    int version = 0;
    EXPECT_EQ(GetNcclApi().get_version_(&version), ncclSuccess);
    EXPECT_EQ(version, 1);

    {
      const ScopedNcclApiOverride second_override{second_api};
      EXPECT_EQ(GetNcclApi().get_version_(&version), ncclSuccess);
      EXPECT_EQ(version, 2);
    }

    EXPECT_EQ(GetNcclApi().get_version_(&version), ncclSuccess);
    EXPECT_EQ(version, 1);
  }

  EXPECT_EQ(&GetNcclApi(), &production_api);
}

}  // namespace
}  // namespace ttl::internal
