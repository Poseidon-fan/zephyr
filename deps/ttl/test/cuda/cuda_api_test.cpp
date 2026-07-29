#include "ttl/internal/cuda_api.hpp"

#include <gtest/gtest.h>

namespace ttl::internal {
namespace {

TEST(ScopedCudaApiOverrideTest, OverridesAndRestoresTheProcessProviderInLifoOrder) {
  const auto *production_cuda_api = &GetCudaApi();
  auto outer_cuda_api = *production_cuda_api;
  auto inner_cuda_api = *production_cuda_api;

  {
    ScopedCudaApiOverride outer_override{outer_cuda_api};
    EXPECT_EQ(&GetCudaApi(), &outer_cuda_api);

    {
      ScopedCudaApiOverride inner_override{inner_cuda_api};
      EXPECT_EQ(&GetCudaApi(), &inner_cuda_api);
    }

    EXPECT_EQ(&GetCudaApi(), &outer_cuda_api);
  }

  EXPECT_EQ(&GetCudaApi(), production_cuda_api);
}

}  // namespace
}  // namespace ttl::internal
