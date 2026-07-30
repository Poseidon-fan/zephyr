#include "ttl/internal/cublas_api.hpp"

#include <gtest/gtest.h>

namespace ttl::internal {
namespace {

TEST(ScopedCublasApiOverrideTest, OverridesAndRestoresTheProcessProviderInLifoOrder) {
  const auto *production_cublas_api = &GetCublasApi();
  auto outer_cublas_api = *production_cublas_api;
  auto inner_cublas_api = *production_cublas_api;

  {
    ScopedCublasApiOverride outer_override{outer_cublas_api};
    EXPECT_EQ(&GetCublasApi(), &outer_cublas_api);

    {
      ScopedCublasApiOverride inner_override{inner_cublas_api};
      EXPECT_EQ(&GetCublasApi(), &inner_cublas_api);
    }

    EXPECT_EQ(&GetCublasApi(), &outer_cublas_api);
  }

  EXPECT_EQ(&GetCublasApi(), production_cublas_api);
}

}  // namespace
}  // namespace ttl::internal
