#include <cmath>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "support/tensor_test_utils.hpp"
#include "ttl/ops/attention.hpp"
#include "ttl/ops/matmul.hpp"
#include "ttl/ops/random.hpp"
#include "ttl/runtime/generator.hpp"

namespace ttl {

TEST(MatmulTest, ComputesMatrixProductAndLinearBias) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto lhs = test::Upload(context, Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto rhs = test::Upload(context, Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto product = Matmul(context, lhs, rhs, MatmulOptions{.allow_tf32_ = false});
  EXPECT_EQ(test::Download<float>(context, product), (std::vector<float>{22, 28, 49, 64}));

  auto weight = test::Upload(context, Shape{2, 3}, std::vector<float>{1, 0, 0, 0, 1, 1});
  auto bias = test::Upload(context, Shape{2}, std::vector<float>{10, -10});
  auto linear = Linear(context, lhs, weight, std::optional<Tensor>{bias},
                       LinearOptions{.activation_ = LinearActivation::RELU,
                                     .gelu_approximation_ = GeluApproximation::NONE,
                                     .matmul_ = MatmulOptions{.allow_tf32_ = false}});
  EXPECT_EQ(test::Download<float>(context, linear), (std::vector<float>{11, 0, 14, 1}));
}

TEST(AttentionTest, ComputesReferenceScaledDotProductAttention) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  auto query = test::Upload(context, Shape{1, 1, 2, 2}, std::vector<float>{1, 0, 0, 1});
  auto key = test::Upload(context, Shape{1, 1, 2, 2}, std::vector<float>{1, 0, 0, 1});
  auto value = test::Upload(context, Shape{1, 1, 2, 1}, std::vector<float>{10, 20});
  auto output = ScaledDotProductAttention(context, query, key, value, std::nullopt,
                                          SdpaOptions{.scale_ = 1.0F, .causal_ = false});
  const auto values = test::Download<float>(context, output);
  EXPECT_NEAR(values[0], 12.689414F, 1.0e-4F);
  EXPECT_NEAR(values[1], 17.310586F, 1.0e-4F);
}

TEST(RandomTest, ReproduciblyAdvancesIndependentPhiloxGenerators) {
  test::RuntimeSession session;
  auto &context = session.GetContext();
  Generator first{context, 12345};
  Generator second{context, 12345};
  const auto options = UniformOptions{.low_ = -2.0, .high_ = 3.0};
  auto first_values = test::Download<float>(context, Uniform(context, Shape{257}, DType::FLOAT32, first, options));
  auto second_values = test::Download<float>(context, Uniform(context, Shape{257}, DType::FLOAT32, second, options));
  EXPECT_EQ(first_values, second_values);
  for (const auto value : first_values) {
    EXPECT_GE(value, -2.0F);
    EXPECT_LT(value, 3.0F);
  }

  auto next_values = test::Download<float>(context, Uniform(context, Shape{257}, DType::FLOAT32, first, options));
  EXPECT_NE(first_values, next_values);
}

}  // namespace ttl
