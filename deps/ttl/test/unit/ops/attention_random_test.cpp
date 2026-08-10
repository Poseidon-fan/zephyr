#include <cmath>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/ops/attention.hpp"
#include "ttl/ops/matmul.hpp"
#include "ttl/ops/random.hpp"
#include "ttl/runtime/generator.hpp"
#include "ttl/tensor/layout.hpp"

namespace ttl::test {
namespace {

class AttentionRandomTest : public SingleDeviceTest {};

TEST_F(AttentionRandomTest, ComputesMatrixProductAndLinearBias) {
  auto &context = GetContext();
  auto lhs = Upload(context, Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto rhs = Upload(context, Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto product = Matmul(context, lhs, rhs, MatmulOptions{.allow_tf32_ = false});
  EXPECT_EQ(Download<float>(context, product), (std::vector<float>{22, 28, 49, 64}));

  auto weight = Upload(context, Shape{2, 3}, std::vector<float>{1, 0, 0, 0, 1, 1});
  auto bias = Upload(context, Shape{2}, std::vector<float>{10, -10});
  auto linear = Linear(context, lhs, weight, std::optional<Tensor>{bias},
                       LinearOptions{.activation_ = LinearActivation::RELU,
                                     .gelu_approximation_ = GeluApproximation::NONE,
                                     .matmul_ = MatmulOptions{.allow_tf32_ = false}});
  EXPECT_EQ(Download<float>(context, linear), (std::vector<float>{11, 0, 14, 1}));
}

TEST_F(AttentionRandomTest, AppliesStridedLinearBiasBeforeActivation) {
  auto &context = GetContext();
  auto input = Upload(context, Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  auto weight = Upload(context, Shape{2, 3}, std::vector<float>{1, 0, 0, 0, 1, 1});
  auto bias_storage = Upload(context, Shape{4}, std::vector<float>{10, 1000, -10, 1000});
  auto bias = Slice(bias_storage, 0, 0, 4, 2);
  ASSERT_FALSE(bias.IsContiguous());

  const auto without_activation = Linear(context, input, weight, std::optional<Tensor>{bias},
                                         LinearOptions{.activation_ = LinearActivation::NONE,
                                                       .gelu_approximation_ = GeluApproximation::NONE,
                                                       .matmul_ = MatmulOptions{.allow_tf32_ = false}});
  EXPECT_EQ(Download<float>(context, without_activation), (std::vector<float>{11, -5, 14, 1}));

  const auto with_relu = Linear(context, input, weight, std::optional<Tensor>{bias},
                                LinearOptions{.activation_ = LinearActivation::RELU,
                                              .gelu_approximation_ = GeluApproximation::NONE,
                                              .matmul_ = MatmulOptions{.allow_tf32_ = false}});
  EXPECT_EQ(Download<float>(context, with_relu), (std::vector<float>{11, 0, 14, 1}));
}

TEST_F(AttentionRandomTest, ComputesReferenceScaledDotProductAttention) {
  auto &context = GetContext();
  auto query = Upload(context, Shape{1, 1, 2, 2}, std::vector<float>{1, 0, 0, 1});
  auto key = Upload(context, Shape{1, 1, 2, 2}, std::vector<float>{1, 0, 0, 1});
  auto value = Upload(context, Shape{1, 1, 2, 1}, std::vector<float>{10, 20});
  auto output = ScaledDotProductAttention(context, query, key, value, std::nullopt,
                                          SdpaOptions{.scale_ = 1.0F, .causal_ = false});
  const auto values = Download<float>(context, output);
  EXPECT_NEAR(values[0], 12.689414F, 1.0e-4F);
  EXPECT_NEAR(values[1], 17.310586F, 1.0e-4F);
}

TEST_F(AttentionRandomTest, ReproduciblyAdvancesIndependentPhiloxGenerators) {
  auto &context = GetContext();
  Generator first{context, 12345};
  Generator second{context, 12345};
  const auto options = UniformOptions{.low_ = -2.0, .high_ = 3.0};
  auto first_values = Download<float>(context, Uniform(context, Shape{257}, DType::FLOAT32, first, options));
  auto second_values = Download<float>(context, Uniform(context, Shape{257}, DType::FLOAT32, second, options));
  EXPECT_EQ(first_values, second_values);
  for (const auto value : first_values) {
    EXPECT_GE(value, -2.0F);
    EXPECT_LT(value, 3.0F);
  }

  auto next_values = Download<float>(context, Uniform(context, Shape{257}, DType::FLOAT32, first, options));
  EXPECT_NE(first_values, next_values);
}

}  // namespace
}  // namespace ttl::test
