#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/random.hpp"
#include "ttl/ops/topk.hpp"

namespace ttl::test {
namespace {

class RandomSelectionTest : public SingleDeviceTest {};

TEST_F(RandomSelectionTest, GeneratorSeedResetReproducesUniformAndNormalStreams) {
  Generator generator{GetContext(), 0x12345678ULL};
  EXPECT_EQ(generator.GetSeed(), 0x12345678ULL);
  Tensor first = Uniform(GetContext(), Shape{32}, DType::FLOAT32, generator, {.low_ = -2.0, .high_ = 3.0});
  generator.SetSeed(GetContext(), 0x12345678ULL);
  Tensor second = Uniform(GetContext(), Shape{32}, DType::FLOAT32, generator, {.low_ = -2.0, .high_ = 3.0});
  const auto first_values = FloatingTensorToValues(GetContext(), first);
  const auto second_values = FloatingTensorToValues(GetContext(), second);
  ASSERT_EQ(first_values, second_values);
  EXPECT_TRUE(std::ranges::all_of(first_values, [](float value) { return value >= -2.0F && value < 3.0F; }));

  generator.SetSeed(GetContext(), 9);
  Tensor normal =
      Normal(GetContext(), Shape{64}, DType::FLOAT32, generator, {.mean_ = 2.0, .standard_deviation_ = 0.0});
  ExpectFloatValues(GetContext(), normal, std::vector<float>(64, 2.0F));
}

TEST_F(RandomSelectionTest, RandomOperationsValidateDtypeBoundsAndGeneratorOwnership) {
  Generator generator{GetContext(), 1};
  EXPECT_THROW(static_cast<void>(Uniform(GetContext(), Shape{2}, DType::INT32, generator)), NotSupportedError);
  EXPECT_THROW(static_cast<void>(Uniform(GetContext(), Shape{2}, DType::FLOAT32, generator, {.low_ = 1, .high_ = 1})),
               InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(
                   Normal(GetContext(), Shape{2}, DType::FLOAT32, generator, {.mean_ = 0, .standard_deviation_ = -1})),
               InvalidArgumentError);

  ExecutionContext other = GetRuntime().CreateExecutionContext(GetDevice());
  Generator other_generator{other, 2};
  EXPECT_THROW(static_cast<void>(Uniform(GetContext(), Shape{2}, DType::FLOAT32, other_generator)),
               InvalidArgumentError);
  other.Synchronize();
}

TEST_F(RandomSelectionTest, TopKSelectsLargestAndSmallestWithStableTieBreaking) {
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{2, 5}, DType::FLOAT32, {3, 7, 7, 1, 2, 9, 4, 4, 4, 0});
  auto [largest_values, largest_indices] = TopK(GetContext(), input, {.axis_ = -1, .k_ = 3, .largest_ = true});
  ExpectFloatValues(GetContext(), largest_values, {7, 7, 3, 9, 4, 4});
  ExpectValues<int64_t>(GetContext(), largest_indices, {1, 2, 0, 0, 1, 2});

  auto [smallest_values, smallest_indices] = TopK(GetContext(), input, {.axis_ = 1, .k_ = 2, .largest_ = false});
  ExpectFloatValues(GetContext(), smallest_values, {1, 2, 0, 4});
  ExpectValues<int64_t>(GetContext(), smallest_indices, {3, 4, 4, 1});

  auto [empty_values, empty_indices] = TopK(GetContext(), input, {.axis_ = 1, .k_ = 0});
  EXPECT_EQ(empty_values.GetShape(), Shape({2, 0}));
  EXPECT_EQ(empty_indices.GetShape(), Shape({2, 0}));
}

TEST_F(RandomSelectionTest, TopKHandlesNanOrderingAndRejectsInvalidRequests) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor input = FloatingTensorFromValues(GetContext(), Shape{1, 4}, DType::FLOAT32, {1, nan, 3, 2});
  auto [values, indices] = TopK(GetContext(), input, {.axis_ = 1, .k_ = 2, .largest_ = true});
  const auto host_values = FloatingTensorToValues(GetContext(), values);
  EXPECT_TRUE(std::isnan(host_values[0]));
  EXPECT_EQ(TensorToValues<int64_t>(GetContext(), indices), (std::vector<int64_t>{1, 2}));

  EXPECT_THROW(static_cast<void>(TopK(GetContext(), input, {.axis_ = 1, .k_ = -1})), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(TopK(GetContext(), input, {.axis_ = 1, .k_ = 5})), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(TopK(GetContext(), input, {.axis_ = 2, .k_ = 1})), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl::test
