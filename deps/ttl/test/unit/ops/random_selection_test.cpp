#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

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

TEST_F(RandomSelectionTest, TopKRadixSortPreservesFloatingRowsStridesAndSpecialValues) {
  constexpr int64_t width = 2053;
  for (const int64_t rows : {1, 3}) {
    std::vector<float> host(static_cast<size_t>(width * rows));
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t index = 0; index < width; ++index) {
        host[(index * rows) + row] = static_cast<float>(((index % 17) - 8) + (row * 32));
      }
      host[row] = std::numeric_limits<float>::quiet_NaN();
      host[rows + row] = -0.0F;
      host[(2 * rows) + row] = 0.0F;
      host[(3 * rows) + row] = std::numeric_limits<float>::infinity();
      host[(4 * rows) + row] = -std::numeric_limits<float>::infinity();
      host[(5 * rows) + row] = std::numeric_limits<float>::quiet_NaN();
    }
    for (const auto dtype : {DType::FLOAT32, DType::FLOAT16, DType::BFLOAT16}) {
      const auto input = FloatingTensorFromValues(GetContext(), Shape{width, rows}, dtype, host);
      for (const bool largest : {false, true}) {
        const auto [values, indices] = TopK(GetContext(), input, {.axis_ = 0, .k_ = width, .largest_ = largest});
        const auto actual_indices = TensorToValues<int64_t>(GetContext(), indices);
        const auto actual_values = FloatingTensorToValues(GetContext(), values);
        std::vector<int64_t> expected_indices(host.size());
        for (int64_t row = 0; row < rows; ++row) {
          std::vector<int64_t> order(static_cast<size_t>(width));
          std::iota(order.begin(), order.end(), int64_t{0});
          std::ranges::stable_sort(order, [&](int64_t left, int64_t right) {
            const auto lhs = host[(left * rows) + row];
            const auto rhs = host[(right * rows) + row];
            if (std::isnan(lhs) || std::isnan(rhs)) {
              return largest ? std::isnan(lhs) && !std::isnan(rhs) : !std::isnan(lhs) && std::isnan(rhs);
            }
            return largest ? lhs > rhs : lhs < rhs;
          });
          for (int64_t rank = 0; rank < width; ++rank) {
            const auto offset = (rank * rows) + row;
            expected_indices[offset] = order[rank];
            const auto expected = host[(order[rank] * rows) + row];
            if (std::isnan(expected)) {
              EXPECT_TRUE(std::isnan(actual_values[offset]));
            } else {
              EXPECT_EQ(actual_values[offset], expected);
              EXPECT_EQ(std::signbit(actual_values[offset]), std::signbit(expected));
            }
          }
        }
        EXPECT_EQ(actual_indices, expected_indices);
      }
    }
  }
}

TEST_F(RandomSelectionTest, TopKRadixSortPreservesIntegralRowsAndFullInt64Range) {
  const auto check = [this]<typename T>() {
    constexpr int64_t rows = 3;
    constexpr int64_t width = 2053;
    constexpr int64_t count = 19;
    std::vector<T> host(static_cast<size_t>(rows * width));
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t index = 0; index < width; ++index) {
        host[(row * width) + index] = static_cast<T>((((index * 13) + (row * 7)) % 71) - 35);
      }
      host[row * width] = std::numeric_limits<T>::min();
      host[(row * width) + 1] = std::numeric_limits<T>::max();
      host[(row * width) + 2] = std::numeric_limits<T>::max();
      if constexpr (std::is_same_v<T, int64_t>) {
        host[(row * width) + 3] = int64_t{1} << 50;
        host[(row * width) + 4] = -(int64_t{1} << 50);
      }
    }
    const auto input = TensorFromValues<T>(GetContext(), Shape{rows, width}, host);
    for (const bool largest : {false, true}) {
      const auto [values, indices] = TopK(GetContext(), input, {.axis_ = 1, .k_ = count, .largest_ = largest});
      std::vector<int64_t> expected_indices(static_cast<size_t>(rows * count));
      std::vector<T> expected_values(expected_indices.size());
      for (int64_t row = 0; row < rows; ++row) {
        std::vector<int64_t> order(static_cast<size_t>(width));
        std::iota(order.begin(), order.end(), int64_t{0});
        std::stable_sort(order.begin(), order.end(), [&](int64_t left, int64_t right) {
          const auto lhs = host[(row * width) + left];
          const auto rhs = host[(row * width) + right];
          return largest ? lhs > rhs : lhs < rhs;
        });
        for (int64_t rank = 0; rank < count; ++rank) {
          expected_indices[(row * count) + rank] = order[rank];
          expected_values[(row * count) + rank] = host[(row * width) + order[rank]];
        }
      }
      EXPECT_EQ(TensorToValues<int64_t>(GetContext(), indices), expected_indices);
      EXPECT_EQ(TensorToValues<T>(GetContext(), values), expected_values);
    }
  };
  check.operator()<uint8_t>();
  check.operator()<int32_t>();
  check.operator()<int64_t>();
}

}  // namespace
}  // namespace ttl::test
