#include "support/test_tensor.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ttl::test {

auto BoolTensorFromValues(ExecutionContext &context, const Shape &shape, std::span<const uint8_t> values) -> Tensor {
  if (static_cast<int64_t>(values.size()) != shape.GetNumElements()) {
    throw std::invalid_argument("host value count does not match tensor shape");
  }
  if (!std::ranges::all_of(values, [](uint8_t value) { return value <= 1; })) {
    throw std::invalid_argument("boolean test values must be zero or one");
  }
  Tensor tensor = Empty(context, shape, DType::BOOL);
  CopyFromHostBlocking(context, tensor, AsBytes<uint8_t>(values));
  return tensor;
}

auto BoolTensorFromValues(ExecutionContext &context, const Shape &shape, std::initializer_list<uint8_t> values)
    -> Tensor {
  return BoolTensorFromValues(context, shape, std::span<const uint8_t>{values.begin(), values.size()});
}

auto BoolTensorToValues(ExecutionContext &context, const Tensor &tensor) -> std::vector<uint8_t> {
  if (tensor.GetDType() != DType::BOOL) {
    throw std::invalid_argument("requested bool values from a non-bool tensor");
  }
  std::vector<uint8_t> values(static_cast<size_t>(tensor.GetNumElements()));
  CopyToHostBlocking(context, AsWritableBytes<uint8_t>(values), tensor);
  return values;
}

auto FloatingTensorFromValues(ExecutionContext &context, const Shape &shape, DType dtype, std::span<const float> values)
    -> Tensor {
  switch (dtype) {
    case DType::FLOAT32:
      return TensorFromValues<float>(context, shape, values);
    case DType::FLOAT16: {
      std::vector<Float16> converted;
      converted.reserve(values.size());
      for (float value : values) {
        converted.push_back(FloatToFloat16(value));
      }
      return TensorFromValues<Float16>(context, shape, converted);
    }
    case DType::BFLOAT16: {
      std::vector<BFloat16> converted;
      converted.reserve(values.size());
      for (float value : values) {
        converted.push_back(FloatToBFloat16(value));
      }
      return TensorFromValues<BFloat16>(context, shape, converted);
    }
    default:
      throw std::invalid_argument("FloatingTensorFromValues requires a floating dtype");
  }
}

auto FloatingTensorFromValues(ExecutionContext &context, const Shape &shape, DType dtype,
                              std::initializer_list<float> values) -> Tensor {
  return FloatingTensorFromValues(context, shape, dtype, std::span<const float>{values.begin(), values.size()});
}

auto FloatingTensorToValues(ExecutionContext &context, const Tensor &tensor) -> std::vector<float> {
  switch (tensor.GetDType()) {
    case DType::FLOAT32:
      return TensorToValues<float>(context, tensor);
    case DType::FLOAT16: {
      const std::vector<Float16> values = TensorToValues<Float16>(context, tensor);
      std::vector<float> converted;
      converted.reserve(values.size());
      for (Float16 value : values) {
        converted.push_back(Float16ToFloat(value));
      }
      return converted;
    }
    case DType::BFLOAT16: {
      const std::vector<BFloat16> values = TensorToValues<BFloat16>(context, tensor);
      std::vector<float> converted;
      converted.reserve(values.size());
      for (BFloat16 value : values) {
        converted.push_back(BFloat16ToFloat(value));
      }
      return converted;
    }
    default:
      throw std::invalid_argument("FloatingTensorToValues requires a floating dtype");
  }
}

void ExpectShape(const Tensor &tensor, std::initializer_list<int64_t> dimensions) {
  const std::span<const int64_t> actual = tensor.GetShape().GetDimensions();
  ASSERT_EQ(actual.size(), dimensions.size());
  size_t index = 0;
  for (int64_t dimension : dimensions) {
    EXPECT_EQ(actual[index], dimension) << "at axis " << index;
    ++index;
  }
}

void ExpectFloatValues(ExecutionContext &context, const Tensor &tensor, std::span<const float> expected,
                       float absolute_tolerance, float relative_tolerance) {
  const std::vector<float> actual = FloatingTensorToValues(context, tensor);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    if (std::isnan(expected[index])) {
      EXPECT_TRUE(std::isnan(actual[index])) << "at element " << index;
      continue;
    }
    if (std::isinf(expected[index])) {
      EXPECT_EQ(actual[index], expected[index]) << "at element " << index;
      continue;
    }
    const float tolerance = absolute_tolerance + (relative_tolerance * std::abs(expected[index]));
    EXPECT_NEAR(actual[index], expected[index], tolerance) << "at element " << index;
  }
}

void ExpectFloatValues(ExecutionContext &context, const Tensor &tensor, std::initializer_list<float> expected,
                       float absolute_tolerance, float relative_tolerance) {
  ExpectFloatValues(context, tensor, std::span<const float>{expected.begin(), expected.size()}, absolute_tolerance,
                    relative_tolerance);
}

void ExpectBoolValues(ExecutionContext &context, const Tensor &tensor, std::initializer_list<uint8_t> expected) {
  const std::vector<uint8_t> actual = BoolTensorToValues(context, tensor);
  ASSERT_EQ(actual.size(), expected.size());
  size_t index = 0;
  for (uint8_t value : expected) {
    EXPECT_EQ(actual[index], value) << "at element " << index;
    ++index;
  }
}

}  // namespace ttl::test
