#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ttl/ops/copy.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::test {

template <typename T>
[[nodiscard]] auto AsBytes(std::span<const T> values) -> std::span<const std::byte> {
  return {reinterpret_cast<const std::byte *>(values.data()), values.size_bytes()};
}

template <typename T>
[[nodiscard]] auto AsWritableBytes(std::span<T> values) -> std::span<std::byte> {
  return {reinterpret_cast<std::byte *>(values.data()), values.size_bytes()};
}

template <TensorStorageType T>
[[nodiscard]] auto TensorFromValues(ExecutionContext &context, const Shape &shape, std::span<const T> values)
    -> Tensor {
  if (static_cast<int64_t>(values.size()) != shape.GetNumElements()) {
    throw std::invalid_argument("host value count does not match tensor shape");
  }
  Tensor tensor = Empty(context, shape, DTYPE_OF<T>);
  CopyFromHostBlocking(context, tensor, AsBytes<T>(values));
  return tensor;
}

template <TensorStorageType T>
[[nodiscard]] auto TensorFromValues(ExecutionContext &context, const Shape &shape, std::initializer_list<T> values)
    -> Tensor {
  return TensorFromValues<T>(context, shape, std::span<const T>{values.begin(), values.size()});
}

template <TensorStorageType T>
[[nodiscard]] auto TensorToValues(ExecutionContext &context, const Tensor &tensor) -> std::vector<T> {
  if (tensor.GetDType() != DTYPE_OF<T>) {
    throw std::invalid_argument("requested host dtype does not match tensor dtype");
  }
  std::vector<T> values(static_cast<size_t>(tensor.GetNumElements()));
  CopyToHostBlocking(context, AsWritableBytes<T>(values), tensor);
  return values;
}

[[nodiscard]] auto BoolTensorFromValues(ExecutionContext &context, const Shape &shape, std::span<const uint8_t> values)
    -> Tensor;
[[nodiscard]] auto BoolTensorFromValues(ExecutionContext &context, const Shape &shape,
                                        std::initializer_list<uint8_t> values) -> Tensor;
[[nodiscard]] auto BoolTensorToValues(ExecutionContext &context, const Tensor &tensor) -> std::vector<uint8_t>;

[[nodiscard]] auto FloatingTensorFromValues(ExecutionContext &context, const Shape &shape, DType dtype,
                                            std::span<const float> values) -> Tensor;
[[nodiscard]] auto FloatingTensorFromValues(ExecutionContext &context, const Shape &shape, DType dtype,
                                            std::initializer_list<float> values) -> Tensor;
[[nodiscard]] auto FloatingTensorToValues(ExecutionContext &context, const Tensor &tensor) -> std::vector<float>;

void ExpectShape(const Tensor &tensor, std::initializer_list<int64_t> dimensions);
void ExpectFloatValues(ExecutionContext &context, const Tensor &tensor, std::span<const float> expected,
                       float absolute_tolerance = 1.0e-5F, float relative_tolerance = 1.0e-5F);
void ExpectFloatValues(ExecutionContext &context, const Tensor &tensor, std::initializer_list<float> expected,
                       float absolute_tolerance = 1.0e-5F, float relative_tolerance = 1.0e-5F);

template <TensorStorageType T>
void ExpectValues(ExecutionContext &context, const Tensor &tensor, std::initializer_list<T> expected) {
  const std::vector<T> actual = TensorToValues<T>(context, tensor);
  ASSERT_EQ(actual.size(), expected.size());
  size_t index = 0;
  for (const T &value : expected) {
    EXPECT_EQ(actual[index], value) << "at element " << index;
    ++index;
  }
}

void ExpectBoolValues(ExecutionContext &context, const Tensor &tensor, std::initializer_list<uint8_t> expected);

}  // namespace ttl::test
