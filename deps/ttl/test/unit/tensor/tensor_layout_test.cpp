#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/tensor/layout.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::test {
namespace {

class TensorLayoutTest : public SingleDeviceTest {
 protected:
  [[nodiscard]] auto Sequence(const Shape &shape) -> Tensor {
    std::vector<int32_t> values(static_cast<size_t>(shape.GetNumElements()));
    for (size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<int32_t>(index);
    }
    return TensorFromValues<int32_t>(GetContext(), shape, values);
  }

  [[nodiscard]] auto ToContiguousValues(const Tensor &tensor) -> std::vector<int32_t> {
    Tensor contiguous = Contiguous(GetContext(), tensor);
    return TensorToValues<int32_t>(GetContext(), contiguous);
  }
};

TEST_F(TensorLayoutTest, EmptyAndEmptyStridedExposeValidatedImmutableMetadata) {
  Tensor contiguous = Empty(GetContext(), Shape{2, 3}, DType::FLOAT32);
  EXPECT_EQ(contiguous.GetDevice(), GetDevice());
  EXPECT_EQ(contiguous.GetDType(), DType::FLOAT32);
  EXPECT_EQ(contiguous.GetRank(), 2U);
  EXPECT_EQ(contiguous.GetNumElements(), 6);
  EXPECT_EQ(contiguous.GetShape(), Shape({2, 3}));
  EXPECT_EQ(contiguous.GetStrides(), Strides({3, 1}));
  EXPECT_EQ(contiguous.GetStorageOffset(), 0);
  EXPECT_TRUE(contiguous.IsContiguous());
  EXPECT_TRUE(contiguous.IsNonOverlappingDense());
  EXPECT_FALSE(contiguous.HasZeroStride());
  EXPECT_NE(contiguous.GetData<float>(), nullptr);
  EXPECT_THROW(static_cast<void>(contiguous.GetData<int32_t>()), InvalidArgumentError);

  Tensor column_major = EmptyStrided(GetContext(), Shape{2, 3}, Strides{1, 2}, DType::FLOAT32);
  EXPECT_FALSE(column_major.IsContiguous());
  EXPECT_TRUE(column_major.IsNonOverlappingDense());
  EXPECT_THROW(static_cast<void>(EmptyStrided(GetContext(), Shape{2, 3}, Strides{0, 1}, DType::FLOAT32)),
               InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(EmptyStrided(GetContext(), Shape{2}, Strides{1, 1}, DType::FLOAT32)),
               InvalidArgumentError);
}

TEST_F(TensorLayoutTest, CopyMoveAndAliasClassificationDistinguishExactOverlapAndDisjointRegions) {
  Tensor base = Sequence(Shape{8});
  Tensor copy = base;
  Tensor left = Narrow(base, 0, 0, 4);
  Tensor right = Narrow(base, 0, 4, 4);
  Tensor overlap = Narrow(base, 0, 2, 4);
  Tensor independent = Sequence(Shape{8});

  EXPECT_EQ(ClassifyAlias(base, copy), AliasKind::EXACT);
  EXPECT_EQ(ClassifyAlias(left, right), AliasKind::DISJOINT);
  EXPECT_EQ(ClassifyAlias(left, overlap), AliasKind::MAY_OVERLAP);
  EXPECT_EQ(ClassifyAlias(base, independent), AliasKind::DISJOINT);

  Tensor moved = std::move(copy);
  EXPECT_EQ(ClassifyAlias(base, moved), AliasKind::EXACT);
  // TTL specifies that moved-from handles are detectably invalid.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(static_cast<void>(copy.GetShape()), InvalidArgumentError);
}

TEST_F(TensorLayoutTest, ViewInferReshapeAndReshapeHandleScalarEmptyAndNonContiguousInputs) {
  Tensor base = Sequence(Shape{2, 3, 4});
  const std::array<int64_t, 2> inferred{3, -1};
  EXPECT_EQ(InferReshape(base, inferred), Shape({3, 8}));
  Tensor viewed = View(base, inferred);
  EXPECT_EQ(viewed.GetShape(), Shape({3, 8}));
  EXPECT_EQ(ClassifyAlias(base, viewed), AliasKind::MAY_OVERLAP);

  Tensor transposed = Transpose(base, 0, 1);
  EXPECT_THROW(static_cast<void>(View(transposed, Shape{6, 4})), InvalidArgumentError);
  Tensor materialized = Reshape(GetContext(), transposed, Shape{6, 4});
  EXPECT_TRUE(materialized.IsContiguous());
  EXPECT_EQ(ClassifyAlias(transposed, materialized), AliasKind::DISJOINT);

  Tensor scalar = Sequence(Shape{});
  EXPECT_EQ(View(scalar, Shape{1, 1}).GetStrides(), Strides({1, 1}));
  Tensor empty = Empty(GetContext(), Shape{0, 3}, DType::INT32);
  EXPECT_EQ(View(empty, Shape{0, 1, 3}).GetStrides(), Strides({3, 3, 1}));

  const std::array<int64_t, 2> two_inferred{-1, -1};
  EXPECT_THROW(static_cast<void>(InferReshape(base, two_inferred)), InvalidArgumentError);
  const std::array<int64_t, 2> incompatible{5, -1};
  EXPECT_THROW(static_cast<void>(InferReshape(base, incompatible)), InvalidArgumentError);
  const std::array<int64_t, 2> empty_inferred{0, -1};
  EXPECT_THROW(static_cast<void>(InferReshape(empty, empty_inferred)), InvalidArgumentError);
}

TEST_F(TensorLayoutTest, PermuteTransposeSqueezeUnsqueezeAndFlattenPreserveLogicalValues) {
  Tensor base = Sequence(Shape{2, 1, 3});
  const std::array<int64_t, 3> axes{2, 0, 1};
  Tensor permuted = Permute(base, axes);
  ExpectShape(permuted, {3, 2, 1});
  EXPECT_EQ(ToContiguousValues(permuted), (std::vector<int32_t>{0, 3, 1, 4, 2, 5}));

  Tensor transposed = Transpose(base, 0, -1);
  ExpectShape(transposed, {3, 1, 2});
  EXPECT_EQ(ToContiguousValues(transposed), (std::vector<int32_t>{0, 3, 1, 4, 2, 5}));

  Tensor squeezed = Squeeze(base);
  ExpectShape(squeezed, {2, 3});
  Tensor unsqueezed = Unsqueeze(squeezed, -1);
  ExpectShape(unsqueezed, {2, 3, 1});
  EXPECT_EQ(ToContiguousValues(unsqueezed), (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));

  Tensor flattened = Flatten(GetContext(), base, 0, 1);
  ExpectShape(flattened, {2, 3});
  EXPECT_EQ(ToContiguousValues(flattened), (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));

  const std::array<int64_t, 2> duplicate_axes{0, 0};
  EXPECT_THROW(static_cast<void>(Permute(squeezed, duplicate_axes)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Squeeze(squeezed, 0)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Flatten(GetContext(), base, 2, 0)), InvalidArgumentError);
}

TEST_F(TensorLayoutTest, NarrowSplitChunkSliceAndSelectUsePythonStyleBounds) {
  Tensor base = Sequence(Shape{3, 4});

  Tensor narrow = Narrow(base, 0, -2, 2);
  ExpectShape(narrow, {2, 4});
  EXPECT_EQ(ToContiguousValues(narrow), (std::vector<int32_t>{4, 5, 6, 7, 8, 9, 10, 11}));

  const std::array<int64_t, 3> sizes{1, 0, 3};
  const auto splits = Split(base, sizes, 1);
  ASSERT_EQ(splits.size(), 3U);
  ExpectShape(splits[0], {3, 1});
  ExpectShape(splits[1], {3, 0});
  ExpectShape(splits[2], {3, 3});

  const auto chunks = Chunk(base, 3, 1);
  ASSERT_EQ(chunks.size(), 2U);
  ExpectShape(chunks[0], {3, 2});
  ExpectShape(chunks[1], {3, 2});

  Tensor slice = Slice(base, 1, 1, -1, 2);
  ExpectShape(slice, {3, 1});
  EXPECT_EQ(ToContiguousValues(slice), (std::vector<int32_t>{1, 5, 9}));

  Tensor selected = Select(base, 0, -1);
  ExpectShape(selected, {4});
  EXPECT_EQ(ToContiguousValues(selected), (std::vector<int32_t>{8, 9, 10, 11}));

  EXPECT_THROW(static_cast<void>(Narrow(base, 0, 2, 2)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Split(base, std::array<int64_t, 2>{1, 2}, 1)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Chunk(base, 0, 0)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Slice(base, 0, std::nullopt, std::nullopt, 0)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Select(base, 0, 3)), InvalidArgumentError);
}

TEST_F(TensorLayoutTest, ExpandAndBroadcastShapesHandleLeadingSingletonAndZeroDimensions) {
  Tensor base = TensorFromValues<int32_t>(GetContext(), Shape{1, 3}, {1, 2, 3});
  Tensor expanded = Expand(base, Shape{2, 4, 3});
  ExpectShape(expanded, {2, 4, 3});
  EXPECT_TRUE(expanded.HasZeroStride());
  EXPECT_FALSE(expanded.IsNonOverlappingDense());
  EXPECT_EQ(ToContiguousValues(expanded),
            (std::vector<int32_t>{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}));

  const std::array<Shape, 3> shapes{Shape{2, 1, 3}, Shape{1, 4, 1}, Shape{3}};
  EXPECT_EQ(BroadcastShapes(shapes), Shape({2, 4, 3}));
  EXPECT_EQ(BroadcastShapes(std::span<const Shape>{}), Shape{});
  const std::array<Shape, 2> incompatible{Shape{2, 3}, Shape{4, 3}};
  EXPECT_THROW(static_cast<void>(BroadcastShapes(incompatible)), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Expand(base, Shape{2, 4})), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl::test
