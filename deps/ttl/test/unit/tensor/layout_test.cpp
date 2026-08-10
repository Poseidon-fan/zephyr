#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"
#include "ttl/common/error.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/tensor/layout.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

class TensorViewTest : public test::SingleDeviceTest {};

TEST_F(TensorViewTest, CreatesViewsAndMaterializesOnlyWhenRequired) {
  auto &context = GetContext();
  auto input = test::Upload(context, Shape{2, 3}, std::vector<int32_t>{0, 1, 2, 3, 4, 5});

  auto viewed = View(input, Shape{3, 2});
  EXPECT_EQ(ClassifyAlias(input, viewed), AliasKind::MAY_OVERLAP);
  EXPECT_EQ(test::Download<int32_t>(context, viewed), (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));

  auto transposed = Transpose(input, 0, 1);
  EXPECT_EQ(transposed.GetShape(), Shape({3, 2}));
  auto reshaped = Reshape(context, transposed, Shape{6});
  EXPECT_EQ(ClassifyAlias(input, reshaped), AliasKind::DISJOINT);
  EXPECT_EQ(test::Download<int32_t>(context, reshaped), (std::vector<int32_t>{0, 3, 1, 4, 2, 5}));

  auto sliced = Slice(input, 1, 0, 3, 2);
  EXPECT_EQ(sliced.GetShape(), Shape({2, 2}));
  EXPECT_EQ(test::Download<int32_t>(context, Contiguous(context, sliced)), (std::vector<int32_t>{0, 2, 3, 5}));

  auto selected = Select(input, -1, 1);
  EXPECT_EQ(selected.GetShape(), Shape({2}));
  EXPECT_EQ(test::Download<int32_t>(context, Contiguous(context, selected)), (std::vector<int32_t>{1, 4}));

  auto empty_slice = Slice(input, 1, 2, 1, 1);
  EXPECT_EQ(empty_slice.GetShape(), Shape({2, 0}));
  EXPECT_TRUE(test::Download<int32_t>(context, Contiguous(context, empty_slice)).empty());
}

TEST_F(TensorViewTest, CoversSqueezeExpandSplitChunkAndInferenceErrors) {
  auto &context = GetContext();
  auto input = test::Upload(context, Shape{1, 4, 1}, std::vector<int64_t>{1, 2, 3, 4});

  auto squeezed = Squeeze(input);
  EXPECT_EQ(squeezed.GetShape(), Shape({4}));
  auto unsqueezed = Unsqueeze(squeezed, -1);
  EXPECT_EQ(unsqueezed.GetShape(), Shape({4, 1}));
  auto expanded = Expand(unsqueezed, Shape{4, 3});
  EXPECT_TRUE(expanded.HasZeroStride());
  EXPECT_EQ(test::Download<int64_t>(context, Contiguous(context, expanded)),
            (std::vector<int64_t>{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4}));

  const auto splits = Split(squeezed, std::array<int64_t, 3>{1, 2, 1}, 0);
  ASSERT_EQ(splits.size(), 3);
  EXPECT_EQ(test::Download<int64_t>(context, splits[1]), (std::vector<int64_t>{2, 3}));
  const auto chunks = Chunk(squeezed, 3, 0);
  ASSERT_EQ(chunks.size(), 2);
  EXPECT_EQ(chunks[0].GetShape(), Shape({2}));
  EXPECT_EQ(chunks[1].GetShape(), Shape({2}));

  EXPECT_EQ(InferReshape(squeezed, std::array<int64_t, 2>{2, -1}), Shape({2, 2}));
  EXPECT_THROW(static_cast<void>(InferReshape(squeezed, std::array<int64_t, 2>{-1, -1})), InvalidArgumentError);
  auto matrix = test::Upload(context, Shape{2, 3}, std::vector<int64_t>{0, 1, 2, 3, 4, 5});
  EXPECT_THROW(static_cast<void>(View(Transpose(matrix, 0, 1), Shape{6})), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Expand(squeezed, Shape{3})), InvalidArgumentError);
  EXPECT_THROW(static_cast<void>(Slice(squeezed, 0, 0, 4, 0)), InvalidArgumentError);
}

}  // namespace ttl
