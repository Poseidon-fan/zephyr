#include <memory>
#include <utility>
#include <vector>

#include "common/exception.h"
#include "common/tensor_type.h"
#include "gtest/gtest.h"
#include "ir/model.h"
#include "ir/operation/core.hpp"
#include "ir/operation/tensor.hpp"

namespace zephyr::ir {
namespace {

auto MakeInput(std::string name, TensorType type) -> std::unique_ptr<const Input> {
  return std::make_unique<const Input>(std::move(name), std::move(type));
}

auto MakeParameter(std::string name, TensorType type) -> std::unique_ptr<const Parameter> {
  return std::make_unique<const Parameter>(std::move(name), std::move(type));
}

TEST(IrValueTest, FormatsValueReferences) {
  const auto input = Input{"tokens", {.dtype_ = ttl::DType::INT64, .shape_ = {int64_t{4}}}};
  const auto parameter = Parameter{"weight", {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{4}, int64_t{4}}}};
  EXPECT_EQ(input.ToString({}), "%tokens");
  EXPECT_EQ(parameter.ToString({}), "@weight");
}

TEST(IrOperationTest, InfersLinearOutputType) {
  const auto input =
      Input{"hidden", {.dtype_ = ttl::DType::BFLOAT16, .shape_ = {DynamicDimension{TOKEN_DIMENSION}, int64_t{4}}}};
  const auto weight = Parameter{"weight", {.dtype_ = ttl::DType::BFLOAT16, .shape_ = {int64_t{8}, int64_t{4}}}};
  const auto operation = Linear{&input, &weight};
  ASSERT_EQ(operation.GetResultTypes().size(), 1U);
  EXPECT_EQ(operation.GetResultTypes()[0],
            (TensorType{.dtype_ = ttl::DType::BFLOAT16, .shape_ = {DynamicDimension{TOKEN_DIMENSION}, int64_t{8}}}));
  EXPECT_NO_THROW(operation.Verify());
}

TEST(IrOperationTest, BroadcastsElementwiseOperands) {
  const auto lhs = Input{"lhs", {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{2}, int64_t{4}}}};
  const auto rhs = Input{"rhs", {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{1}, int64_t{4}}}};
  const auto operation = Add{&lhs, &rhs};
  EXPECT_EQ(operation.GetResultTypes()[0], lhs.GetType());
  EXPECT_NO_THROW(operation.Verify());
}

TEST(IrOperationTest, RejectsInvalidReshape) {
  const auto input = Input{"input", {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{2}, int64_t{4}}}};
  EXPECT_THROW((static_cast<void>(Reshape{&input, Shape{int64_t{3}, int64_t{3}}})), InvalidArgumentException);
}

TEST(IrModelTest, PrintsCompleteModelThroughOperationDispatch) {
  auto input =
      MakeInput("hidden", {.dtype_ = ttl::DType::BFLOAT16, .shape_ = {DynamicDimension{TOKEN_DIMENSION}, int64_t{4}}});
  auto weight = MakeParameter("weight", {.dtype_ = ttl::DType::BFLOAT16, .shape_ = {int64_t{8}, int64_t{4}}});
  const auto *input_pointer = input.get();
  const auto *weight_pointer = weight.get();
  auto linear = std::make_unique<const Linear>(input_pointer, weight_pointer);
  const auto *linear_pointer = linear.get();

  auto inputs = std::vector<std::unique_ptr<const Input>>{};
  inputs.push_back(std::move(input));
  auto parameters = std::vector<std::unique_ptr<const Parameter>>{};
  parameters.push_back(std::move(weight));
  auto operations = std::vector<std::unique_ptr<const Operation>>{};
  operations.push_back(std::move(linear));
  auto model = Model{"example", std::move(inputs), std::move(parameters), std::move(operations),
                     std::vector<const Value *>{&linear_pointer->GetResult(0)}};

  EXPECT_NO_THROW(model.Verify());
  ASSERT_EQ(input_pointer->GetUsers().size(), 1U);
  EXPECT_EQ(input_pointer->GetUsers()[0].user_, linear_pointer);
  EXPECT_EQ(input_pointer->GetUsers()[0].operand_index_, 0U);
  ASSERT_EQ(weight_pointer->GetUsers().size(), 1U);
  EXPECT_EQ(weight_pointer->GetUsers()[0].user_, linear_pointer);
  EXPECT_EQ(weight_pointer->GetUsers()[0].operand_index_, 1U);
  EXPECT_TRUE(linear_pointer->GetResult(0).GetUsers().empty());
  EXPECT_EQ(model.ToString(),
            "model @example {\n"
            "  input %hidden : tensor<[tokens, 4], bf16>\n"
            "\n"
            "  parameter @weight : tensor<[8, 4], bf16>\n"
            "\n"
            "  %0 = core.linear(%hidden, @weight) : tensor<[tokens, 8], bf16>\n"
            "  return (%0)\n"
            "}");
}

TEST(IrModelTest, RejectsNonTopologicalOperationOrder) {
  auto input = MakeInput("hidden", {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{4}}});
  auto weight = MakeParameter("weight", {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{4}, int64_t{4}}});
  const auto *input_pointer = input.get();
  const auto *weight_pointer = weight.get();
  auto first = std::make_unique<const Linear>(input_pointer, weight_pointer);
  const auto *first_pointer = first.get();
  auto second = std::make_unique<const Add>(&first_pointer->GetResult(0), &first_pointer->GetResult(0));

  auto inputs = std::vector<std::unique_ptr<const Input>>{};
  inputs.push_back(std::move(input));
  auto parameters = std::vector<std::unique_ptr<const Parameter>>{};
  parameters.push_back(std::move(weight));
  auto operations = std::vector<std::unique_ptr<const Operation>>{};
  operations.push_back(std::move(second));
  operations.push_back(std::move(first));
  auto model = Model{"example", std::move(inputs), std::move(parameters), std::move(operations), {}};
  EXPECT_THROW(model.Verify(), ConfigurationException);
}

TEST(IrModelTest, TracksValueUseListsOnOperationResults) {
  auto input = MakeInput("hidden", {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{4}}});
  auto weight = MakeParameter("weight", {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{4}, int64_t{4}}});
  const auto *input_pointer = input.get();
  const auto *weight_pointer = weight.get();
  auto linear = std::make_unique<const Linear>(input_pointer, weight_pointer);
  const auto *linear_pointer = linear.get();
  auto add = std::make_unique<const Add>(&linear_pointer->GetResult(0), &linear_pointer->GetResult(0));
  const auto *add_pointer = add.get();

  auto inputs = std::vector<std::unique_ptr<const Input>>{};
  inputs.push_back(std::move(input));
  auto parameters = std::vector<std::unique_ptr<const Parameter>>{};
  parameters.push_back(std::move(weight));
  auto operations = std::vector<std::unique_ptr<const Operation>>{};
  operations.push_back(std::move(linear));
  operations.push_back(std::move(add));
  auto model = Model{"uses", std::move(inputs), std::move(parameters), std::move(operations),
                     std::vector<const Value *>{&add_pointer->GetResult(0)}};

  ASSERT_NO_THROW(model.Verify());
  ASSERT_EQ(linear_pointer->GetResult(0).GetUsers().size(), 2U);
  EXPECT_EQ(linear_pointer->GetResult(0).GetUsers()[0].user_, add_pointer);
  EXPECT_EQ(linear_pointer->GetResult(0).GetUsers()[0].operand_index_, 0U);
  EXPECT_EQ(linear_pointer->GetResult(0).GetUsers()[1].operand_index_, 1U);
}

}  // namespace
}  // namespace zephyr::ir
