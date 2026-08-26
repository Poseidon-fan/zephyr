#include <memory>
#include <utility>
#include <vector>

#include "common/tensor_type.hpp"
#include "gtest/gtest.h"
#include "ir/model.hpp"
#include "ir/operation/core.hpp"
#include "ir/operation/tensor.hpp"

namespace zephyr::ir {
namespace {

auto MakeInput(std::string name, TensorType type) -> std::unique_ptr<const Input> {
  return std::make_unique<const Input>(Input{.name_ = std::move(name), .type_ = std::move(type)});
}

auto MakeParameter(std::string name, TensorType type) -> std::unique_ptr<const Parameter> {
  return std::make_unique<const Parameter>(Parameter{.name_ = std::move(name), .type_ = std::move(type)});
}

TEST(IrValueTest, FormatsNamedValues) {
  const auto input = Input{.name_ = "tokens", .type_ = {.dtype_ = ttl::DType::INT64, .shape_ = {int64_t{4}}}};
  const auto parameter =
      Parameter{.name_ = "weight", .type_ = {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{4}, int64_t{4}}}};
  EXPECT_EQ(input.ToString(), "input %tokens : tensor<[4], i64>");
  EXPECT_EQ(parameter.ToString(), "parameter @weight : tensor<[4, 4], f32>");
}

TEST(IrOperationTest, InfersLinearOutputType) {
  const auto input =
      Input{.name_ = "hidden",
            .type_ = {.dtype_ = ttl::DType::BFLOAT16, .shape_ = {DynamicDimension{TOKEN_DIMENSION}, int64_t{4}}}};
  const auto weight =
      Parameter{.name_ = "weight", .type_ = {.dtype_ = ttl::DType::BFLOAT16, .shape_ = {int64_t{8}, int64_t{4}}}};
  const auto operation = Linear{Value{&input}, &weight};
  ASSERT_EQ(operation.GetResultTypes().size(), 1U);
  EXPECT_EQ(operation.GetResultTypes()[0],
            (TensorType{.dtype_ = ttl::DType::BFLOAT16, .shape_ = {DynamicDimension{TOKEN_DIMENSION}, int64_t{8}}}));
}

TEST(IrOperationTest, BroadcastsElementwiseOperands) {
  const auto lhs = Input{.name_ = "lhs", .type_ = {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{2}, int64_t{4}}}};
  const auto rhs = Input{.name_ = "rhs", .type_ = {.dtype_ = ttl::DType::FLOAT32, .shape_ = {int64_t{1}, int64_t{4}}}};
  const auto operation = Add{Value{&lhs}, Value{&rhs}};
  EXPECT_EQ(operation.GetResultTypes()[0], lhs.type_);
}

TEST(IrModelTest, PrintsCompleteModelThroughOperationDispatch) {
  auto input =
      MakeInput("hidden", {.dtype_ = ttl::DType::BFLOAT16, .shape_ = {DynamicDimension{TOKEN_DIMENSION}, int64_t{4}}});
  auto weight = MakeParameter("weight", {.dtype_ = ttl::DType::BFLOAT16, .shape_ = {int64_t{8}, int64_t{4}}});
  const auto *input_pointer = input.get();
  const auto *weight_pointer = weight.get();
  auto linear = std::make_unique<const Linear>(Value{input_pointer}, weight_pointer);
  const auto *linear_pointer = linear.get();

  auto inputs = std::vector<std::unique_ptr<const Input>>{};
  inputs.push_back(std::move(input));
  auto parameters = std::vector<std::unique_ptr<const Parameter>>{};
  parameters.push_back(std::move(weight));
  auto operations = std::vector<std::unique_ptr<const Operation>>{};
  operations.push_back(std::move(linear));
  auto model = Model{"example", std::move(inputs), std::move(parameters), std::move(operations),
                     std::vector<Value>{OpResult{.operation_ = linear_pointer, .result_index_ = 0}}};

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

}  // namespace
}  // namespace zephyr::ir
