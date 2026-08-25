#include "planner/planner.h"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/exception.h"
#include "ir/operation/core.hpp"
#include "ir/operation/tensor.hpp"
#include "planner/instruction/core.h"
#include "planner/instruction/moe.h"
#include "planner/instruction/tensor.h"

namespace zephyr::planner {
namespace {

using InputViews = std::unordered_map<const ir::Input *, BufferView>;
using ResultViews = std::unordered_map<const ir::Operation *, std::vector<BufferView>>;
using ParameterViews = std::unordered_map<const ir::Parameter *, BufferView>;

class LoweringVisitor final : public ir::OperationVisitor {
 public:
  LoweringVisitor(WorkerPlan &plan, bool has_kv_cache, std::span<const DynamicDimensionBinding> bindings,
                  std::span<const ir::Value> outputs, InputViews &inputs)
      : plan_(plan), has_kv_cache_(has_kv_cache), bindings_(bindings), outputs_(outputs), inputs_(inputs) {}

  void Visit(const ir::Linear &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      auto biases = std::vector<std::optional<BufferView>>{};
      biases.emplace_back(
          operation.GetBias() == nullptr ? std::nullopt : std::optional<BufferView>{GetParameter(operation.GetBias())});
      plan_.execution_.instructions_.push_back(std::make_unique<planner::Linear>(
          GetValue(operation.GetInput()), std::vector<BufferView>{GetParameter(operation.GetWeight())},
          std::move(biases), results));
    });
  }

  void Visit(const ir::Embedding &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      plan_.execution_.instructions_.push_back(std::make_unique<planner::Embedding>(
          GetValue(operation.GetIndices()), GetParameter(operation.GetWeight()), results[0]));
    });
  }

  void Visit(const ir::RmsNorm &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      plan_.execution_.instructions_.push_back(std::make_unique<planner::RmsNorm>(
          GetValue(operation.GetInput()), GetParameter(operation.GetWeight()), results[0], operation.GetEpsilon()));
    });
  }

  void Visit(const ir::RotaryEmbedding &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      const auto operands = operation.GetOperands();
      plan_.execution_.instructions_.push_back(std::make_unique<planner::RotaryEmbedding>(
          GetValue(operands[0]), GetValue(operands[1]), GetValue(operands[2]), results[0], results[1],
          operation.GetTheta(), operation.GetRotaryDimension(), operation.GetLayout()));
    });
  }

  void Visit(const ir::SelfAttention &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      if (has_kv_cache_ && operation.GetMaskKind() != AttentionMaskKind::CAUSAL) {
        throw ConfigurationException{"decoder planning requires causal SelfAttention"};
      }
      const auto operands = operation.GetOperands();
      const auto &query_shape = ir::GetType(operands[0]).shape_;
      const auto &key_shape = ir::GetType(operands[1]).shape_;
      auto layer = std::optional<kv_layer_id_t>{};
      if (has_kv_cache_) {
        // Decoder attention consumes only the current step, so each causal layer receives one persistent KV entry.
        layer = static_cast<kv_layer_id_t>(plan_.kv_cache_.entries_.size());
        plan_.kv_cache_.entries_.push_back(
            KVCacheEntry{.kv_layer_id_ = *layer,
                         .dtype_ = ir::GetType(operands[0]).dtype_,
                         .kv_head_count_ = GetStaticExtent(key_shape[1], "KV heads"),
                         .head_dimension_ = GetStaticExtent(query_shape[2], "head dimension")});
      }
      plan_.execution_.instructions_.push_back(std::make_unique<planner::SelfAttention>(
          GetValue(operands[0]), GetValue(operands[1]), GetValue(operands[2]), results[0],
          GetStaticExtent(query_shape[1], "query heads"), GetStaticExtent(key_shape[1], "KV heads"),
          GetStaticExtent(query_shape[2], "head dimension"), operation.GetMaskKind(), operation.GetWindow(),
          operation.GetScale(), operation.GetSoftcap(), layer));
    });
  }

  void Visit(const ir::Moe &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      auto experts = std::vector<ExpertWeightViews>{};
      experts.reserve(operation.GetExperts().size());
      for (const auto &expert : operation.GetExperts()) {
        experts.push_back(ExpertWeightViews{.gate_ = GetParameter(expert.gate_weight_),
                                            .up_ = GetParameter(expert.up_weight_),
                                            .down_ = GetParameter(expert.down_weight_)});
      }
      const auto operands = operation.GetOperands();
      plan_.execution_.instructions_.push_back(std::make_unique<planner::Moe>(
          GetValue(operands[0]), GetValue(operands[1]),
          operation.GetSelectionBias() == nullptr
              ? std::nullopt
              : std::optional<BufferView>{GetParameter(operation.GetSelectionBias())},
          std::move(experts), results[0], operation.GetScoreFunction(), operation.GetTopK(),
          operation.GetWeightNormalization(), operation.GetRoutingScale(), operation.GetGroupRouting(),
          operation.GetActivation()));
    });
  }

  void Visit(const ir::Add &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      const auto operands = operation.GetOperands();
      plan_.execution_.instructions_.push_back(
          std::make_unique<planner::Add>(GetValue(operands[0]), GetValue(operands[1]), results[0]));
    });
  }

  void Visit(const ir::Multiply &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      const auto operands = operation.GetOperands();
      plan_.execution_.instructions_.push_back(
          std::make_unique<planner::Multiply>(GetValue(operands[0]), GetValue(operands[1]), results[0]));
    });
  }

  void Visit(const ir::Silu &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      plan_.execution_.instructions_.push_back(
          std::make_unique<planner::Silu>(GetValue(operation.GetOperands()[0]), results[0]));
    });
  }

  void Visit(const ir::Sigmoid &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      plan_.execution_.instructions_.push_back(
          std::make_unique<planner::Sigmoid>(GetValue(operation.GetOperands()[0]), results[0]));
    });
  }

  void Visit(const ir::Reshape &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      const auto input = GetValue(operation.GetOperands()[0]);
      results[0].buffer_ = input.buffer_;
      results[0].element_offset_ = input.element_offset_;
      plan_.buffers_.pop_back();
    });
  }

  void Visit(const ir::Split &operation) override {
    Lower(operation, [this, &operation](auto &results) {
      plan_.execution_.instructions_.push_back(
          std::make_unique<planner::Split>(GetValue(operation.GetOperands()[0]), operation.GetDimension(), results));
    });
  }

  [[nodiscard]] auto GetValue(const ir::Value &value) -> BufferView {
    return std::visit(
        [this](const auto &source) -> BufferView {
          using Source = std::remove_cvref_t<decltype(source)>;
          if constexpr (std::same_as<Source, const ir::Input *>) {
            return inputs_.at(source);
          } else if constexpr (std::same_as<Source, const ir::Parameter *>) {
            return GetParameter(source);
          } else {
            return results_.at(source.operation_).at(source.result_index_);
          }
        },
        value);
  }

  auto GetParameter(const ir::Parameter *parameter) -> BufferView {
    if (const auto iterator = parameters_.find(parameter); iterator != parameters_.end()) {
      return iterator->second;
    }
    const auto view = Allocate(parameter->type_, BufferKind::WEIGHT);
    const auto slice = MakeFullSlice(parameter->type_.shape_, "parameter dimension");
    plan_.weights_.targets_.push_back(
        WeightTarget{.buffer_ = view.buffer_,
                     .sources_ = {WeightSourcePart{
                         .source_name_ = parameter->name_, .source_slice_ = slice, .target_slice_ = slice}}});
    parameters_.emplace(parameter, view);
    return view;
  }

 private:
  template <typename OperationType, typename Lowering>
  void Lower(const OperationType &operation, Lowering &&lowering) {
    auto results = std::vector<BufferView>{};
    results.reserve(operation.GetResultTypes().size());
    for (size_t index = 0; index < operation.GetResultTypes().size(); index++) {
      const auto result = ir::OpResult{.operation_ = &operation, .result_index_ = static_cast<uint32_t>(index)};
      const auto &type = operation.GetResultTypes()[index];
      results.push_back(Allocate(type, IsOutput(result) ? BufferKind::OUTPUT : BufferKind::ACTIVATION));
    }
    std::forward<Lowering>(lowering)(results);
    results_.emplace(&operation, std::move(results));
  }

  [[nodiscard]] auto IsOutput(const ir::OpResult &result) const -> bool {
    return std::ranges::find(outputs_, ir::Value{result}) != outputs_.end();
  }

  auto Allocate(const TensorType &type, BufferKind kind) -> BufferView {
    const auto id = static_cast<buffer_id_t>(plan_.buffers_.size());
    plan_.buffers_.push_back(
        BufferSpec{.id_ = id, .kind_ = kind, .dtype_ = type.dtype_, .capacity_ = ResolveShape(type.shape_, bindings_)});
    return BufferView{.buffer_ = id, .element_offset_ = 0, .shape_ = type.shape_};
  }

  WorkerPlan &plan_;
  bool has_kv_cache_;
  std::span<const DynamicDimensionBinding> bindings_;
  std::span<const ir::Value> outputs_;
  InputViews &inputs_;
  ParameterViews parameters_;
  ResultViews results_;
};

}  // namespace

auto Planner::Lower(const ir::Model &model, ttl::Device device, const PlanConfig &config,
                    std::span<const DynamicDimensionBinding> bindings) const -> WorkerPlan {
  const auto has_kv_cache = config.kv_cache_.block_size_ != 0;
  auto plan = WorkerPlan{
      .rank_ = 0,
      .device_ = device,
      .tensor_parallel_size_ = 1,
      .data_parallel_size_ = 1,
      .tensor_parallel_rank_ = 0,
      .data_parallel_rank_ = 0,
      .buffers_ = {},
      .execution_ = {},
      .weights_ = {},
      .kv_cache_ = KVCachePlan{.block_size_ = has_kv_cache ? config.kv_cache_.block_size_ : 0, .entries_ = {}},
      .inputs_ = {},
      .outputs_ = {}};

  auto inputs = InputViews{};

  for (size_t index = 0; index < model.GetInputs().size(); index++) {
    const auto *input = model.GetInputs()[index].get();
    const auto id = static_cast<buffer_id_t>(plan.buffers_.size());
    const auto view = BufferView{.buffer_ = id, .element_offset_ = 0, .shape_ = input->type_.shape_};
    plan.buffers_.push_back(BufferSpec{.id_ = id,
                                       .kind_ = BufferKind::INPUT,
                                       .dtype_ = input->type_.dtype_,
                                       .capacity_ = ResolveShape(input->type_.shape_, bindings)});
    inputs.emplace(input, view);
    plan.inputs_.push_back(InputBinding{.input_index_ = index, .target_ = view});
  }

  auto visitor = LoweringVisitor{plan, has_kv_cache, bindings, model.GetOutputs(), inputs};

  for (const auto &operation : model.GetOperations()) {
    operation->Accept(visitor);
  }

  for (size_t index = 0; index < model.GetOutputs().size(); index++) {
    const auto &output = model.GetOutputs()[index];
    plan.outputs_.push_back(OutputBinding{.output_index_ = index, .source_ = visitor.GetValue(output)});
  }
  return plan;
}

}  // namespace zephyr::planner
