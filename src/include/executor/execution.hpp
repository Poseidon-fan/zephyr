#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <variant>
#include <vector>

#include <ttl/common/device.hpp>
#include <ttl/runtime/execution_context.hpp>
#include <ttl/runtime/runtime.hpp>
#include <ttl/tensor/tensor.hpp>

#include "parallel/tp.hpp"
#include "weight/weight_builder.hpp"

namespace zephyr::executor {

/** Borrowed logical inputs. Preparation checks their category before publishing device work. */
struct ExecutionBatch {
  virtual ~ExecutionBatch() = default;
};

/** Immutable host inputs owned by the executor and borrowed by every tensor-parallel rank. */
struct ExecutionPlan {
  virtual ~ExecutionPlan() = default;
};

/** Common placement information; derived specifications describe a category's capabilities. */
struct ExecutionSpec {
  ExecutionSpec(ttl::Device device, ttl::DType dtype) : device_(device), dtype_(dtype) {}
  virtual ~ExecutionSpec() = default;

  ttl::Device device_;
  ttl::DType dtype_;
};

struct CausalLMResult final {
  /** One [selected_tokens, vocabulary] tensor per input, in input order, on the output rank's device. */
  std::vector<ttl::Tensor> logits_;
};

struct EmbeddingResult final {
  /** One [embedding_size] tensor per input, in input order, on the output rank's device. */
  std::vector<ttl::Tensor> embeddings_;
};

// Results describe what the caller consumes, independently of the model or input modality that produced them.
using ExecutionResult = std::variant<CausalLMResult, EmbeddingResult>;

/** Owns host capabilities and prepares inputs without accessing a rank's model or device resources. */
class InputProcessor {
 public:
  virtual ~InputProcessor() = default;

  [[nodiscard]] virtual auto GetSpec() const noexcept -> const ExecutionSpec & = 0;
  /** Compare category and capabilities; device placement is checked separately by the executor. */
  [[nodiscard]] virtual auto IsCompatible(const InputProcessor &other) const noexcept -> bool = 0;
  [[nodiscard]] virtual auto Prepare(const ExecutionBatch &batch) const -> std::unique_ptr<const ExecutionPlan> = 0;
};

/** Owns one rank's model and device resources. Its factory pairs it with the matching input processor. */
class Execution {
 public:
  virtual ~Execution() = default;

  /** Return an independent host object whose lifetime is not tied to this execution or its model. */
  [[nodiscard]] virtual auto CreateInputProcessor() const -> std::unique_ptr<InputProcessor> = 0;
  /** Every rank executes the same plan; the complete result must be available on rank zero. */
  [[nodiscard]] virtual auto Execute(ttl::ExecutionContext &context, const ExecutionPlan &plan) -> ExecutionResult = 0;
};

/** Captures category configuration and constructs one rank. Concurrent calls must not mutate captured state. */
using ExecutionFactory =
    std::function<std::unique_ptr<Execution>(ttl::Runtime &, ttl::ExecutionContext &, const std::filesystem::path &,
                                             const weight::WeightBuilder &, const parallel::TpRankContext &)>;

}  // namespace zephyr::executor
