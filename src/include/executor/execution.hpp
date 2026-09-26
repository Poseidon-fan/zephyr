#pragma once

#include <cstddef>
#include <cstdint>
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

struct ExecutorOptions;

/** Hard batch bounds shared by initialization, input preparation, and scheduling. */
struct ExecutionLimits final {
  size_t max_num_seqs_{1};
  /** Total physical input tokens, including padding; zero resolves to max_num_seqs_ times the context limit. */
  size_t max_num_batched_tokens_{512};
  /** Total selected vocabulary-logit rows across all sequences in one batch. */
  size_t max_num_output_tokens_{1};
  /** Zero inherits the model's context limit. */
  int64_t max_seq_len_{0};

  /** Resolve inherited bounds and validate the complete batch against INT32 indexing. */
  [[nodiscard]] auto Resolve(int64_t model_max_seq_len) const -> ExecutionLimits;
  [[nodiscard]] auto operator==(const ExecutionLimits &) const noexcept -> bool = default;
};

/** Borrowed logical inputs. Preparation checks their category before publishing device work. */
struct ExecutionBatch {
  virtual ~ExecutionBatch() = default;
};

/** Immutable host inputs owned by the executor and borrowed by every tensor-parallel rank. */
struct ExecutionPlan {
  virtual ~ExecutionPlan() = default;
};

/** Common placement and resolved limits; derived specifications describe a category's capabilities. */
struct ExecutionSpec {
  ExecutionSpec(ttl::Device device, ttl::DType dtype, ExecutionLimits limits)
      : device_(device), dtype_(dtype), limits_(limits) {}
  virtual ~ExecutionSpec() = default;

  ttl::Device device_;
  ttl::DType dtype_;
  ExecutionLimits limits_;
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

  /**
   * Initialize rank resources after every model has loaded and synchronized. Every rank must submit the same
   * collective sequence. Return an independent host processor describing the finalized execution capabilities.
   */
  [[nodiscard]] virtual auto Initialize(ttl::ExecutionContext &context, const ExecutorOptions &options,
                                        const parallel::TpRankContext &rank) -> std::unique_ptr<InputProcessor> = 0;
  /** Every rank executes the same plan; the complete result must be available on rank zero. */
  [[nodiscard]] virtual auto Execute(ttl::ExecutionContext &context, const ExecutionPlan &plan) -> ExecutionResult = 0;
};

/**
 * Load one rank of the same model and execution category on every device. Concurrent calls must not mutate captured
 * state. Defer profiling and rank collectives until Initialize, after all model loads have completed.
 */
using ExecutionFactory =
    std::function<std::unique_ptr<Execution>(ttl::Runtime &, ttl::ExecutionContext &, const std::filesystem::path &,
                                             const weight::WeightBuilder &, const parallel::TpRankContext &)>;

}  // namespace zephyr::executor
