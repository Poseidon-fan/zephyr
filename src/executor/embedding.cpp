#include "executor/embedding.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <ttl/ops/copy.hpp>
#include <ttl/ops/creation.hpp>
#include <ttl/tensor/layout.hpp>
#include <ttl/tensor/shape.hpp>

#include "common/exception.hpp"

namespace zephyr::executor {

namespace {

/** Equal-length groups avoid padding changing attention or the model's pooling result. */
struct EmbeddingPlan final : ExecutionPlan {
  struct Group final {
    int64_t sequence_length_;
    std::vector<size_t> row_indices_;
    std::vector<token_id_t> token_ids_;
  };

  size_t num_inputs_{0};
  std::vector<Group> groups_;
};

class EmbeddingInputProcessor final : public InputProcessor {
 public:
  explicit EmbeddingInputProcessor(const model::embedding::ModelSpec &spec) : spec_(spec) {}

  [[nodiscard]] auto GetSpec() const noexcept -> const EmbeddingExecutionSpec & override { return spec_; }

  [[nodiscard]] auto IsCompatible(const InputProcessor &other) const noexcept -> bool override {
    const auto *processor = dynamic_cast<const EmbeddingInputProcessor *>(&other);
    return processor != nullptr && spec_.max_seq_len_ == processor->spec_.max_seq_len_ &&
           spec_.vocab_size_ == processor->spec_.vocab_size_ &&
           spec_.embedding_size_ == processor->spec_.embedding_size_ &&
           spec_.causal_attention_ == processor->spec_.causal_attention_;
  }

  [[nodiscard]] auto Prepare(const ExecutionBatch &batch) const -> std::unique_ptr<const ExecutionPlan> override {
    const auto *inputs = dynamic_cast<const EmbeddingBatch *>(&batch);
    if (inputs == nullptr || inputs->token_ids_.empty()) {
      throw InvalidArgumentException("embedding execution requires a nonempty embedding batch");
    }

    auto plan = std::make_unique<EmbeddingPlan>();
    plan->num_inputs_ = inputs->token_ids_.size();
    std::map<int64_t, size_t> groups;
    constexpr auto max_index = std::numeric_limits<int32_t>::max();
    const auto max_length = std::min(spec_.max_seq_len_, static_cast<int64_t>(max_index));
    for (size_t row = 0; row < inputs->token_ids_.size(); ++row) {
      const auto &tokens = inputs->token_ids_[row];
      if (tokens.empty() || std::cmp_greater(tokens.size(), max_length)) {
        throw InvalidArgumentException("embedding tokens must fit the model context and INT32 indexing");
      }
      if (std::ranges::any_of(tokens, [&](auto token) { return token < 0 || token >= spec_.vocab_size_; })) {
        throw InvalidArgumentException("input token ID is outside the model vocabulary");
      }

      const auto length = static_cast<int64_t>(tokens.size());
      const auto [entry, inserted] = groups.try_emplace(length, plan->groups_.size());
      if (inserted) {
        plan->groups_.push_back({.sequence_length_ = length, .row_indices_ = {}, .token_ids_ = {}});
      }
      auto &group = plan->groups_[entry->second];
      if (group.token_ids_.size() > static_cast<size_t>(max_index) - tokens.size()) {
        throw InvalidArgumentException("embedding batch dimensions exceed INT32 indexing");
      }
      // Preserve first-occurrence group order, so every rank submits collectives in the same order.
      group.row_indices_.push_back(row);
      group.token_ids_.insert(group.token_ids_.end(), tokens.begin(), tokens.end());
    }
    return plan;
  }

 private:
  const EmbeddingExecutionSpec spec_;
};

class EmbeddingExecution final : public Execution {
 public:
  EmbeddingExecution(ttl::Runtime &runtime, std::unique_ptr<model::embedding::Embedding> model)
      : runtime_(runtime), model_(std::move(model)) {
    const auto &spec = model_->GetSpec();
    if (spec.max_seq_len_ <= 0 || spec.vocab_size_ <= 0 || spec.embedding_size_ <= 0) {
      throw ConfigurationException("embedding model dimensions must be positive");
    }
  }

  [[nodiscard]] auto CreateInputProcessor() const -> std::unique_ptr<InputProcessor> override {
    return std::make_unique<EmbeddingInputProcessor>(model_->GetSpec());
  }

  [[nodiscard]] auto Execute(ttl::ExecutionContext &context, const ExecutionPlan &plan) -> ExecutionResult override {
    const auto &inputs = dynamic_cast<const EmbeddingPlan &>(plan);
    const auto &spec = model_->GetSpec();
    std::vector<std::optional<ttl::Tensor>> rows(inputs.num_inputs_);
    for (const auto &group : inputs.groups_) {
      const auto batch_size = static_cast<int64_t>(group.row_indices_.size());
      auto input_ids = ttl::Empty(context, ttl::Shape{batch_size, group.sequence_length_}, ttl::DTYPE_OF<token_id_t>);
      auto pinned = runtime_.AllocatePinned(group.token_ids_.size() * sizeof(token_id_t));
      std::ranges::copy(std::as_bytes(std::span{group.token_ids_}), pinned.AsBytes().begin());
      ttl::CopyFromPinnedAsync(context, input_ids, pinned);

      attention::FlashParams flash{};
      flash.max_q_ = static_cast<int32_t>(group.sequence_length_);
      flash.logical_k_.max_ = flash.max_q_;
      flash.causal_ = spec.causal_attention_;
      const auto embeddings = model_->Forward(context, input_ids, flash);
      if (embeddings.GetShape() != ttl::Shape{batch_size, spec.embedding_size_} ||
          embeddings.GetDevice() != spec.device_ || !ttl::IsFloating(embeddings.GetDType())) {
        throw InternalException("embedding model must return floating vectors of its declared shape and device");
      }
      for (size_t row = 0; row < group.row_indices_.size(); ++row) {
        rows[group.row_indices_[row]] = ttl::Select(embeddings, 0, static_cast<int64_t>(row));
      }
    }

    EmbeddingResult result;
    result.embeddings_.reserve(rows.size());
    for (auto &row : rows) {
      result.embeddings_.push_back(std::move(*row));
    }
    return result;
  }

 private:
  ttl::Runtime &runtime_;
  std::unique_ptr<model::embedding::Embedding> model_;
};

}  // namespace

auto CreateEmbeddingFactory(model::ModelLoader<model::embedding::Embedding> loader) -> ExecutionFactory {
  if (!loader) {
    throw ConfigurationException("embedding execution requires a model loader");
  }
  return [loader = std::move(loader)](ttl::Runtime &runtime, ttl::ExecutionContext &context,
                                      const std::filesystem::path &config_path, const weight::WeightBuilder &builder,
                                      const parallel::TpRankContext &rank) -> std::unique_ptr<Execution> {
    auto model = loader(context, config_path, builder, rank);
    if (model == nullptr) {
      throw ConfigurationException("embedding loader returned no model");
    }
    return std::make_unique<EmbeddingExecution>(runtime, std::move(model));
  };
}

}  // namespace zephyr::executor
