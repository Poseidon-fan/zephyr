#include "executor/embedding.hpp"

#include <algorithm>
#include <chrono>
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
#include "common/logger.hpp"
#include "executor/executor.hpp"

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
  EmbeddingInputProcessor(const model::embedding::ModelSpec &spec, ExecutionLimits limits) : spec_(spec, limits) {}

  [[nodiscard]] auto GetSpec() const noexcept -> const EmbeddingExecutionSpec & override { return spec_; }

  [[nodiscard]] auto IsCompatible(const InputProcessor &other) const noexcept -> bool override {
    const auto *processor = dynamic_cast<const EmbeddingInputProcessor *>(&other);
    return processor != nullptr && spec_.vocab_size_ == processor->spec_.vocab_size_ &&
           spec_.embedding_size_ == processor->spec_.embedding_size_ &&
           spec_.causal_attention_ == processor->spec_.causal_attention_;
  }

  [[nodiscard]] auto Prepare(const ExecutionBatch &batch) const -> std::unique_ptr<const ExecutionPlan> override {
    const auto *inputs = dynamic_cast<const EmbeddingBatch *>(&batch);
    if (inputs == nullptr || inputs->token_ids_.empty()) {
      throw InvalidArgumentException("embedding execution requires a nonempty embedding batch");
    }
    if (inputs->token_ids_.size() > spec_.limits_.max_num_seqs_) {
      throw InvalidArgumentException("embedding batch exceeds the execution sequence limit");
    }

    auto plan = std::make_unique<EmbeddingPlan>();
    plan->num_inputs_ = inputs->token_ids_.size();
    std::map<int64_t, size_t> groups;
    size_t total_tokens = 0;
    for (size_t row = 0; row < inputs->token_ids_.size(); ++row) {
      const auto &tokens = inputs->token_ids_[row];
      if (tokens.empty() || std::cmp_greater(tokens.size(), spec_.limits_.max_seq_len_)) {
        throw InvalidArgumentException("embedding tokens must fit the execution context limit");
      }
      if (tokens.size() > spec_.limits_.max_num_batched_tokens_ - total_tokens) {
        throw InvalidArgumentException("embedding batch exceeds the execution token limit");
      }
      total_tokens += tokens.size();
      if (std::ranges::any_of(tokens, [&](auto token) { return token < 0 || token >= spec_.vocab_size_; })) {
        throw InvalidArgumentException("input token ID is outside the model vocabulary");
      }

      const auto length = static_cast<int64_t>(tokens.size());
      const auto [entry, inserted] = groups.try_emplace(length, plan->groups_.size());
      if (inserted) {
        plan->groups_.push_back({.sequence_length_ = length, .row_indices_ = {}, .token_ids_ = {}});
      }
      auto &group = plan->groups_[entry->second];
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

  [[nodiscard]] auto Initialize(ttl::ExecutionContext &context, const ExecutorOptions &options,
                                const parallel::TpRankContext &rank) -> std::unique_ptr<InputProcessor> override {
    const auto &spec = model_->GetSpec();
    if (context.GetDevice() != spec.device_ || rank.Device() != spec.device_) {
      throw ConfigurationException("embedding initialization must use the model's tensor-parallel device");
    }
    auto requested_limits = options.execution_limits_;
    // Embedding batches consume full sequences; length-bucket scheduling never chunks an input.
    requested_limits.max_num_batched_tokens_ = 0;
    const auto limits = requested_limits.Resolve(spec.max_seq_len_);
    const auto profile_start = std::chrono::steady_clock::now();
    ZEPHYR_LOG_INFO(
        "rank {} on cuda:{}: profiling embedding memory (dtype={}, max_seq_len={}, max_num_seqs={}, "
        "max_batch_tokens={}, embedding_size={})",
        rank.Rank(), spec.device_.GetOrdinal(), ttl::GetDTypeInfo(spec.dtype_).name_, limits.max_seq_len_,
        limits.max_num_seqs_, limits.max_num_batched_tokens_, spec.embedding_size_);
    auto processor = std::make_unique<EmbeddingInputProcessor>(spec, limits);
    context.Synchronize();
    runtime_.SynchronizeMemory(spec.device_);
    runtime_.TrimMemory(spec.device_, 0);
    runtime_.ResetPeakMemoryStatistics(spec.device_);
    const auto warmup = [&](size_t rows) {
      const auto probe_start = std::chrono::steady_clock::now();
      ZEPHYR_LOG_DEBUG("rank {} on cuda:{}: embedding profile started (rows={}, sequence_length={})", rank.Rank(),
                       spec.device_.GetOrdinal(), rows, limits.max_seq_len_);
      EmbeddingBatch batch;
      batch.token_ids_.assign(rows, std::vector<token_id_t>(static_cast<size_t>(limits.max_seq_len_), 0));
      const auto plan = processor->Prepare(batch);
      static_cast<void>(Execute(context, *plan));
      // Release returned tensors before synchronization so their retirements are also observed.
      context.Synchronize();
      ZEPHYR_LOG_DEBUG("rank {} on cuda:{}: embedding profile completed in {:.3f}s", rank.Rank(),
                       spec.device_.GetOrdinal(),
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - probe_start).count());
    };
    warmup(1);
    if (limits.max_num_seqs_ > 1) {
      warmup(limits.max_num_seqs_);
    }

    runtime_.SynchronizeMemory(spec.device_);
    runtime_.TrimMemory(spec.device_, 0);
    const auto statistics = [&] {
      for (const auto &entry : runtime_.GetStatistics().devices_) {
        if (entry.device_ == spec.device_) {
          return entry;
        }
      }
      throw InternalException("embedding device is missing from runtime statistics");
    }();
    const auto persistent_bytes = statistics.logical_live_bytes_;
    auto transient_bytes = statistics.peak_physical_in_use_bytes_ > persistent_bytes
                               ? statistics.peak_physical_in_use_bytes_ - persistent_bytes
                               : uint64_t{0};
    if (rank.Rank() == 0) {
      // Engine readback may retain a float32 conversion alongside the model's output vectors.
      const auto output_elements = static_cast<uint64_t>(
          ttl::Shape{static_cast<int64_t>(limits.max_num_seqs_), spec.embedding_size_}.GetNumElements());
      if (output_elements > (std::numeric_limits<uint64_t>::max() - transient_bytes) / sizeof(float)) {
        throw ConfigurationException("embedding readback memory exceeds the addressable byte range");
      }
      transient_bytes += output_elements * sizeof(float);
    }
    const auto memory = runtime_.GetDeviceMemoryInfo(spec.device_);
    const auto used = memory.total_bytes_ - memory.free_bytes_;
    const auto target = static_cast<uint64_t>(static_cast<long double>(memory.total_bytes_) *
                                              static_cast<long double>(options.gpu_memory_utilization_));
    auto available = std::min(memory.free_bytes_, target > used ? target - used : uint64_t{0});
    if (memory.max_live_bytes_ != 0) {
      available =
          std::min(available,
                   memory.max_live_bytes_ > persistent_bytes ? memory.max_live_bytes_ - persistent_bytes : uint64_t{0});
    }
    ZEPHYR_LOG_INFO(
        "rank {} on cuda:{}: embedding memory profiled in {:.3f}s (persistent={} MiB, "
        "execution_reserve={} MiB, usable={} MiB)",
        rank.Rank(), spec.device_.GetOrdinal(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - profile_start).count(),
        persistent_bytes / (1024 * 1024), transient_bytes / (1024 * 1024), available / (1024 * 1024));
    if (used > target || transient_bytes > available) {
      throw OutOfMemoryException("embedding execution exceeds the GPU budget; reduce batch or context limits");
    }
    return processor;
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
