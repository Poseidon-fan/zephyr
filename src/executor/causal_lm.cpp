#include "executor/causal_lm.hpp"

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

#include "attention/sdpa.hpp"
#include "common/exception.hpp"
#include "kv_cache/cache_engine.hpp"
#include "layer/paged_attention.hpp"

namespace zephyr::executor {

CausalLMExecutionSpec::CausalLMExecutionSpec(const model::causal_lm::ModelSpec &model_spec,
                                             std::optional<kv_cache::CacheCapacity> cache_capacity)
    : ExecutionSpec(model_spec.device_, model_spec.dtype_),
      max_seq_len_(model_spec.max_seq_len_),
      vocab_size_(model_spec.vocab_size_),
      cache_capacity_(cache_capacity),
      supports_packed_prefill_(model_spec.supports_packed_prefill_) {}

namespace {

/** Groups have a common output length and deterministic order across all ranks. */
struct CausalLMPlan final : ExecutionPlan {
  enum class Kind : uint8_t { PROMPT, PACKED_PROMPT, PREFIX, DECODE };

  struct Group final {
    Kind kind_{Kind::PROMPT};
    int64_t batch_size_{0};
    int64_t query_length_{0};
    int64_t max_blocks_{0};
    int64_t max_context_len_{0};
    std::vector<size_t> row_indices_;
    std::vector<token_id_t> token_ids_;
    std::vector<int32_t> positions_;
    std::vector<int64_t> slots_;
    std::vector<int32_t> block_tables_;
    std::vector<int32_t> context_lens_;
    std::vector<int64_t> cached_tokens_;
    std::vector<int64_t> query_lens_;
    std::vector<int32_t> cumulative_query_lens_;
    std::vector<int32_t> cumulative_kv_lens_;
    std::vector<model::causal_lm::LogitsRange> logits_ranges_;
  };

  size_t num_inputs_{0};
  std::vector<Group> groups_;
};

class CausalLMInputProcessor final : public InputProcessor {
 public:
  explicit CausalLMInputProcessor(CausalLMExecutionSpec spec) : spec_(std::move(spec)) {}

  [[nodiscard]] auto GetSpec() const noexcept -> const CausalLMExecutionSpec & override { return spec_; }
  [[nodiscard]] auto IsCompatible(const InputProcessor &other) const noexcept -> bool override;
  [[nodiscard]] auto Prepare(const ExecutionBatch &inputs) const -> std::unique_ptr<const ExecutionPlan> override;

 private:
  const CausalLMExecutionSpec spec_;
};

/** Owns one rank's model and GPU cache until all submitted work has completed. */
class CausalLMExecution final : public Execution {
 public:
  CausalLMExecution(ttl::Runtime &runtime, std::unique_ptr<model::causal_lm::CausalLM> model,
                    const CausalLMOptions &options);

  [[nodiscard]] auto CreateInputProcessor() const -> std::unique_ptr<InputProcessor> override;
  [[nodiscard]] auto Execute(ttl::ExecutionContext &context, const ExecutionPlan &prepared) -> ExecutionResult override;

 private:
  ttl::Runtime &runtime_;
  std::unique_ptr<model::causal_lm::CausalLM> model_;
  std::unique_ptr<kv_cache::CacheEngine> cache_;
};

auto CausalLMInputProcessor::IsCompatible(const InputProcessor &other) const noexcept -> bool {
  const auto *processor = dynamic_cast<const CausalLMInputProcessor *>(&other);
  if (processor == nullptr) {
    return false;
  }
  const auto &spec = processor->spec_;
  return spec_.max_seq_len_ == spec.max_seq_len_ && spec_.vocab_size_ == spec.vocab_size_ &&
         spec_.cache_capacity_ == spec.cache_capacity_ &&
         spec_.supports_packed_prefill_ == spec.supports_packed_prefill_;
}

auto CausalLMInputProcessor::Prepare(const ExecutionBatch &inputs) const -> std::unique_ptr<const ExecutionPlan> {
  const auto *causal_batch = dynamic_cast<const CausalLMBatch *>(&inputs);
  if (causal_batch == nullptr) {
    throw InvalidArgumentException("causal execution requires a causal language model batch");
  }
  const auto &batch = *causal_batch;
  const auto &spec = spec_;
  const auto &capacity = spec.cache_capacity_;
  if (batch.inputs_.empty() || (batch.phase_ != CausalLMPhase::PREFILL && batch.phase_ != CausalLMPhase::DECODE)) {
    throw InvalidArgumentException("causal execution requires a nonempty prefill or decode batch");
  }
  if (!capacity.has_value() && batch.phase_ != CausalLMPhase::PREFILL) {
    throw InvalidArgumentException("decode requires an allocated KV cache");
  }

  constexpr auto max_index = std::numeric_limits<int32_t>::max();
  const auto max_context = std::min(spec.max_seq_len_, static_cast<int64_t>(max_index));
  auto owned_plan = std::make_unique<CausalLMPlan>();
  auto &plan = *owned_plan;
  plan.num_inputs_ = batch.inputs_.size();
  std::map<std::pair<int64_t, size_t>, size_t> groups;
  for (size_t row = 0; row < batch.inputs_.size(); ++row) {
    const auto &input = batch.inputs_[row];
    const auto cached = input.num_computed_tokens_;
    if (input.token_ids_.empty() || cached < 0 || cached > max_context ||
        std::cmp_greater(input.token_ids_.size(), max_context - cached)) {
      throw InvalidArgumentException("new tokens and cached prefix must fit the model context and INT32 indexing");
    }
    const auto query = static_cast<int64_t>(input.token_ids_.size());
    const auto &range = input.logits_range_;
    if (range.start_ < 0 || range.start_ > query || range.length_ < 0 || range.length_ > query - range.start_) {
      throw InvalidArgumentException("logits range must select positions within this call's new tokens");
    }
    if (std::ranges::any_of(input.token_ids_, [&](auto token) { return token < 0 || token >= spec.vocab_size_; })) {
      throw InvalidArgumentException("input token ID is outside the model vocabulary");
    }
    if (capacity.has_value()) {
      const auto context_length = static_cast<size_t>(cached + query);
      const auto pages =
          (context_length / capacity->block_size_) + static_cast<size_t>(context_length % capacity->block_size_ != 0);
      if (pages > input.block_ids_.size()) {
        throw InvalidArgumentException("KV pages must be allocated before submitting their tokens");
      }
      for (const auto page : std::span{input.block_ids_}.first(pages)) {
        if (page == 0 || page >= capacity->num_gpu_blocks_ || !std::in_range<int32_t>(page)) {
          throw InvalidArgumentException("active KV pages must have valid non-null INT32 IDs");
        }
      }
    } else if (cached != 0 || !input.block_ids_.empty()) {
      throw InvalidArgumentException("uncached prefill requires zero computed tokens and no page table");
    }

    // Preserve the first occurrence of each group; every rank must execute collective calls in this order.
    const auto key = std::pair{range.length_, batch.phase_ == CausalLMPhase::DECODE ? input.token_ids_.size() : 0};
    const auto [entry, inserted] = groups.try_emplace(key, plan.groups_.size());
    if (inserted) {
      plan.groups_.emplace_back();
    }
    plan.groups_[entry->second].row_indices_.push_back(row);
  }

  for (auto &group : plan.groups_) {
    int64_t max_query = 0;
    int64_t total_query = 0;
    bool has_prefix = false;
    bool selects_last = true;
    for (const auto row : group.row_indices_) {
      const auto &input = batch.inputs_[row];
      const auto query = static_cast<int64_t>(input.token_ids_.size());
      if (query > max_index - total_query) {
        throw InvalidArgumentException("batch token count exceeds INT32 indexing");
      }
      total_query += query;
      max_query = std::max(max_query, query);
      has_prefix = has_prefix || input.num_computed_tokens_ != 0;
      selects_last = selects_last && input.logits_range_.length_ == 1 && input.logits_range_.start_ == query - 1;
      group.max_context_len_ = std::max(group.max_context_len_, input.num_computed_tokens_ + query);
    }
    const auto rows = static_cast<int64_t>(group.row_indices_.size());
    const auto packed = batch.phase_ == CausalLMPhase::PREFILL && !has_prefix && batch.is_final_prompt_chunk_ &&
                        selects_last && spec.supports_packed_prefill_ && total_query != (rows * max_query);
    if (batch.phase_ == CausalLMPhase::DECODE) {
      group.kind_ = CausalLMPlan::Kind::DECODE;
    } else if (has_prefix) {
      group.kind_ = CausalLMPlan::Kind::PREFIX;
    } else if (packed) {
      group.kind_ = CausalLMPlan::Kind::PACKED_PROMPT;
    }
    group.batch_size_ = packed ? 1 : rows;
    group.query_length_ = packed ? total_query : max_query;
    if (group.batch_size_ > max_index / group.query_length_) {
      throw InvalidArgumentException("physical batch dimensions exceed INT32 indexing");
    }
    const auto physical_tokens = group.batch_size_ * group.query_length_;
    group.token_ids_.assign(static_cast<size_t>(physical_tokens), 0);
    group.positions_.assign(static_cast<size_t>(physical_tokens), 0);
    group.slots_.assign(static_cast<size_t>(physical_tokens), kv_cache::PADDING_SLOT_ID);

    const auto decode = group.kind_ == CausalLMPlan::Kind::DECODE;
    const auto prefix = group.kind_ == CausalLMPlan::Kind::PREFIX;
    if (decode || prefix) {
      const auto block_size = static_cast<int64_t>(capacity->block_size_);
      group.max_blocks_ =
          (group.max_context_len_ / block_size) + static_cast<int64_t>(group.max_context_len_ % block_size != 0);
      const auto table_rows = decode ? physical_tokens : rows;
      if (table_rows > max_index / group.max_blocks_) {
        throw InvalidArgumentException("block table dimensions exceed INT32 indexing");
      }
      group.block_tables_.assign(static_cast<size_t>(table_rows * group.max_blocks_), 0);
      group.context_lens_.reserve(static_cast<size_t>(table_rows));
    }
    if (packed) {
      group.cumulative_query_lens_.push_back(0);
    }
    if (prefix) {
      group.cumulative_kv_lens_.push_back(0);
    }

    size_t token_offset = 0;
    size_t table_row = 0;
    for (const auto row : group.row_indices_) {
      const auto &input = batch.inputs_[row];
      const auto query = static_cast<int64_t>(input.token_ids_.size());
      const auto cached = input.num_computed_tokens_;
      group.logits_ranges_.push_back(input.logits_range_);
      if (packed || prefix) {
        group.query_lens_.push_back(query);
      }
      if (packed) {
        group.cumulative_query_lens_.push_back(group.cumulative_query_lens_.back() + static_cast<int32_t>(query));
      }
      if (prefix) {
        group.cached_tokens_.push_back(cached);
        const auto previous = group.cumulative_kv_lens_.back();
        if (cached + query > max_index - previous) {
          throw InvalidArgumentException("gathered KV token count exceeds INT32 indexing");
        }
        group.cumulative_kv_lens_.push_back(previous + static_cast<int32_t>(cached + query));
      }

      std::ranges::copy(input.token_ids_, group.token_ids_.begin() + static_cast<std::ptrdiff_t>(token_offset));
      for (int64_t index = 0; index < query; ++index) {
        const auto position = cached + index;
        const auto token = token_offset + static_cast<size_t>(index);
        group.positions_[token] = static_cast<int32_t>(position);
        if (capacity.has_value()) {
          const auto logical_block = static_cast<size_t>(position) / capacity->block_size_;
          const auto offset = static_cast<size_t>(position) % capacity->block_size_;
          group.slots_[token] =
              static_cast<int64_t>((input.block_ids_[logical_block] * capacity->block_size_) + offset);
        }
      }
      token_offset += static_cast<size_t>(packed ? query : group.query_length_);

      if (decode || prefix) {
        const auto repeats = decode ? query : 1;
        for (int64_t index = 0; index < repeats; ++index) {
          // Each decode query sees only its causal prefix even though all new K/V rows are already written.
          const auto context_length = cached + (decode ? index + 1 : query);
          const auto block_size = static_cast<int64_t>(capacity->block_size_);
          const auto pages = (context_length / block_size) + static_cast<int64_t>(context_length % block_size != 0);
          const auto destination = group.block_tables_.begin() +
                                   static_cast<std::ptrdiff_t>(table_row * static_cast<size_t>(group.max_blocks_));
          std::ranges::transform(std::span{input.block_ids_}.first(static_cast<size_t>(pages)), destination,
                                 [](auto page) { return static_cast<int32_t>(page); });
          group.context_lens_.push_back(static_cast<int32_t>(context_length));
          ++table_row;
        }
      }
    }
  }
  return owned_plan;
}

CausalLMExecution::CausalLMExecution(ttl::Runtime &runtime, std::unique_ptr<model::causal_lm::CausalLM> model,
                                     const CausalLMOptions &options)
    : runtime_(runtime), model_(std::move(model)) {
  if (options.cache_capacity_.has_value()) {
    const auto &spec = model_->GetSpec();
    cache_ = std::make_unique<kv_cache::CacheEngine>(
        runtime_,
        kv_cache::CacheConfig{
            .capacity_ = *options.cache_capacity_, .dtype_ = spec.dtype_, .layer_specs_ = spec.layer_specs_},
        spec.device_);
  }
}

auto CausalLMExecution::CreateInputProcessor() const -> std::unique_ptr<InputProcessor> {
  return std::make_unique<CausalLMInputProcessor>(CausalLMExecutionSpec{
      model_->GetSpec(), cache_ != nullptr ? std::optional{cache_->GetConfig().capacity_} : std::nullopt});
}

auto CausalLMExecution::Execute(ttl::ExecutionContext &context, const ExecutionPlan &prepared) -> ExecutionResult {
  const auto &plan = dynamic_cast<const CausalLMPlan &>(prepared);
  std::vector<std::optional<ttl::Tensor>> rows(plan.num_inputs_);
  const auto upload = [&]<typename T>(const std::vector<T> &values, const ttl::Shape &shape) {
    auto tensor = ttl::Empty(context, shape, ttl::DTYPE_OF<T>);
    auto pinned = runtime_.AllocatePinned(values.size() * sizeof(T));
    std::ranges::copy(std::as_bytes(std::span{values}), pinned.AsBytes().begin());
    // TTL records stream use of the pinned allocation, so releasing this handle does not recycle it early.
    ttl::CopyFromPinnedAsync(context, tensor, pinned);
    return tensor;
  };
  for (const auto &group : plan.groups_) {
    const auto packed = group.kind_ == CausalLMPlan::Kind::PACKED_PROMPT;
    const auto prefix = group.kind_ == CausalLMPlan::Kind::PREFIX;
    const auto decode = group.kind_ == CausalLMPlan::Kind::DECODE;
    const auto tokens = group.batch_size_ * group.query_length_;
    const auto input_ids = upload(group.token_ids_, ttl::Shape{group.batch_size_, group.query_length_});
    const auto positions = upload(group.positions_, ttl::Shape{tokens});
    layer::PagedAttentionInputMetadata paged{.block_tables_ = std::nullopt,
                                             .context_lens_ = std::nullopt,
                                             .paged_context_lens_cpu_ = std::nullopt,
                                             .slot_mappings_ = upload(group.slots_, ttl::Shape{tokens}),
                                             .max_context_len_ = std::nullopt,
                                             .is_first_prompt_chunk_ = !prefix && !decode,
                                             .num_cached_tokens_ = std::nullopt,
                                             .query_lens_ = std::nullopt,
                                             .cu_seqlens_kv_ = std::nullopt};
    attention::FlashParams flash{};
    flash.causal_ = true;
    flash.packed_ = packed;
    if (packed || prefix) {
      paged.query_lens_ = group.query_lens_;
    }
    if (packed) {
      flash.max_q_ = static_cast<int32_t>(*std::ranges::max_element(group.query_lens_));
      flash.cumulative_seqlens_q_ =
          upload(group.cumulative_query_lens_, ttl::Shape{static_cast<int64_t>(group.cumulative_query_lens_.size())});
      flash.logical_k_ = {.max_ = flash.max_q_, .cumulative_seqlens_ = flash.cumulative_seqlens_q_};
    }
    if (prefix || decode) {
      const auto table_rows = static_cast<int64_t>(group.context_lens_.size());
      paged.block_tables_ = upload(group.block_tables_, ttl::Shape{table_rows, group.max_blocks_});
      paged.paged_context_lens_cpu_.emplace(group.context_lens_.begin(), group.context_lens_.end());
      if (prefix) {
        paged.num_cached_tokens_ = group.cached_tokens_;
        paged.cu_seqlens_kv_ =
            upload(group.cumulative_kv_lens_, ttl::Shape{static_cast<int64_t>(group.cumulative_kv_lens_.size())});
      } else {
        paged.context_lens_ = upload(group.context_lens_, ttl::Shape{table_rows});
        paged.max_context_len_ = group.max_context_len_;
      }
    }
    const model::causal_lm::ModelForwardContext forward_context{.positions_ = positions,
                                                                .paged_attention_ = paged,
                                                                .flash_params_ = flash,
                                                                .logits_ranges_ = group.logits_ranges_,
                                                                .cache_ = cache_.get()};
    const auto logits = model_->Forward(context, input_ids, forward_context);
    const auto length = group.logits_ranges_.front().length_;
    const auto selected = ttl::Reshape(
        context, logits,
        ttl::Shape{static_cast<int64_t>(group.row_indices_.size()) * length, logits.GetShape().GetDimension(2)});
    for (size_t index = 0; index < group.row_indices_.size(); ++index) {
      // Offsets count selected tokens, so all empty rows correctly share the zero-length storage view.
      rows[group.row_indices_[index]] = ttl::Narrow(selected, 0, static_cast<int64_t>(index) * length, length);
    }
  }
  CausalLMResult result;
  result.logits_.reserve(rows.size());
  for (auto &row : rows) {
    result.logits_.push_back(std::move(*row));
  }
  return result;
}

}  // namespace

auto CreateCausalLMFactory(CausalLMOptions options, model::ModelLoader<model::causal_lm::CausalLM> loader)
    -> ExecutionFactory {
  if (!loader) {
    throw ConfigurationException("causal execution requires a model loader");
  }
  return [options, loader = std::move(loader)](
             ttl::Runtime &runtime, ttl::ExecutionContext &context, const std::filesystem::path &config_path,
             const weight::WeightBuilder &builder, const parallel::TpRankContext &rank) -> std::unique_ptr<Execution> {
    auto model = loader(context, config_path, builder, rank);
    if (model == nullptr) {
      throw ConfigurationException("causal model loader returned no model");
    }
    return std::make_unique<CausalLMExecution>(runtime, std::move(model), options);
  };
}

}  // namespace zephyr::executor
