#include "executor/causal_lm.hpp"

#include <algorithm>
#include <array>
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

#include "attention/paged_decode.hpp"
#include "attention/sdpa.hpp"
#include "common/exception.hpp"
#include "common/logger.hpp"
#include "executor/executor.hpp"
#include "kv_cache/cache_engine.hpp"
#include "layer/paged_attention.hpp"
#include "sampler/sampler.hpp"

namespace zephyr::executor {

CausalLMExecutionSpec::CausalLMExecutionSpec(const model::causal_lm::ModelSpec &model_spec,
                                             std::optional<kv_cache::CacheCapacity> cache_capacity,
                                             ExecutionLimits limits)
    : ExecutionSpec(model_spec.device_, model_spec.dtype_, limits),
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

  [[nodiscard]] auto Initialize(ttl::ExecutionContext &context, const ExecutorOptions &options,
                                const parallel::TpRankContext &rank) -> std::unique_ptr<InputProcessor> override;
  [[nodiscard]] auto Execute(ttl::ExecutionContext &context, const ExecutionPlan &prepared) -> ExecutionResult override;

 private:
  /** Execute representative maximum-shape paths using a small, initialized scratch cache. */
  void Profile(ttl::ExecutionContext &context, const ExecutionLimits &limits);

  ttl::Runtime &runtime_;
  std::unique_ptr<model::causal_lm::CausalLM> model_;
  CausalLMOptions options_;
  std::unique_ptr<kv_cache::CacheEngine> cache_;
};

auto CausalLMInputProcessor::IsCompatible(const InputProcessor &other) const noexcept -> bool {
  const auto *processor = dynamic_cast<const CausalLMInputProcessor *>(&other);
  if (processor == nullptr) {
    return false;
  }
  const auto &spec = processor->spec_;
  return spec_.vocab_size_ == spec.vocab_size_ && spec_.cache_capacity_ == spec.cache_capacity_ &&
         spec_.supports_packed_prefill_ == spec.supports_packed_prefill_;
}

auto CausalLMInputProcessor::Prepare(const ExecutionBatch &inputs) const -> std::unique_ptr<const ExecutionPlan> {
  const auto *causal_batch = dynamic_cast<const CausalLMBatch *>(&inputs);
  if (causal_batch == nullptr) {
    throw InvalidArgumentException("causal execution requires a causal language model batch");
  }
  const auto &batch = *causal_batch;
  const auto &spec = spec_;
  const auto &limits = spec.limits_;
  const auto &capacity = spec.cache_capacity_;
  if (batch.inputs_.empty() || (batch.phase_ != CausalLMPhase::PREFILL && batch.phase_ != CausalLMPhase::DECODE)) {
    throw InvalidArgumentException("causal execution requires a nonempty prefill or decode batch");
  }
  if (!capacity.has_value() && batch.phase_ != CausalLMPhase::PREFILL) {
    throw InvalidArgumentException("decode requires an allocated KV cache");
  }
  if (batch.inputs_.size() > limits.max_num_seqs_) {
    throw InvalidArgumentException("causal batch exceeds the initialized sequence limit");
  }

  constexpr auto max_index = std::numeric_limits<int32_t>::max();
  const auto max_context = limits.max_seq_len_;
  auto owned_plan = std::make_unique<CausalLMPlan>();
  auto &plan = *owned_plan;
  plan.num_inputs_ = batch.inputs_.size();
  std::map<std::pair<int64_t, size_t>, size_t> groups;
  size_t output_tokens = 0;
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
    if (static_cast<size_t>(range.length_) > limits.max_num_output_tokens_ - output_tokens) {
      throw InvalidArgumentException("selected logits exceed the initialized output token limit");
    }
    output_tokens += static_cast<size_t>(range.length_);
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

  size_t input_tokens = 0;
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
    // Resolved limits keep both factors within INT32, so their product fits INT64.
    const auto physical_tokens = group.batch_size_ * group.query_length_;
    if (static_cast<size_t>(physical_tokens) > limits.max_num_batched_tokens_ - input_tokens) {
      throw InvalidArgumentException("causal batch including padding exceeds the initialized token limit");
    }
    input_tokens += static_cast<size_t>(physical_tokens);
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
    : runtime_(runtime), model_(std::move(model)), options_(options) {}

void CausalLMExecution::Profile(ttl::ExecutionContext &context, const ExecutionLimits &limits) {
  const auto &spec = model_->GetSpec();
  auto profile_limits = limits;
  profile_limits.max_num_seqs_ = std::min(limits.max_num_seqs_, limits.max_num_batched_tokens_);
  // Round rectangular probes up, so neither padding nor a short last row understates the token budget.
  profile_limits.max_num_batched_tokens_ = std::min(static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                                                    limits.max_num_batched_tokens_ + profile_limits.max_num_seqs_ - 1);
  std::optional<kv_cache::CacheCapacity> capacity;
  if (options_.kv_cache_.has_value()) {
    const auto block_size = options_.kv_cache_->block_size_;
    const auto pages = (profile_limits.max_num_batched_tokens_ / block_size) + (2 * profile_limits.max_num_seqs_) + 3;
    capacity = kv_cache::CacheCapacity{.block_size_ = block_size, .num_gpu_blocks_ = pages};
    cache_ = std::make_unique<kv_cache::CacheEngine>(
        runtime_,
        kv_cache::CacheConfig{.capacity_ = *capacity, .dtype_ = spec.dtype_, .layer_specs_ = spec.layer_specs_},
        spec.device_);
    for (size_t layer = 0; layer < cache_->GetNumLayers(); ++layer) {
      auto &storage = cache_->GetLayerCache(layer);
      ttl::FillOut(context, storage.key_cache_, ttl::Scalar{0.0F});
      ttl::FillOut(context, storage.value_cache_, ttl::Scalar{0.0F});
    }
  }
  context.Synchronize();
  runtime_.ResetPeakMemoryStatistics(spec.device_);
  // Attention-path probes must keep rows in one group even when the serving output budget is smaller.
  profile_limits.max_num_output_tokens_ = std::max(limits.max_num_output_tokens_, profile_limits.max_num_seqs_);
  const CausalLMInputProcessor processor{CausalLMExecutionSpec{spec, capacity, profile_limits}};
  // Earlier groups' outputs remain live while later groups execute; reserve that overlap on every rank.
  const auto retained_logits = ttl::Empty(
      context, ttl::Shape{static_cast<int64_t>(limits.max_num_output_tokens_), spec.vocab_size_}, spec.dtype_);
  const auto max_length = static_cast<size_t>(limits.max_seq_len_);
  enum class ProfileKind : uint8_t { PROMPT, PACKED_PROMPT, PREFIX, DECODE, LOGITS };
  const auto native_decode = capacity.has_value() && std::ranges::all_of(spec.layer_specs_, [&](const auto &layer) {
                               return layer.key_head_dim_ == layer.value_head_dim_ &&
                                      attention::SupportsPagedDecode(static_cast<int64_t>(layer.key_head_dim_),
                                                                     static_cast<int64_t>(capacity->block_size_));
                             });
  std::array batch_sizes{size_t{1}, std::min(size_t{2}, profile_limits.max_num_seqs_), profile_limits.max_num_seqs_};
  const auto end = std::ranges::unique(batch_sizes).begin();
  for (const auto rows : std::span{batch_sizes.begin(), end}) {
    for (const auto kind : {ProfileKind::PROMPT, ProfileKind::PACKED_PROMPT, ProfileKind::PREFIX, ProfileKind::DECODE,
                            ProfileKind::LOGITS}) {
      const auto prefix = kind == ProfileKind::PREFIX;
      const auto decode = kind == ProfileKind::DECODE;
      const auto packed = kind == ProfileKind::PACKED_PROMPT;
      if ((!capacity.has_value() && (prefix || decode)) || (prefix && max_length == 1) ||
          (packed && (rows < 2 || !spec.supports_packed_prefill_))) {
        continue;
      }
      // Native decode uses grid.y for query rows; a larger prefill budget does not enlarge that kernel domain.
      const auto token_budget = decode && native_decode ? std::min(limits.max_num_batched_tokens_, size_t{65535})
                                                        : limits.max_num_batched_tokens_;
      if (decode && rows > token_budget) {
        continue;
      }
      const auto query = std::min(max_length, decode ? token_budget / rows : (token_budget + rows - 1) / rows);
      if (packed && query < 2) {
        continue;
      }
      CausalLMBatch batch;
      batch.phase_ = decode ? CausalLMPhase::DECODE : CausalLMPhase::PREFILL;
      batch.is_final_prompt_chunk_ = true;
      size_t remaining_outputs = limits.max_num_output_tokens_;
      kv_cache::block_id_t next_page = 2;
      for (size_t row = 0; row < rows; ++row) {
        auto length = prefix ? std::min(query, max_length - 1) : query;
        if (packed && row == 0 && length < max_length) {
          ++length;
        }
        if ((packed || prefix) && row + 1 == rows && rows > 1 && length > 1) {
          --length;
        }
        auto context_length = prefix || decode ? max_length : length;
        // Every short context creates its own padding/concat buffers; keep one full context to fix their width.
        if (prefix && row > 0 && context_length > length) {
          --context_length;
        }
        const auto cached = context_length - length;
        const auto output_length = kind == ProfileKind::LOGITS
                                       ? std::min(length, (remaining_outputs + rows - row - 1) / (rows - row))
                                       : size_t{1};
        if (kind == ProfileKind::LOGITS) {
          remaining_outputs -= output_length;
        }
        CausalLMInput input{.token_ids_ = std::vector<token_id_t>(length, 0),
                            .num_computed_tokens_ = static_cast<int64_t>(cached),
                            .block_ids_ = {},
                            .logits_range_ = {.start_ = static_cast<int64_t>(length - output_length),
                                              .length_ = static_cast<int64_t>(output_length)}};
        if (kind == ProfileKind::LOGITS && row > 0 && output_length < length) {
          input.logits_range_.start_ = 0;
        }
        if (capacity.has_value()) {
          const auto block_size = capacity->block_size_;
          const auto pages = (context_length + block_size - 1) / block_size;
          // Read-only history may repeat the initialized zero page. Every written page is private to its row.
          input.block_ids_.assign(pages, 1);
          for (size_t page = cached / block_size; page < pages; ++page) {
            input.block_ids_[page] = next_page++;
          }
        }
        batch.inputs_.push_back(std::move(input));
      }
      {
        const auto plan = processor.Prepare(batch);
        static_cast<void>(Execute(context, *plan));
      }
      context.Synchronize();
    }
  }
}

auto CausalLMExecution::Initialize(ttl::ExecutionContext &context, const ExecutorOptions &options,
                                   const parallel::TpRankContext &rank) -> std::unique_ptr<InputProcessor> {
  const auto &spec = model_->GetSpec();
  auto limits = options.execution_limits_;
  if (!options_.kv_cache_.has_value()) {
    // Uncached execution reevaluates whole histories; a scheduler's new-token quantum is not its allocation bound.
    limits.max_num_batched_tokens_ = 0;
  }
  limits = limits.Resolve(spec.max_seq_len_);
  size_t block_bytes = 0;
  if (options_.kv_cache_.has_value()) {
    const auto &cache = *options_.kv_cache_;
    if ((cache.block_size_ != 8 && cache.block_size_ != 16 && cache.block_size_ != 32) || cache.memory_bytes_ == 0) {
      throw ConfigurationException("KV cache requires page size 8, 16, or 32 and a positive explicit byte budget");
    }
    block_bytes = kv_cache::GetCacheBlockBytes(spec.layer_specs_, spec.dtype_, cache.block_size_);
  }

  const auto statistics = [&] {
    for (const auto &entry : runtime_.GetStatistics().devices_) {
      if (entry.device_ == spec.device_) {
        return entry;
      }
    }
    throw InternalException("execution device is missing from runtime statistics");
  };
  runtime_.SynchronizeMemory(spec.device_);
  runtime_.TrimMemory(spec.device_, 0);
  Profile(context, limits);
  const auto peak_bytes = statistics().peak_physical_in_use_bytes_;
  const auto temporary_cache_bytes =
      cache_ != nullptr ? cache_->GetConfig().capacity_.num_gpu_blocks_ * block_bytes : 0;
  cache_.reset();
  context.Synchronize();
  runtime_.SynchronizeMemory(spec.device_);
  runtime_.TrimMemory(spec.device_, 0);
  const auto persistent_bytes = statistics().logical_live_bytes_;
  const auto model_peak = peak_bytes - temporary_cache_bytes;
  size_t transient_bytes = model_peak > persistent_bytes ? model_peak - persistent_bytes : 0;
  if (rank.Rank() == 0) {
    const auto sampling_bytes = sampler::ProfileSamplingMemory(runtime_, spec.device_, limits.max_num_output_tokens_,
                                                               spec.vocab_size_, spec.dtype_);
    if (sampling_bytes > std::numeric_limits<size_t>::max() - transient_bytes) {
      throw ConfigurationException("execution memory requirement exceeds the addressable range");
    }
    // Conservative addition also covers retained logits and independent worker/sampling contexts.
    transient_bytes += sampling_bytes;
  }
  context.Synchronize();
  runtime_.TrimMemory(spec.device_, 0);
  const auto memory = runtime_.GetDeviceMemoryInfo(spec.device_);
  // Retain headroom for native allocations, allocator granularity, and shape-dependent library workspaces.
  const auto headroom = std::max<uint64_t>(512ULL * 1024 * 1024, memory.free_bytes_ / 50);
  auto available = memory.free_bytes_ > headroom ? memory.free_bytes_ - headroom : 0;
  const auto used = memory.total_bytes_ - memory.free_bytes_;
  const auto target = static_cast<uint64_t>(static_cast<long double>(memory.total_bytes_) *
                                            static_cast<long double>(options.gpu_memory_utilization_));
  const auto explicit_bytes = options_.kv_cache_.has_value() ? options_.kv_cache_->memory_bytes_ : std::nullopt;
  if (!explicit_bytes.has_value()) {
    available = std::min(available, target > used ? target - used : 0);
  }
  if (memory.max_live_bytes_ != 0) {
    runtime_.ResetPeakMemoryStatistics(spec.device_);
    const auto charged = statistics().peak_physical_in_use_bytes_;
    available = std::min(available, memory.max_live_bytes_ > charged ? memory.max_live_bytes_ - charged : 0);
  }
  if (transient_bytes > available) {
    throw OutOfMemoryException("execution and sampling peaks exceed the GPU budget; reduce batch or context limits");
  }
  available -= transient_bytes;
  if (!options_.kv_cache_.has_value()) {
    return std::make_unique<CausalLMInputProcessor>(CausalLMExecutionSpec{spec, std::nullopt, limits});
  }
  if (explicit_bytes.has_value()) {
    if (*explicit_bytes > available) {
      throw OutOfMemoryException("explicit KV cache budget does not leave enough memory for execution");
    }
    available = *explicit_bytes;
  }
  auto num_blocks =
      static_cast<int64_t>(std::min<uint64_t>(available / block_bytes, std::numeric_limits<int32_t>::max()));
  auto count = ttl::Empty(context, ttl::Shape{1}, ttl::DType::INT64);
  ttl::CopyFromHostBlocking(context, count, std::as_bytes(std::span{&num_blocks, size_t{1}}));
  rank.AllReduce(context, count, count, ttl::ReduceOp::MINIMUM);
  ttl::CopyToHostBlocking(context, std::as_writable_bytes(std::span{&num_blocks, size_t{1}}), count);
  const auto block_size = options_.kv_cache_->block_size_;
  const auto required_pages = (static_cast<size_t>(limits.max_seq_len_) + block_size - 1) / block_size;
  if (num_blocks <= 1 || std::cmp_less(num_blocks - 1, required_pages)) {
    throw OutOfMemoryException("KV cache cannot hold the configured context plus its reserved null page");
  }
  const kv_cache::CacheCapacity capacity{.block_size_ = block_size, .num_gpu_blocks_ = static_cast<size_t>(num_blocks)};
  cache_ = std::make_unique<kv_cache::CacheEngine>(
      runtime_, kv_cache::CacheConfig{.capacity_ = capacity, .dtype_ = spec.dtype_, .layer_specs_ = spec.layer_specs_},
      spec.device_);
  ZEPHYR_LOG_INFO("rank {}: KV cache {} pages x {} tokens, {} MiB; execution reserve {} MiB", rank.Rank(),
                  capacity.num_gpu_blocks_, capacity.block_size_,
                  (capacity.num_gpu_blocks_ * block_bytes) / (1024 * 1024), transient_bytes / (1024 * 1024));
  return std::make_unique<CausalLMInputProcessor>(CausalLMExecutionSpec{spec, capacity, limits});
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
