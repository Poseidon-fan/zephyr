#include "scheduler/paged_scheduler.hpp"

#include <algorithm>
#include <limits>
#include <ranges>
#include <utility>

#include "common/exception.hpp"
#include "common/macros.hpp"
#include "kv_cache/block_hash.hpp"

namespace zephyr::scheduler {

PagedScheduler::PagedScheduler(PagedSchedulerConfig config, SequenceTable &sequences,
                               kv_cache::KVCacheManager &cache_manager, bool supports_packed_prefill)
    : config_(config),
      sequences_(sequences),
      cache_manager_(cache_manager),
      supports_packed_prefill_(supports_packed_prefill) {
  if (config.max_num_seqs_ == 0 || config.max_num_batched_tokens_ == 0 || config.max_prefill_chunk_tokens_ == 0 ||
      config.max_decode_steps_before_prefill_ == 0) {
    throw ConfigurationException("Scheduler sequence and token budgets must be positive");
  }
}

void PagedScheduler::Add(sequence_id_t sequence_id) {
  const auto &sequence = sequences_.at(sequence_id);
  if (sequence.id_ != sequence_id || sequence.state_ != SequenceState::WAITING || sequence.num_computed_tokens_ != 0 ||
      sequence.step_type_ != SequenceStepType::PROMPT_AND_DECODE || sequence.token_ids_.empty() ||
      entries_.contains(sequence_id)) {
    throw InvalidArgumentException("Paged scheduling requires a fresh, nonempty generation sequence with a unique ID");
  }
  entries_.emplace(sequence_id, SequenceEntry{.arrival_order_ = next_arrival_order_++, .block_hashes_ = {}});
  waiting_.push_back(sequence_id);
}

auto PagedScheduler::GetPrefillBudget() const -> size_t {
  const auto has_decode =
      std::ranges::any_of(running_, [&](auto id) { return sequences_.at(id).state_ == SequenceState::DECODE; });
  return has_decode && !prompt_admission_epoch_
             ? std::min(config_.max_num_batched_tokens_, config_.max_prefill_chunk_tokens_)
             : config_.max_num_batched_tokens_;
}

auto PagedScheduler::IsDecodeDue() const -> bool {
  const auto budget = GetPrefillBudget();
  size_t prompt_tokens = 0;
  size_t decode_count = 0;
  bool prompts_fit = true;
  for (const auto id : running_) {
    const auto &sequence = sequences_.at(id);
    if (sequence.state_ == SequenceState::DECODE) {
      ++decode_count;
    } else {
      const auto remaining = sequence.token_ids_.size() - sequence.num_computed_tokens_;
      if (remaining > budget - prompt_tokens) {
        prompts_fit = false;
      } else {
        prompt_tokens += remaining;
      }
    }
  }
  if (decode_count == 0) {
    return false;
  }
  const auto has_prompt = decode_count != running_.size();
  if (has_prompt && prompts_fit && decode_steps_since_prefill_ > 0) {
    return false;
  }
  if (!has_prompt && !waiting_.empty() && running_.size() < config_.max_num_seqs_ && decode_steps_since_prefill_ > 0) {
    const auto tokens = sequences_.at(waiting_.front()).token_ids_.size();
    const auto block_size = cache_manager_.GetBlockSize();
    const auto blocks = (tokens / block_size) + static_cast<size_t>(tokens % block_size != 0);
    const auto available = cache_manager_.GetNumUnreservedBlocks();
    // Give an admissible prompt a turn while leaving one page of headroom per decoder.
    if (decode_count <= available && blocks <= available - decode_count) {
      return false;
    }
  }
  const auto max_decode_steps = prompt_admission_epoch_ ? size_t{1} : config_.max_decode_steps_before_prefill_;
  return (!has_prompt && waiting_.empty()) || decode_steps_since_prefill_ < max_decode_steps;
}

void PagedScheduler::AdmitWaiting() {
  constexpr size_t waiting_timeout = 64;
  while (!waiting_.empty() && running_.size() < config_.max_num_seqs_) {
    const auto num_prompts =
        std::ranges::count_if(running_, [&](auto id) { return sequences_.at(id).state_ == SequenceState::PREFILL; });
    if (std::cmp_greater_equal(num_prompts, config_.max_num_batched_tokens_)) {
      break;
    }
    const auto id = waiting_.front();
    auto &sequence = sequences_.at(id);
    const auto block_size = cache_manager_.GetBlockSize();
    const auto num_tokens = sequence.token_ids_.size();
    const auto num_blocks = (num_tokens / block_size) + static_cast<size_t>(num_tokens % block_size != 0);
    if (num_blocks > cache_manager_.GetNumGpuBlocks() - 1) {
      RejectWaiting(SequenceState::REJECTED, "Sequence exceeds the total usable KV cache capacity");
      continue;
    }

    auto &entry = entries_.at(id);
    kv_cache::AppendBlockHashes(sequence.token_ids_, block_size, entry.block_hashes_);
    // Keep this admission's prefix fixed across retries. Preemption releases pages but does not reuse them.
    const auto computed = cache_manager_.GetComputedBlocks(entry.block_hashes_, num_tokens);
    if (!cache_manager_.ReservePrompt(id, num_tokens, computed.block_ids_)) {
      if (entry.waiting_count_ < std::numeric_limits<size_t>::max()) {
        ++entry.waiting_count_;
      }
      if (entry.waiting_count_ <= waiting_timeout) {
        break;
      }
      // Let reserved prompts finish: repeated eviction can discard their progress indefinitely.
      if (num_prompts != 0) {
        break;
      }

      bool allocated = false;
      while (!running_.empty()) {
        const auto victim = running_.back();
        running_.pop_back();
        waiting_.pop_front();
        Preempt(victim);
        waiting_.push_front(id);
        if (cache_manager_.ReservePrompt(id, num_tokens, computed.block_ids_)) {
          allocated = true;
          break;
        }
      }
      if (!allocated) {
        RejectWaiting(SequenceState::ERROR, "KV cache exhausted after preempting all running sequences");
        continue;
      }
    }
    entry.waiting_count_ = 0;
    sequence.prefix_cache_len_ = computed.num_computed_tokens_;
    sequence.num_computed_tokens_ = computed.num_computed_tokens_;
    sequence.state_ = SequenceState::PREFILL;
    waiting_.pop_front();
    running_.push_back(id);
  }
}

void PagedScheduler::RejectWaiting(SequenceState state, std::string reason) {
  const auto id = waiting_.front();
  auto &sequence = sequences_.at(id);
  sequence.state_ = state;
  sequence.error_message_ = std::move(reason);
  cache_manager_.Free(id);
  entries_.erase(id);
  waiting_.pop_front();
}

void PagedScheduler::CacheComputedBlocks(sequence_id_t sequence_id) {
  const auto &sequence = sequences_.at(sequence_id);
  auto &hashes = entries_.at(sequence_id).block_hashes_;
  kv_cache::AppendBlockHashes(sequence.token_ids_, cache_manager_.GetBlockSize(), hashes);
  cache_manager_.CacheBlocks(sequence_id, hashes, sequence.num_computed_tokens_);
}

void PagedScheduler::Preempt(sequence_id_t sequence_id) {
  auto &sequence = sequences_.at(sequence_id);
  CacheComputedBlocks(sequence_id);
  cache_manager_.Free(sequence_id);
  sequence.prefix_cache_len_ = 0;
  sequence.num_computed_tokens_ = 0;
  sequence.state_ = SequenceState::WAITING;
  waiting_.push_front(sequence_id);
}

auto PagedScheduler::SchedulePrefill() -> ScheduledBatch {
  ScheduledBatch batch;
  std::vector<sequence_id_t> candidates;
  for (const auto id : running_) {
    if (sequences_.at(id).state_ == SequenceState::PREFILL) {
      candidates.push_back(id);
    }
  }
  if (candidates.empty()) {
    next_prompt_sequence_id_.reset();
    return batch;
  }
  if (next_prompt_sequence_id_.has_value()) {
    const auto next = std::ranges::find(candidates, *next_prompt_sequence_id_);
    if (next != candidates.end()) {
      std::ranges::rotate(candidates, next);
    }
  }
  const auto budget = GetPrefillBudget();
  const auto num_candidates = std::min(candidates.size(), budget);
  auto chunk_size = budget / num_candidates;
  if (candidates.size() != running_.size()) {
    chunk_size = std::min(chunk_size, config_.max_prefill_chunk_tokens_);
  }
  const auto &first = sequences_.at(candidates.front());
  const auto first_remaining = first.token_ids_.size() - first.num_computed_tokens_;
  const auto first_query_len = std::min(first_remaining, chunk_size);
  batch.is_final_prompt_chunk_ = first_remaining <= chunk_size;
  next_prompt_sequence_id_.reset();
  for (size_t index = 0; index < candidates.size(); ++index) {
    const auto id = candidates[index];
    const auto &sequence = sequences_.at(id);
    const auto remaining = sequence.token_ids_.size() - sequence.num_computed_tokens_;
    const auto is_final = remaining <= chunk_size;
    const auto num_tokens = std::min(remaining, chunk_size);
    // Rotate omitted rows to the front next time; non-packed execution also groups equal query lengths.
    if (index >= num_candidates || is_final != batch.is_final_prompt_chunk_ ||
        (!supports_packed_prefill_ && num_tokens != first_query_len)) {
      if (!next_prompt_sequence_id_.has_value()) {
        next_prompt_sequence_id_ = id;
      }
      continue;
    }
    const auto end = sequence.num_computed_tokens_ + num_tokens;
    ZEPHYR_ENSURE(cache_manager_.AllocateSlots(id, end).has_value(), "Reserved prompt pages must be available");
    batch.sequences_.push_back(
        {.sequence_id_ = id, .start_token_ = sequence.num_computed_tokens_, .num_tokens_ = num_tokens});
  }
  return batch;
}

auto PagedScheduler::ScheduleDecode() -> ScheduledBatch {
  std::deque<sequence_id_t> prompts;
  for (const auto id : std::exchange(running_, {})) {
    if (sequences_.at(id).state_ == SequenceState::PREFILL) {
      prompts.push_back(id);
    } else {
      running_.push_back(id);
    }
  }
  std::ranges::sort(running_, {}, [&](auto id) { return entries_.at(id).arrival_order_; });

  // Allocate every resident decoder in FCFS order before selecting a token-budgeted batch.
  // Victims come only from the unprocessed decode tail; prompt reservations remain untouched.
  std::deque<sequence_id_t> allocated;
  while (!running_.empty()) {
    const auto id = running_.front();
    running_.pop_front();
    const auto &sequence = sequences_.at(id);
    const auto num_tokens = sequence.token_ids_.size();
    const auto target = num_tokens + static_cast<size_t>(sequence.num_computed_tokens_ == num_tokens);
    while (true) {
      if (cache_manager_.AllocateSlots(id, target).has_value()) {
        allocated.push_back(id);
        break;
      }
      if (running_.empty()) {
        Preempt(id);
        break;
      }
      const auto victim = running_.back();
      running_.pop_back();
      Preempt(victim);
    }
  }
  running_ = std::move(allocated);
  for (const auto id : running_) {
    CacheComputedBlocks(id);
  }
  if (decode_steps_since_prefill_ < config_.max_decode_steps_before_prefill_) {
    ++decode_steps_since_prefill_;
  }

  ScheduledBatch batch{.phase_ = BatchPhase::DECODE, .sequences_ = {}};
  if (!running_.empty()) {
    const auto start = decode_cursor_ % running_.size();
    auto remaining_tokens = config_.max_num_batched_tokens_;
    for (size_t offset = 0; offset < running_.size(); ++offset) {
      const auto index = (start + offset) % running_.size();
      const auto id = running_[index];
      const auto &sequence = sequences_.at(id);
      const auto cost = std::max(sequence.token_ids_.size() - sequence.num_computed_tokens_, size_t{1});
      if (cost <= remaining_tokens || batch.sequences_.empty()) {
        remaining_tokens -= std::min(cost, remaining_tokens);
        // Ordinary decoding evaluates the last token, including a replay when all tokens were computed.
        batch.sequences_.push_back(
            {.sequence_id_ = id, .start_token_ = sequence.token_ids_.size() - 1, .num_tokens_ = 1});
        decode_cursor_ = (index + 1) % running_.size();
      }
    }
  }
  running_.insert(running_.end(), prompts.begin(), prompts.end());
  return batch;
}

auto PagedScheduler::Schedule() -> SchedulePlan {
  // Retire completed work in queue order before admission can reuse its pages.
  std::vector<sequence_id_t> finished;
  for (const auto *queue : {&running_, &waiting_}) {
    for (const auto id : *queue) {
      if (sequences_.at(id).IsTerminal()) {
        finished.push_back(id);
      }
    }
  }
  for (const auto id : finished) {
    Remove(id);
  }

  SchedulePlan plan;
  for (const auto id : running_) {
    auto &sequence = sequences_.at(id);
    if (sequence.state_ == SequenceState::PREFILL && sequence.num_computed_tokens_ == sequence.token_ids_.size()) {
      sequence.state_ = SequenceState::DECODE;
    }
  }
  const auto has_prompt = !waiting_.empty() || std::ranges::any_of(running_, [&](auto id) {
    return sequences_.at(id).state_ == SequenceState::PREFILL;
  });
  const auto has_decode =
      std::ranges::any_of(running_, [&](auto id) { return sequences_.at(id).state_ == SequenceState::DECODE; });
  if (!has_prompt && decode_steps_since_prefill_ > 0) {
    prompt_admission_epoch_ = false;
  } else if (!has_decode) {
    prompt_admission_epoch_ = true;
  }
  if (!IsDecodeDue()) {
    AdmitWaiting();
    auto batch = SchedulePrefill();
    if (!batch.sequences_.empty()) {
      decode_steps_since_prefill_ = 0;
      plan.batches_.push_back(std::move(batch));
      return plan;
    }
  }
  auto batch = ScheduleDecode();
  if (!batch.sequences_.empty()) {
    plan.batches_.push_back(std::move(batch));
  }
  return plan;
}

void PagedScheduler::Remove(sequence_id_t sequence_id) {
  if (!entries_.contains(sequence_id)) {
    return;
  }
  auto &sequence = sequences_.at(sequence_id);
  if (sequence.state_ != SequenceState::ERROR && sequence.state_ != SequenceState::REJECTED) {
    if (sequence.num_computed_tokens_ >= cache_manager_.GetBlockSize()) {
      CacheComputedBlocks(sequence_id);
    }
    sequence.state_ = SequenceState::FINISHED;
  }
  cache_manager_.Free(sequence_id);
  std::erase(waiting_, sequence_id);
  std::erase(running_, sequence_id);
  entries_.erase(sequence_id);
}

}  // namespace zephyr::scheduler
