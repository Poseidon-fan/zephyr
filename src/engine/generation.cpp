#include "engine/engine.hpp"

#include <algorithm>
#include <span>
#include <utility>

#include <ttl/tensor/layout.hpp>

#include "common/macros.hpp"

namespace zephyr::engine {

void Engine::ExecuteGeneration(const scheduler::ScheduledBatch &batch) {
  executor::CausalLMBatch input;
  const bool has_cache = cache_manager_ != nullptr;
  const bool sample = batch.phase_ == scheduler::BatchPhase::DECODE || batch.is_final_prompt_chunk_;
  input.phase_ = has_cache && batch.phase_ == scheduler::BatchPhase::DECODE ? executor::CausalLMPhase::DECODE
                                                                            : executor::CausalLMPhase::PREFILL;
  input.is_final_prompt_chunk_ = !has_cache || batch.is_final_prompt_chunk_;
  input.inputs_.reserve(batch.sequences_.size());
  for (const auto &scheduled : batch.sequences_) {
    const auto &sequence = sequences_.at(scheduled.sequence_id_);
    const auto tokens = std::span{sequence.token_ids_}.subspan(scheduled.start_token_, scheduled.num_tokens_);
    std::vector<kv_cache::block_id_t> blocks;
    if (has_cache) {
      const auto table = cache_manager_->GetBlockTable(sequence.id_);
      blocks.assign(table.begin(), table.end());
    }
    input.inputs_.push_back(
        {.token_ids_ = {tokens.begin(), tokens.end()},
         .num_computed_tokens_ = has_cache ? static_cast<int64_t>(scheduled.start_token_) : 0,
         .block_ids_ = std::move(blocks),
         .logits_range_ = {.start_ = sample ? static_cast<int64_t>(tokens.size() - 1) : 0, .length_ = sample ? 1 : 0}});
  }
  auto result = ExecuteBatch(input, batch);
  if (!result.has_value()) {
    return;
  }
  const auto &logits = std::get<executor::CausalLMResult>(*result).logits_;
  ZEPHYR_ENSURE(logits.size() == batch.sequences_.size(), "Execution must preserve the scheduled row count");
  for (const auto &scheduled : batch.sequences_) {
    auto &sequence = sequences_.at(scheduled.sequence_id_);
    if (has_cache) {
      // Commit a completed range, not a token increment: a replay may evaluate an already cached token.
      sequence.num_computed_tokens_ =
          std::max(sequence.num_computed_tokens_, scheduled.start_token_ + scheduled.num_tokens_);
    }
  }
  if (!sample) {
    return;
  }

  std::vector<ttl::Tensor> rows;
  std::vector<sampler::SamplingInput> sampling_inputs;
  rows.reserve(logits.size());
  sampling_inputs.reserve(logits.size());
  for (size_t index = 0; index < logits.size(); ++index) {
    const auto id = batch.sequences_[index].sequence_id_;
    const auto &sequence = sequences_.at(id);
    auto &state = sequence_contexts_.at(id);
    const auto &request = std::get<GenerationRequest>(requests_.at(state.request_id_).request_);
    rows.push_back(ttl::Select(logits[index], 0, 0));
    sampling_inputs.push_back({.logits_ = rows.back(),
                               .params_ = request.sampling_,
                               .history_ = sequence.token_ids_,
                               .prompt_length_ = sequence.prompt_length_,
                               .rng_ = *state.rng_});
  }
  auto samples = sampler::Sample(runtime_, *context_, sampling_inputs);
  for (size_t index = 0; index < samples.size(); ++index) {
    const auto id = batch.sequences_[index].sequence_id_;
    auto &sequence = sequences_.at(id);
    auto &state = sequence_contexts_.at(id);
    auto &request = requests_.at(state.request_id_);
    const auto &generation = std::get<GenerationRequest>(request.request_);
    const auto token = samples[index].token_id_;
    std::optional<FinishReason> reason;
    if (!generation.ignore_eos_ && std::ranges::find(options_.eos_token_ids_, token) != options_.eos_token_ids_.end()) {
      reason = FinishReason::EOS;
    } else if (std::ranges::find(generation.stop_token_ids_, token) != generation.stop_token_ids_.end()) {
      reason = FinishReason::STOP_TOKEN;
    } else if (generation.max_new_tokens_.has_value() &&
               sequence.token_ids_.size() - sequence.prompt_length_ + 1 >= *generation.max_new_tokens_) {
      reason = FinishReason::LENGTH;
    } else if (std::cmp_greater_equal(sequence.token_ids_.size(), generation_spec_->max_seq_len_)) {
      reason = FinishReason::MODEL_LENGTH;
    }
    // The sampled token enters history now; its KV is produced only by a later forward call.
    sequence.token_ids_.push_back(token);
    sequence.state_ = reason.has_value() ? scheduler::SequenceState::FINISHED : scheduler::SequenceState::DECODE;
    auto &output = request.GetChoiceOutput(state.choice_index_);
    output.tokens_.push_back(std::move(samples[index]));
    output.finish_reason_ = reason;
    request.finish_reasons_[state.choice_index_] = reason;
    ++request.usage_.completion_tokens_;
  }
}

}  // namespace zephyr::engine
