#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/types.hpp"

namespace zephyr::scheduler {

enum class SequenceState : uint8_t { WAITING, PREFILL, DECODE, FINISHED, ERROR, REJECTED };

enum class SequenceStepType : uint8_t { PROMPT_AND_DECODE, ONE_SHOT };

/**
 * Engine-owned, append-only token history. Only successful execution advances num_computed_tokens_;
 * it counts reusable KV, not tokens evaluated without a cache. The original prompt survives preemption.
 * The engine consumes a ONE_SHOT result once; PROMPT_AND_DECODE continues until a stopping condition.
 */
struct Sequence final {
  Sequence(sequence_id_t id, std::vector<token_id_t> token_ids,
           SequenceStepType step_type = SequenceStepType::PROMPT_AND_DECODE)
      : id_(id), token_ids_(std::move(token_ids)), prompt_length_(token_ids_.size()), step_type_(step_type) {}

  [[nodiscard]] auto IsRunning() const noexcept -> bool {
    return state_ == SequenceState::PREFILL || state_ == SequenceState::DECODE;
  }

  [[nodiscard]] auto IsTerminal() const noexcept -> bool {
    return state_ == SequenceState::FINISHED || state_ == SequenceState::ERROR || state_ == SequenceState::REJECTED;
  }

  sequence_id_t id_;
  std::vector<token_id_t> token_ids_;
  size_t prompt_length_;
  size_t num_computed_tokens_{0};
  /** Prefix reused at admission; distinct from progress made by subsequent execution. */
  size_t prefix_cache_len_{0};
  SequenceStepType step_type_;
  SequenceState state_{SequenceState::WAITING};
  std::string error_message_;
};

using SequenceTable = std::unordered_map<sequence_id_t, Sequence>;

}  // namespace zephyr::scheduler
