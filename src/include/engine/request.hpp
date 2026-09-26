#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "common/types.hpp"
#include "sampler/sampler.hpp"

namespace zephyr::engine {

/** Tokenized generation input. Sampling settings are effective values, with defaults already resolved. */
struct GenerationRequest final {
  std::vector<token_id_t> token_ids_;
  sampler::SamplingParams sampling_;
  /** Absent permits generation until EOS, an explicit stop token, or the model context limit. */
  std::optional<size_t> max_new_tokens_;
  std::vector<token_id_t> stop_token_ids_;
  std::optional<uint64_t> seed_;
  size_t num_choices_{1};
  bool ignore_eos_{false};
};

/** One complete token sequence; batching independent requests belongs to the scheduler. */
struct EmbeddingRequest final {
  std::vector<token_id_t> token_ids_;
};

using Request = std::variant<GenerationRequest, EmbeddingRequest>;

}  // namespace zephyr::engine
