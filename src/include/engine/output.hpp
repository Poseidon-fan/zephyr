#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/types.hpp"
#include "sampler/sampler.hpp"

namespace zephyr::engine {

enum class FinishReason : uint8_t { EOS, STOP_TOKEN, LENGTH, MODEL_LENGTH, CANCELED, ERROR };
enum class RequestStatus : uint8_t { COMPLETED, CANCELED, REJECTED, ERROR };

struct Usage final {
  size_t prompt_tokens_{0};
  size_t completion_tokens_{0};
};

/** Unread token increments for one choice. A terminating EOS/stop token is included in tokens_. */
struct ChoiceOutput final {
  size_t index_;
  std::vector<sampler::SamplingResult> tokens_;
  std::optional<FinishReason> finish_reason_;
};

/**
 * CPU-owned output, independent of the engine and GPU runtime. Generation increments can be coalesced;
 * an embedding result is a complete vector. status_ is present exactly once, on the final request output.
 * Final usage counts the prompt once and sampled tokens across all choices, including EOS/stop tokens.
 */
struct RequestOutput final {
  request_id_t request_id_;
  std::variant<std::vector<ChoiceOutput>, std::vector<float>> result_;
  std::optional<RequestStatus> status_;
  Usage usage_;
  std::string error_message_;
};

}  // namespace zephyr::engine
