#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/runtime/runtime.hpp>
#include <ttl/tensor/tensor.hpp>

#include "common/types.hpp"

namespace zephyr::sampler {

/** Effective sampling settings after request and model defaults have been merged. */
struct SamplingParams final {
  /** Nonnegative; values below 1e-7 select greedily without advancing the random engine. */
  double temperature_{1.0};
  /** Nonpositive disables top-k; positive values are capped at the vocabulary size. */
  int64_t top_k_{-1};
  /** Probability filters are enabled only for values strictly between zero and one. */
  double top_p_{1.0};
  double min_p_{0.0};
  float frequency_penalty_{0.0F};
  float presence_penalty_{0.0F};
  float repetition_penalty_{1.0F};
  std::unordered_map<token_id_t, float> logits_bias_;
  /** Absent disables reporting; zero reports only the sampled token, N also reports the top N tokens. */
  std::optional<size_t> top_logprobs_;
};

/** Validate effective settings and bias token IDs before admitting a request or drawing random samples. */
void ValidateSamplingParams(const SamplingParams &params, int64_t vocab_size);

struct TokenLogprob final {
  token_id_t token_id_;
  float logprob_;
};

/** Natural-log probabilities after penalties and temperature, before candidate filtering. */
struct SamplingLogprobs final {
  float logprob_;
  std::vector<TokenLogprob> top_logprobs_;
};

struct SamplingResult final {
  token_id_t token_id_;
  std::optional<SamplingLogprobs> logprobs_;
};

/** Borrowed inputs for one next-token decision; history contains prompt followed by generated tokens. */
struct SamplingInput final {
  const ttl::Tensor &logits_;
  const SamplingParams &params_;
  std::span<const token_id_t> history_;
  size_t prompt_length_;
  std::mt19937_64 &rng_;
};

/**
 * Sample [vocabulary] rows on one device and return host results in input order.
 * Rows must share a floating dtype and vocabulary size; strided views are accepted. GPU producers must have
 * completed or established a dependency on context. Only results are read back; logits and history are unchanged.
 * Calls belong to one control thread. Each random row advances its supplied engine once, including top-k=1.
 * Shape and parameter checks precede random draws; execution failures do not roll the engines back.
 */
[[nodiscard]] auto Sample(ttl::Runtime &runtime, ttl::ExecutionContext &context, std::span<const SamplingInput> inputs)
    -> std::vector<SamplingResult>;

/**
 * Profile full-vocabulary sampling in a temporary independent context before allocating the KV cache.
 * Returns conservative additional GPU bytes, including input logits, context storage, cold scratch growth and
 * retained scratch. Every sampling mode is measured at max_rows; their transient peaks are added to cover mixed
 * groups without assuming that asynchronous frees complete between groups. CPU result materialization is skipped.
 * The caller must synchronize other work on this device and exclude concurrent allocations or peak-stat resets.
 */
[[nodiscard]] auto ProfileSamplingMemory(ttl::Runtime &runtime, ttl::Device device, size_t max_rows, int64_t vocab_size,
                                         ttl::DType logits_dtype) -> size_t;

}  // namespace zephyr::sampler
