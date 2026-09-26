#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <ttl/runtime/device_error.hpp>

#include "common/types.hpp"

namespace zephyr::sampler {

/** Per-row device parameters. Zero disables a probability filter; greedy rows use temperature one. */
struct SamplingRow final {
  double temperature_;
  int64_t top_k_;
  int64_t num_logprobs_;
  int64_t result_offset_;
  float frequency_penalty_;
  float presence_penalty_;
  float repetition_penalty_;
  float top_p_;
  float min_p_;
  float uniform_;
};

/** One deduplicated history/bias entry. No two entries write the same row and token. */
struct TokenAdjustment final {
  int64_t row_;
  int64_t generated_count_;
  token_id_t token_id_;
  float bias_;
  bool seen_;
};

static_assert(std::is_trivially_copyable_v<SamplingRow>);
static_assert(std::is_trivially_copyable_v<TokenAdjustment>);

void LaunchApplyAdjustments(cudaStream_t stream, float *logits, const SamplingRow *rows,
                            const TokenAdjustment *adjustments, int64_t count, int64_t vocabulary,
                            const ttl::CudaDeviceErrorContext &error_context);

/** Shift by the row maximum before scaling, avoiding overflow for finite logits and small temperatures. */
void LaunchPrepareSampling(cudaStream_t stream, float *logits, const float *maxima, const SamplingRow *rows,
                           int64_t batch_size, int64_t vocabulary, const ttl::CudaDeviceErrorContext &error_context);

/** cumulative is needed only for top-p; maxima only for min-p. Both arrays remain immutable. */
void LaunchFilterCandidates(cudaStream_t stream, float *weights, const float *cumulative, const float *maxima,
                            const SamplingRow *rows, int64_t batch_size, int64_t width,
                            const ttl::CudaDeviceErrorContext &error_context);

/** Select positive weights and gather precomputed log probabilities; null candidate_ids means vocabulary order. */
void LaunchSampleCdf(cudaStream_t stream, const float *cumulative, const float *weights, const int64_t *candidate_ids,
                     const float *log_probabilities, const int64_t *top_ids, const SamplingRow *rows,
                     token_id_t *output_ids, float *output_logprobs, int64_t batch_size, int64_t vocabulary,
                     int64_t width, int64_t report_width, const ttl::CudaDeviceErrorContext &error_context);

void LaunchSampleGreedy(cudaStream_t stream, const int64_t *selected_ids, const float *log_probabilities,
                        const int64_t *top_ids, const SamplingRow *rows, token_id_t *output_ids, float *output_logprobs,
                        int64_t batch_size, int64_t vocabulary, int64_t report_width,
                        const ttl::CudaDeviceErrorContext &error_context);

}  // namespace zephyr::sampler
