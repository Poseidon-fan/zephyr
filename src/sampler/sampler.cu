#include "sampler/kernels.cuh"

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <ttl/runtime/device_error.cuh>

namespace zephyr::sampler {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int64_t MAX_BLOCKS = 65535;

/** The first eligible CDF crossing and the last positive candidate for accumulated rounding at the tail. */
struct SampleCandidates final {
  int64_t first_;
  int64_t last_;
};

struct MergeSampleCandidates final {
  __device__ auto operator()(const SampleCandidates &left, const SampleCandidates &right) const -> SampleCandidates {
    return {.first_ = left.first_ < right.first_ ? left.first_ : right.first_,
            .last_ = left.last_ > right.last_ ? left.last_ : right.last_};
  }
};

__global__ void ApplyAdjustmentsKernel(float *logits, const SamplingRow *rows, const TokenAdjustment *adjustments,
                                       int64_t count, int64_t vocabulary, ttl::CudaDeviceErrorContext error_context) {
  if (error_context.record_ != nullptr &&
      error_context.record_->code_ != static_cast<uint32_t>(ttl::CudaDeviceErrorCode::NONE)) {
    return;
  }
  const auto stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (auto index = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x; index < count; index += stride) {
    const auto &adjustment = adjustments[index];
    const auto &row = rows[adjustment.row_];
    const auto offset = (adjustment.row_ * vocabulary) + adjustment.token_id_;
    auto score = logits[offset];
    // Each token appears once: generated counts and complete-history membership have distinct penalty semantics.
    if (adjustment.generated_count_ > 0) {
      score -= row.frequency_penalty_ * static_cast<float>(adjustment.generated_count_);
      score -= row.presence_penalty_;
    }
    if (adjustment.seen_ && row.repetition_penalty_ != 1.0F) {
      score = score > 0.0F ? score / row.repetition_penalty_ : score * row.repetition_penalty_;
    }
    score += adjustment.bias_;
    if (isnan(score) || (isinf(score) && score > 0.0F)) {
      ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INVALID_VALUE, offset,
                                 __float_as_uint(score));
    }
    logits[offset] = score;
  }
}

__global__ void PrepareSamplingKernel(float *logits, const float *maxima, const SamplingRow *rows, int64_t batch_size,
                                      int64_t vocabulary, ttl::CudaDeviceErrorContext error_context) {
  if (error_context.record_ != nullptr &&
      error_context.record_->code_ != static_cast<uint32_t>(ttl::CudaDeviceErrorCode::NONE)) {
    return;
  }
  const auto stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const auto elements = batch_size * vocabulary;
  for (auto index = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x; index < elements; index += stride) {
    const auto row_index = index / vocabulary;
    const auto maximum = maxima[row_index];
    if (!isfinite(maximum)) {
      if (index % vocabulary == 0) {
        ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INVALID_VALUE, index,
                                   __float_as_uint(maximum));
      }
      continue;
    }
    const auto score = logits[index];
    if (isnan(score) || (isinf(score) && score > 0.0F)) {
      ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INVALID_VALUE, index, __float_as_uint(score));
      continue;
    }
    // Subtract in double before scaling: finite FP32 extremes must not overflow to positive infinity.
    logits[index] =
        static_cast<float>((static_cast<double>(score) - static_cast<double>(maximum)) / rows[row_index].temperature_);
  }
}

__global__ void FilterCandidatesKernel(float *weights, const float *cumulative, const float *maxima,
                                       const SamplingRow *rows, int64_t batch_size, int64_t width,
                                       ttl::CudaDeviceErrorContext error_context) {
  if (error_context.record_ != nullptr &&
      error_context.record_->code_ != static_cast<uint32_t>(ttl::CudaDeviceErrorCode::NONE)) {
    return;
  }
  const auto stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const auto elements = batch_size * width;
  for (auto index = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x; index < elements; index += stride) {
    const auto row_index = index / width;
    const auto rank = index % width;
    const auto &row = rows[row_index];
    bool keep = rank < row.top_k_;
    if (keep && row.top_p_ > 0.0F) {
      // Test the preceding prefix so the candidate that reaches the threshold is retained.
      const auto prefix = rank == 0 ? 0.0F : cumulative[index - 1];
      const auto total = cumulative[(row_index * width) + (row.top_k_ - 1)];
      keep = prefix < (row.top_p_ * total);
    }
    if (keep && row.min_p_ > 0.0F) {
      keep = weights[index] > (row.min_p_ * maxima[row_index]);
    }
    if (!keep) {
      weights[index] = 0.0F;
    }
  }
}

template <bool GREEDY>
__global__ void SampleKernel(const float *weights, const float *cumulative, const int64_t *candidate_ids,
                             const int64_t *selected_ids, const float *log_probabilities, const int64_t *top_ids,
                             const SamplingRow *rows, token_id_t *output_ids, float *output_logprobs,
                             int64_t batch_size, int64_t vocabulary, int64_t width, int64_t report_width,
                             ttl::CudaDeviceErrorContext error_context) {
  __shared__ bool failed;
  if (threadIdx.x == 0) {
    failed = error_context.record_ != nullptr &&
             error_context.record_->code_ != static_cast<uint32_t>(ttl::CudaDeviceErrorCode::NONE);
  }
  // Other blocks may report a failure during this launch; all lanes must agree before entering a block collective.
  __syncthreads();
  if (failed) {
    return;
  }
  for (auto row_index = static_cast<int64_t>(blockIdx.x); row_index < batch_size; row_index += gridDim.x) {
    const auto &row = rows[row_index];
    int64_t token_id = 0;
    if constexpr (GREEDY) {
      if (threadIdx.x == 0) {
        token_id = selected_ids[row_index];
      }
    } else {
      const auto offset = row_index * width;
      const auto total = cumulative[offset + (width - 1)];
      if (!isfinite(total) || total <= 0.0F) {
        if (threadIdx.x == 0) {
          ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INVALID_VALUE, row_index,
                                     __float_as_uint(total));
        }
        continue;
      }
      // Parallel scans can round adjacent prefixes differently, even across zero weights. Select over positive
      // candidates explicitly instead of assuming an exactly monotone CDF or assigning mass to masked tokens.
      const auto target = fminf(row.uniform_ * total, nextafterf(total, 0.0F));
      auto candidates = SampleCandidates{.first_ = width, .last_ = -1};
      for (auto rank = static_cast<int64_t>(threadIdx.x); rank < width; rank += blockDim.x) {
        if (weights[offset + rank] > 0.0F) {
          candidates.last_ = rank;
          if (candidates.first_ == width && cumulative[offset + rank] > target) {
            candidates.first_ = rank;
          }
        }
      }
      using BlockReduce = cub::BlockReduce<SampleCandidates, THREADS_PER_BLOCK>;
      __shared__ typename BlockReduce::TempStorage reduction_storage;
      const auto selected = BlockReduce(reduction_storage).Reduce(candidates, MergeSampleCandidates{});
      if (threadIdx.x == 0) {
        const auto rank = selected.first_ < width ? selected.first_ : selected.last_;
        token_id = candidate_ids == nullptr ? rank : candidate_ids[offset + rank];
      }
      // Complete this row's shared reduction before any thread proceeds to the next row.
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      output_ids[row.result_offset_] = static_cast<token_id_t>(token_id);
    }
    if (output_logprobs != nullptr) {
      if (threadIdx.x == 0) {
        output_logprobs[row.result_offset_] =
            row.num_logprobs_ >= 0 ? log_probabilities[(row_index * vocabulary) + token_id] : 0.0F;
      }

      // Reporting uses the complete distribution, before any candidate filtering or renormalization.
      for (auto rank = static_cast<int64_t>(threadIdx.x); rank < row.num_logprobs_; rank += blockDim.x) {
        const auto token_id = top_ids[(row_index * report_width) + rank];
        const auto result_index = (row.result_offset_ + 1) + rank;
        output_ids[result_index] = static_cast<token_id_t>(token_id);
        output_logprobs[result_index] = log_probabilities[(row_index * vocabulary) + token_id];
      }
    }
  }
}

void LaunchApplyAdjustments(cudaStream_t stream, float *logits, const SamplingRow *rows,
                            const TokenAdjustment *adjustments, int64_t count, int64_t vocabulary,
                            const ttl::CudaDeviceErrorContext &error_context) {
  if (count == 0) {
    return;
  }
  const auto blocks = static_cast<unsigned int>(std::min(((count - 1) / THREADS_PER_BLOCK) + 1, MAX_BLOCKS));
  ApplyAdjustmentsKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(logits, rows, adjustments, count, vocabulary,
                                                                   error_context);
}

void LaunchPrepareSampling(cudaStream_t stream, float *logits, const float *maxima, const SamplingRow *rows,
                           int64_t batch_size, int64_t vocabulary, const ttl::CudaDeviceErrorContext &error_context) {
  const auto elements = batch_size * vocabulary;
  const auto blocks = static_cast<unsigned int>(std::min(((elements - 1) / THREADS_PER_BLOCK) + 1, MAX_BLOCKS));
  PrepareSamplingKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(logits, maxima, rows, batch_size, vocabulary,
                                                                  error_context);
}

void LaunchFilterCandidates(cudaStream_t stream, float *weights, const float *cumulative, const float *maxima,
                            const SamplingRow *rows, int64_t batch_size, int64_t width,
                            const ttl::CudaDeviceErrorContext &error_context) {
  const auto elements = batch_size * width;
  const auto blocks = static_cast<unsigned int>(std::min(((elements - 1) / THREADS_PER_BLOCK) + 1, MAX_BLOCKS));
  FilterCandidatesKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(weights, cumulative, maxima, rows, batch_size, width,
                                                                   error_context);
}

void LaunchSampleCdf(cudaStream_t stream, const float *cumulative, const float *weights, const int64_t *candidate_ids,
                     const float *log_probabilities, const int64_t *top_ids, const SamplingRow *rows,
                     token_id_t *output_ids, float *output_logprobs, int64_t batch_size, int64_t vocabulary,
                     int64_t width, int64_t report_width, const ttl::CudaDeviceErrorContext &error_context) {
  const auto blocks = static_cast<unsigned int>(std::min(batch_size, MAX_BLOCKS));
  SampleKernel<false><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      weights, cumulative, candidate_ids, nullptr, log_probabilities, top_ids, rows, output_ids, output_logprobs,
      batch_size, vocabulary, width, report_width, error_context);
}

void LaunchSampleGreedy(cudaStream_t stream, const int64_t *selected_ids, const float *log_probabilities,
                        const int64_t *top_ids, const SamplingRow *rows, token_id_t *output_ids, float *output_logprobs,
                        int64_t batch_size, int64_t vocabulary, int64_t report_width,
                        const ttl::CudaDeviceErrorContext &error_context) {
  const auto blocks = static_cast<unsigned int>(std::min(batch_size, MAX_BLOCKS));
  SampleKernel<true><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
      nullptr, nullptr, nullptr, selected_ids, log_probabilities, top_ids, rows, output_ids, output_logprobs,
      batch_size, vocabulary, 0, report_width, error_context);
}

}  // namespace zephyr::sampler
