#include "sampler/sampler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ttl/ops/cast.hpp>
#include <ttl/ops/composition.hpp>
#include <ttl/ops/copy.hpp>
#include <ttl/ops/reduction.hpp>
#include <ttl/ops/scan.hpp>
#include <ttl/ops/softmax.hpp>
#include <ttl/ops/topk.hpp>
#include <ttl/runtime/kernel_launch.hpp>
#include <ttl/runtime/pinned_buffer.hpp>

#include "common/exception.hpp"
#include "sampler/kernels.cuh"

namespace zephyr::sampler {
namespace {

constexpr double GREEDY_TEMPERATURE = 1e-7;

// Full-vocabulary nucleus sampling is kept separate from bounded top-k and unsorted sampling.
enum class SamplingMode : uint8_t { GREEDY, UNSORTED, TOP_K, TOP_P };

struct PendingSamples final {
  std::vector<size_t> indices_;
  std::vector<SamplingRow> rows_;
  ttl::PinnedBuffer ids_;
  std::optional<ttl::PinnedBuffer> logprobs_;
};

/** Upload private, trivially copyable kernel records without imposing a tensor dtype on their fields. */
template <typename T>
auto UploadRecords(ttl::Runtime &runtime, ttl::ExecutionContext &context, std::span<const T> records) -> ttl::Tensor {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto bytes = std::as_bytes(records);
  if (!std::in_range<int64_t>(bytes.size())) {
    throw InvalidArgumentException("sampling parameters exceed the tensor size limit");
  }
  auto output = ttl::Empty(context, ttl::Shape{static_cast<int64_t>(bytes.size())}, ttl::DType::UINT8);
  auto staging = runtime.AllocatePinned(bytes.size());
  std::memcpy(staging.GetData(), bytes.data(), bytes.size());
  ttl::CopyFromPinnedAsync(context, output, staging);
  return output;
}

auto EnqueueGroup(ttl::Runtime &runtime, ttl::ExecutionContext &context, std::span<const SamplingInput> inputs,
                  std::span<const float> uniforms, std::vector<size_t> indices, SamplingMode mode) -> PendingSamples {
  const auto batch_size = static_cast<int64_t>(indices.size());
  const auto vocabulary = inputs[indices.front()].logits_.GetNumElements();
  const auto greedy = mode == SamplingMode::GREEDY;
  const auto ranked = mode == SamplingMode::TOP_K || mode == SamplingMode::TOP_P;
  std::vector<ttl::Tensor> input_rows;
  std::vector<SamplingRow> rows;
  std::vector<TokenAdjustment> adjustments;
  input_rows.reserve(indices.size());
  rows.reserve(indices.size());
  int64_t result_count = 0;
  int64_t report_width = 0;
  int64_t candidate_width = 0;
  bool reporting = false;
  bool has_top_p = false;
  bool has_min_p = false;

  for (const auto index : indices) {
    const auto &input = inputs[index];
    const auto &params = input.params_;
    const auto top_p = static_cast<float>(params.top_p_);
    const auto min_p = static_cast<float>(params.min_p_);
    const auto row_index = static_cast<int64_t>(rows.size());
    const auto top_k = params.top_k_ > 0 ? std::min(params.top_k_, vocabulary) : vocabulary;
    const auto num_logprobs =
        params.top_logprobs_.has_value()
            ? static_cast<int64_t>(std::min(*params.top_logprobs_, static_cast<size_t>(vocabulary)))
            : int64_t{-1};
    const auto row_result_count = 1 + std::max(num_logprobs, int64_t{0});
    if (result_count > std::numeric_limits<int64_t>::max() - row_result_count) {
      throw InvalidArgumentException("sampling results exceed the tensor size limit");
    }
    rows.push_back({.temperature_ = greedy ? 1.0 : params.temperature_,
                    .top_k_ = top_k,
                    .num_logprobs_ = num_logprobs,
                    .result_offset_ = result_count,
                    .frequency_penalty_ = params.frequency_penalty_,
                    .presence_penalty_ = params.presence_penalty_,
                    .repetition_penalty_ = params.repetition_penalty_,
                    .top_p_ = !greedy && top_p > 0.0F && top_p < 1.0F ? top_p : 0.0F,
                    .min_p_ = !greedy && min_p > 0.0F && min_p < 1.0F ? min_p : 0.0F,
                    .uniform_ = uniforms[index]});
    result_count += row_result_count;
    reporting = reporting || num_logprobs >= 0;
    report_width = std::max(report_width, num_logprobs);
    candidate_width = std::max(candidate_width, top_k);
    has_top_p = has_top_p || rows.back().top_p_ > 0.0F;
    has_min_p = has_min_p || rows.back().min_p_ > 0.0F;
    input_rows.push_back(input.logits_);

    // Merge history and bias before upload: one device thread updates each affected token exactly once.
    std::unordered_map<token_id_t, TokenAdjustment> token_updates;
    if (params.frequency_penalty_ != 0.0F || params.presence_penalty_ != 0.0F || params.repetition_penalty_ != 1.0F) {
      for (size_t position = 0; position < input.history_.size(); ++position) {
        const auto token = input.history_[position];
        // Non-vocabulary history entries do not describe a possible output token.
        if (token < 0 || token >= vocabulary) {
          continue;
        }
        auto &update = token_updates[token];
        update.row_ = row_index;
        update.token_id_ = token;
        update.seen_ = true;
        if (position >= input.prompt_length_) {
          ++update.generated_count_;
        }
      }
    }
    for (const auto &[token, bias] : params.logits_bias_) {
      auto &update = token_updates[token];
      update.row_ = row_index;
      update.token_id_ = token;
      update.bias_ = bias;
    }
    for (const auto &entry : token_updates) {
      adjustments.push_back(entry.second);
    }
  }

  auto logits = ttl::Stack(context, input_rows, 0);
  if (logits.GetDType() != ttl::DType::FLOAT32) {
    logits = ttl::Cast(context, logits, ttl::DType::FLOAT32);
  }
  const auto device_rows = UploadRecords(runtime, context, std::span<const SamplingRow>{rows});
  if (!adjustments.empty()) {
    const auto updates = UploadRecords(runtime, context, std::span<const TokenAdjustment>{adjustments});
    const std::array operands{device_rows, updates};
    const std::array outputs{&logits};
    ttl::SubmitCudaKernel(context, "SamplingAdjustments", operands, outputs, [&](ttl::CudaKernelLaunch &launch) {
      LaunchApplyAdjustments(launch.GetStream(), launch.GetOutputDataAs<float>(logits),
                             static_cast<const SamplingRow *>(launch.GetInputData(device_rows)),
                             static_cast<const TokenAdjustment *>(launch.GetInputData(updates)),
                             static_cast<int64_t>(adjustments.size()), vocabulary,
                             launch.GetDeviceErrorContext(ttl::DType::FLOAT32));
    });
  }

  const auto maxima = ttl::Maximum(context, logits, {.axes_ = {1}});
  {
    const std::array operands{device_rows, maxima};
    const std::array outputs{&logits};
    ttl::SubmitCudaKernel(context, "PrepareSampling", operands, outputs, [&](ttl::CudaKernelLaunch &launch) {
      LaunchPrepareSampling(launch.GetStream(), launch.GetOutputDataAs<float>(logits),
                            launch.GetInputDataAs<float>(maxima),
                            static_cast<const SamplingRow *>(launch.GetInputData(device_rows)), batch_size, vocabulary,
                            launch.GetDeviceErrorContext(ttl::DType::FLOAT32));
    });
  }

  std::optional<ttl::Tensor> probabilities;
  if (!greedy) {
    probabilities = ttl::Softmax(context, logits, {.axes_ = {1}});
  }
  std::optional<ttl::Tensor> log_probabilities;
  if (reporting) {
    // Compute reports directly from logits so small probabilities retain finite log values.
    log_probabilities = ttl::LogSoftmax(context, logits, {.axes_ = {1}});
  }
  std::optional<ttl::Tensor> candidate_ids;
  std::optional<ttl::Tensor> top_ids;
  std::optional<ttl::Tensor> cumulative;
  std::optional<ttl::Tensor> sampling_weights;
  std::optional<ttl::Tensor> selected_ids;
  if (greedy) {
    selected_ids = ttl::ArgMax(context, logits, 1);
  } else {
    auto weights = *probabilities;
    if (ranked) {
      auto candidates = ttl::TopK(context, *probabilities, {.axis_ = 1, .k_ = candidate_width});
      weights = std::move(candidates.first);
      candidate_ids = std::move(candidates.second);
    } else {
      candidate_width = vocabulary;
    }
    if (ranked || has_min_p) {
      std::optional<ttl::Tensor> prefix;
      std::optional<ttl::Tensor> probability_maxima;
      std::vector<ttl::Tensor> operands{device_rows};
      if (has_top_p) {
        prefix = ttl::CumulativeSum(context, weights, 1);
        operands.push_back(*prefix);
      }
      if (has_min_p) {
        probability_maxima = ttl::Maximum(context, *probabilities, {.axes_ = {1}});
        operands.push_back(*probability_maxima);
      }
      const std::array outputs{&weights};
      ttl::SubmitCudaKernel(context, "FilterSamplingCandidates", operands, outputs, [&](ttl::CudaKernelLaunch &launch) {
        LaunchFilterCandidates(
            launch.GetStream(), launch.GetOutputDataAs<float>(weights),
            prefix.has_value() ? launch.GetInputDataAs<float>(*prefix) : nullptr,
            probability_maxima.has_value() ? launch.GetInputDataAs<float>(*probability_maxima) : nullptr,
            static_cast<const SamplingRow *>(launch.GetInputData(device_rows)), batch_size, candidate_width,
            launch.GetDeviceErrorContext(ttl::DType::FLOAT32));
      });
    }
    cumulative = ttl::CumulativeSum(context, weights, 1);
    sampling_weights = std::move(weights);
  }

  if (report_width > 0) {
    // Sampling probabilities can underflow to equal zeros; rank the report in log space instead.
    top_ids = ttl::TopK(context, *log_probabilities, {.axis_ = 1, .k_ = report_width}).second;
  }
  auto output_ids = ttl::Empty(context, ttl::Shape{result_count}, ttl::DType::INT32);
  std::optional<ttl::Tensor> output_logprobs;
  std::vector<ttl::Tensor *> outputs{&output_ids};
  if (reporting) {
    output_logprobs = ttl::Empty(context, ttl::Shape{result_count}, ttl::DType::FLOAT32);
    outputs.push_back(&*output_logprobs);
  }
  std::vector<ttl::Tensor> operands{device_rows};
  for (const auto *tensor :
       {&log_probabilities, &candidate_ids, &top_ids, &cumulative, &sampling_weights, &selected_ids}) {
    if (tensor->has_value()) {
      operands.push_back(**tensor);
    }
  }
  ttl::SubmitCudaKernel(context, "SampleTokens", operands, outputs, [&](ttl::CudaKernelLaunch &launch) {
    const auto *device_parameters = static_cast<const SamplingRow *>(launch.GetInputData(device_rows));
    const auto *report_logprobs = reporting ? launch.GetInputDataAs<float>(*log_probabilities) : nullptr;
    const auto *report_ids = top_ids.has_value() ? launch.GetInputDataAs<int64_t>(*top_ids) : nullptr;
    auto *ids = launch.GetOutputDataAs<token_id_t>(output_ids);
    auto *logprobs = reporting ? launch.GetOutputDataAs<float>(*output_logprobs) : nullptr;
    const auto error_context = launch.GetDeviceErrorContext(ttl::DType::FLOAT32);
    if (greedy) {
      LaunchSampleGreedy(launch.GetStream(), launch.GetInputDataAs<int64_t>(*selected_ids), report_logprobs, report_ids,
                         device_parameters, ids, logprobs, batch_size, vocabulary, report_width, error_context);
    } else {
      LaunchSampleCdf(launch.GetStream(), launch.GetInputDataAs<float>(*cumulative),
                      launch.GetInputDataAs<float>(*sampling_weights),
                      candidate_ids.has_value() ? launch.GetInputDataAs<int64_t>(*candidate_ids) : nullptr,
                      report_logprobs, report_ids, device_parameters, ids, logprobs, batch_size, vocabulary,
                      candidate_width, report_width, error_context);
    }
  });

  auto pending = PendingSamples{
      .indices_ = std::move(indices),
      .rows_ = std::move(rows),
      .ids_ = runtime.AllocatePinned(static_cast<size_t>(output_ids.GetNumElements()) * sizeof(token_id_t)),
      .logprobs_ = std::nullopt};
  ttl::CopyToPinnedAsync(context, pending.ids_, output_ids);
  if (reporting) {
    pending.logprobs_ = runtime.AllocatePinned(static_cast<size_t>(output_logprobs->GetNumElements()) * sizeof(float));
    ttl::CopyToPinnedAsync(context, *pending.logprobs_, *output_logprobs);
  }
  return pending;
}

}  // namespace

auto Sample(ttl::Runtime &runtime, ttl::ExecutionContext &context, std::span<const SamplingInput> inputs)
    -> std::vector<SamplingResult> {
  if (inputs.empty()) {
    return {};
  }
  const auto vocabulary = inputs.front().logits_.GetNumElements();
  const auto dtype = inputs.front().logits_.GetDType();
  const auto device = context.GetDevice();
  if (vocabulary <= 0 || vocabulary > std::numeric_limits<token_id_t>::max()) {
    throw InvalidArgumentException("sampling requires a nonempty vocabulary representable by token_id_t");
  }
  std::array<std::vector<size_t>, 4> groups;
  for (size_t index = 0; index < inputs.size(); ++index) {
    const auto &input = inputs[index];
    const auto &params = input.params_;
    if (input.logits_.GetRank() != 1 || input.logits_.GetNumElements() != vocabulary ||
        input.logits_.GetDType() != dtype || !ttl::IsFloating(dtype) || input.logits_.GetDevice() != device) {
      throw InvalidArgumentException("sampling rows must share a floating dtype, vocabulary, and context device");
    }
    if (input.prompt_length_ > input.history_.size() || !std::in_range<int64_t>(input.history_.size())) {
      throw InvalidArgumentException("sampling history must include the complete prompt");
    }
    if (!std::isfinite(params.temperature_) || params.temperature_ < 0.0 || !std::isfinite(params.top_p_) ||
        !std::isfinite(params.min_p_) || !std::isfinite(params.frequency_penalty_) ||
        !std::isfinite(params.presence_penalty_) || !std::isfinite(params.repetition_penalty_) ||
        params.repetition_penalty_ <= 0.0F) {
      throw InvalidArgumentException(
          "sampling parameters must be finite, with nonnegative temperature and positive repetition penalty");
    }
    for (const auto &[token, bias] : params.logits_bias_) {
      if (token < 0 || token >= vocabulary || std::isnan(bias) || bias == std::numeric_limits<float>::infinity()) {
        throw InvalidArgumentException("logits bias requires a valid token and a finite or negative-infinite bias");
      }
    }
    auto mode = SamplingMode::UNSORTED;
    if (params.temperature_ < GREEDY_TEMPERATURE) {
      mode = SamplingMode::GREEDY;
    } else if (params.top_k_ > 0) {
      mode = SamplingMode::TOP_K;
    } else if (static_cast<float>(params.top_p_) > 0.0F && static_cast<float>(params.top_p_) < 1.0F) {
      mode = SamplingMode::TOP_P;
    }
    groups[static_cast<size_t>(mode)].push_back(index);
  }

  // Draw in caller order before grouping. A shared fallback engine therefore has a defined consumption order too.
  std::vector<float> uniforms(inputs.size(), 0.0F);
  for (size_t index = 0; index < inputs.size(); ++index) {
    if (inputs[index].params_.temperature_ >= GREEDY_TEMPERATURE) {
      uniforms[index] = static_cast<float>(inputs[index].rng_() >> 40U) * 0x1p-24F;
    }
  }
  std::vector<PendingSamples> pending;
  pending.reserve(groups.size());
  for (size_t group = 0; group < groups.size(); ++group) {
    if (!groups[group].empty()) {
      pending.push_back(
          EnqueueGroup(runtime, context, inputs, uniforms, std::move(groups[group]), static_cast<SamplingMode>(group)));
    }
  }
  // Pinned buffers and tensors retain their submitted uses; all groups share one final host wait.
  context.Synchronize();
  std::vector<SamplingResult> results(inputs.size());
  for (const auto &group : pending) {
    const auto *ids = static_cast<const token_id_t *>(group.ids_.GetData());
    const auto *logprobs =
        group.logprobs_.has_value() ? static_cast<const float *>(group.logprobs_->GetData()) : nullptr;
    for (size_t index = 0; index < group.indices_.size(); ++index) {
      const auto &row = group.rows_[index];
      auto &result = results[group.indices_[index]];
      result.token_id_ = ids[row.result_offset_];
      if (row.num_logprobs_ >= 0) {
        auto &report = result.logprobs_.emplace();
        report.logprob_ = logprobs[row.result_offset_];
        report.top_logprobs_.reserve(static_cast<size_t>(row.num_logprobs_));
        for (int64_t rank = 0; rank < row.num_logprobs_; ++rank) {
          const auto offset = row.result_offset_ + 1 + rank;
          report.top_logprobs_.push_back({.token_id_ = ids[offset], .logprob_ = logprobs[offset]});
        }
      }
    }
  }
  return results;
}

}  // namespace zephyr::sampler
