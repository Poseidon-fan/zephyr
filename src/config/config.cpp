#include "config/config.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_set>

#include "common/exception.h"

namespace zephyr {
namespace {

/** Throws when a configuration value is not strictly positive. */
void ValidatePositive(int64_t value, std::string_view name) {
  if (value <= 0) {
    throw ConfigurationException{std::string{name} + " must be positive"};
  }
}

/** Validates TTL runtime inputs that can be checked without constructing a Runtime. */
void ValidateRuntimeConfig(const RuntimeConfig &config) {
  if (config.devices_.empty()) {
    throw ConfigurationException{"runtime.devices must not be empty"};
  }

  std::unordered_set<int32_t> ordinals;
  for (const auto device : config.devices_) {
    if (!ordinals.insert(device.GetOrdinal()).second) {
      throw ConfigurationException{"runtime.devices must not contain duplicates"};
    }
  }
}

/** Validates the rank topology against the configured CUDA devices. */
void ValidateParallelConfig(const ParallelConfig &config, size_t device_count) {
  ValidatePositive(config.tensor_parallel_size_, "plan.parallel.tensor_parallel_size");
  ValidatePositive(config.data_parallel_size_, "plan.parallel.data_parallel_size");
  const auto world_size =
      static_cast<int64_t>(config.tensor_parallel_size_) * static_cast<int64_t>(config.data_parallel_size_);
  if (world_size != static_cast<int64_t>(device_count)) {
    throw ConfigurationException{"runtime device count must equal tensor_parallel_size * data_parallel_size"};
  }
}

/** Validates capacities consumed by every runner mode. */
void ValidatePlanConfig(const PlanConfig &config, size_t device_count) {
  ValidateParallelConfig(config.parallel_, device_count);
  ValidatePositive(config.limits_.max_tokens_per_worker_, "plan.limits.max_tokens_per_worker");
  ValidatePositive(config.limits_.max_sequences_per_worker_, "plan.limits.max_sequences_per_worker");
  ValidatePositive(config.limits_.max_sequence_length_, "plan.limits.max_sequence_length");

  for (const auto &[name, capacity] : config.dynamic_dimension_capacities_) {
    if (name.empty()) {
      throw ConfigurationException{"plan.dynamic_dimension_capacities contains an empty name"};
    }
    if (capacity <= 0) {
      throw ConfigurationException{"dynamic dimension capacity must be positive: " + name};
    }
  }
}

/** Validates stateful decoder-only capacity and scheduling constraints. */
void ValidateDecoderConfig(const Config &config) {
  ValidatePositive(config.plan_.kv_cache_block_size_, "plan.kv_cache_block_size");
  if (config.plan_.limits_.kv_cache_bytes_per_worker_ == 0) {
    throw ConfigurationException{"plan.limits.kv_cache_bytes_per_worker must be positive in decoder mode"};
  }
  ValidatePositive(config.scheduler_.max_batch_tokens_, "scheduler.max_batch_tokens");
  ValidatePositive(config.scheduler_.max_batch_sequences_, "scheduler.max_batch_sequences");
  ValidatePositive(config.scheduler_.prefill_chunk_size_, "scheduler.prefill_chunk_size");

  const auto data_parallel_size = static_cast<int64_t>(config.plan_.parallel_.data_parallel_size_);
  const auto max_batch_tokens = data_parallel_size * static_cast<int64_t>(config.plan_.limits_.max_tokens_per_worker_);
  const auto max_batch_sequences =
      data_parallel_size * static_cast<int64_t>(config.plan_.limits_.max_sequences_per_worker_);
  if (config.scheduler_.max_batch_tokens_ > max_batch_tokens) {
    throw ConfigurationException{"scheduler.max_batch_tokens exceeds worker capacity"};
  }
  if (config.scheduler_.max_batch_sequences_ > max_batch_sequences) {
    throw ConfigurationException{"scheduler.max_batch_sequences exceeds worker capacity"};
  }
  if (config.scheduler_.prefill_chunk_size_ > config.scheduler_.max_batch_tokens_ ||
      config.scheduler_.prefill_chunk_size_ > config.plan_.limits_.max_tokens_per_worker_) {
    throw ConfigurationException{"scheduler.prefill_chunk_size exceeds token capacity"};
  }
}

/** Validates that embedding mode does not carry decoder scheduling state. */
void ValidateEmbeddingConfig(const Config &config) {
  if (config.scheduler_.max_batch_tokens_ != 0 || config.scheduler_.max_batch_sequences_ != 0 ||
      config.scheduler_.prefill_chunk_size_ != 0) {
    throw ConfigurationException{"scheduler settings must be zero in embedding mode"};
  }
}

}  // namespace

void Config::Validate() const {
  if (model_directory_.empty()) {
    throw ConfigurationException{"model_directory must not be empty"};
  }
  ValidateRuntimeConfig(runtime_);
  ValidatePlanConfig(plan_, runtime_.devices_.size());

  switch (runner_mode_) {
    case RunnerMode::DECODER:
      ValidateDecoderConfig(*this);
      return;
    case RunnerMode::EMBEDDING:
      ValidateEmbeddingConfig(*this);
      return;
  }
  throw ConfigurationException{"unknown runner mode"};
}

}  // namespace zephyr
