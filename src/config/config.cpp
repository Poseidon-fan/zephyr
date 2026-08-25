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
  if (static_cast<size_t>(world_size) != device_count) {
    throw ConfigurationException{"runtime device count must equal tensor_parallel_size * data_parallel_size"};
  }
}

/** Validates model and parallel configuration. */
void ValidatePlanConfig(const PlanConfig &config, size_t device_count) {
  ValidateParallelConfig(config.parallel_, device_count);
  ValidatePositive(config.model_.max_model_len_, "plan.model.max_model_len");
}

/** Validates scheduler limits shared by all runner modes. */
void ValidateSchedulerConfig(const SchedulerConfig &config) {
  ValidatePositive(config.max_num_batched_tokens_, "scheduler.max_num_batched_tokens");
  ValidatePositive(config.max_num_seqs_, "scheduler.max_num_seqs");
  if (config.max_num_batched_tokens_ < config.max_num_seqs_) {
    throw ConfigurationException{"scheduler.max_num_batched_tokens must be at least max_num_seqs"};
  }
}

/** Validates stateful decoder-only capacity and scheduling constraints. */
void ValidateDecoderConfig(const Config &config) {
  ValidatePositive(config.plan_.kv_cache_.block_size_, "plan.kv_cache.block_size");
  if (config.plan_.kv_cache_.kv_cache_memory_bytes_ == 0) {
    throw ConfigurationException{"plan.kv_cache.kv_cache_memory_bytes must be positive in decoder mode"};
  }
  ValidatePositive(config.scheduler_.prefill_chunk_size_, "scheduler.prefill_chunk_size");

  if (config.scheduler_.prefill_chunk_size_ > config.scheduler_.max_num_batched_tokens_) {
    throw ConfigurationException{"scheduler.prefill_chunk_size exceeds max_num_batched_tokens"};
  }
}

/** Validates that embedding mode does not carry decoder-only state. */
void ValidateEmbeddingConfig(const Config &config) {
  if (config.plan_.kv_cache_.block_size_ != 0 || config.plan_.kv_cache_.kv_cache_memory_bytes_ != 0) {
    throw ConfigurationException{"KV cache settings must be zero in embedding mode"};
  }
  if (config.scheduler_.prefill_chunk_size_ != 0) {
    throw ConfigurationException{"scheduler.prefill_chunk_size must be zero in embedding mode"};
  }
}

}  // namespace

void Config::Validate() const {
  if (model_directory_.empty()) {
    throw ConfigurationException{"model_directory must not be empty"};
  }
  ValidateRuntimeConfig(runtime_);
  ValidatePlanConfig(plan_, runtime_.devices_.size());
  ValidateSchedulerConfig(scheduler_);

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
