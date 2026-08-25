#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <ttl/common/device.hpp>
#include <ttl/runtime/memory.hpp>

namespace zephyr {

/** Selects the task-level input and output contract exposed by an Engine. */
enum class RunnerMode : uint8_t {
  /** Autoregressive decoder inference with scheduling, KV cache, and sampling. */
  DECODER,

  /** Stateless embedding inference. */
  EMBEDDING,
};

/** TTL devices and memory policies used by the Engine runtime. */
struct RuntimeConfig final {
  /** CUDA devices in process-wide rank order. */
  std::vector<ttl::Device> devices_;

  /** Per-device TTL memory-pool policy. */
  ttl::DeviceMemoryOptions device_memory_;

  /** Process-wide TTL pinned-memory policy. */
  ttl::PinnedMemoryOptions pinned_memory_;
};

/** Model parallelism applied by the Planner. */
struct ParallelConfig final {
  /** Number of tensor-parallel ranks in each data-parallel replica. */
  int32_t tensor_parallel_size_{1};

  /** Number of independent data-parallel replicas. */
  int32_t data_parallel_size_{1};

  /** Whether routed experts are distributed across the global rank space. */
  bool enable_expert_parallel_{false};
};

/** Model-level limits loaded from model metadata. */
struct ModelConfig final {
  /** Maximum length of one sequence, including prompt and generated tokens. */
  int32_t max_model_len_{0};
};

/** KV-cache storage policy. */
struct KVCacheConfig final {
  /** Number of tokens stored in one paged KV-cache block. */
  int32_t block_size_{0};

  /** Device memory budget available to the KV cache on each rank. */
  uint64_t kv_cache_memory_bytes_{0};
};

/** Parallelism, model, and cache inputs consumed by the Planner. */
struct PlanConfig final {
  /** TP, DP, and EP configuration. */
  ParallelConfig parallel_;

  /** Model limits used by the scheduler and KV cache manager. */
  ModelConfig model_;

  /** KV-cache memory and paging policy. */
  KVCacheConfig kv_cache_;
};

/** Scheduling limits applied when forming one engine step. */
struct SchedulerConfig final {
  /** Maximum packed tokens selected in one scheduler iteration. */
  int32_t max_num_batched_tokens_{0};

  /** Maximum sequences selected in one scheduler iteration. */
  int32_t max_num_seqs_{0};

  /** Decoder-only maximum prompt tokens admitted from one sequence in one step. */
  int32_t prefill_chunk_size_{0};
};

/** Immutable startup configuration shared by the Engine's components. */
struct Config final {
  /** Directory containing the model checkpoint and metadata. */
  std::filesystem::path model_directory_;

  /** Task-level runner selected for this Engine. */
  RunnerMode runner_mode_{RunnerMode::DECODER};

  /** Runtime resource configuration. */
  RuntimeConfig runtime_;

  /** Planner configuration. */
  PlanConfig plan_;

  /** Scheduler configuration; zero-initialized in embedding mode. */
  SchedulerConfig scheduler_;

  /** Validates configuration-local invariants before Engine construction. */
  void Validate() const;
};

}  // namespace zephyr
