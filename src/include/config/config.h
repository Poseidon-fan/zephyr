#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
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

/** Per-Worker capacities fixed into each executable plan. */
struct ExecutionLimits final {
  /** Maximum packed token rows processed by one Worker invocation. */
  int32_t max_tokens_per_worker_{0};

  /** Maximum sequences processed by one Worker invocation. */
  int32_t max_sequences_per_worker_{0};

  /** Maximum logical length of one sequence. */
  int32_t max_sequence_length_{0};

  /** Device-memory budget available to one Worker's KV cache. */
  uint64_t kv_cache_bytes_per_worker_{0};
};

/** Parallelism and capacity inputs consumed by the Planner. */
struct PlanConfig final {
  /** TP, DP, and EP configuration. */
  ParallelConfig parallel_;

  /** Per-Worker execution limits. */
  ExecutionLimits limits_;

  /** Number of tokens stored in one paged KV-cache block. */
  int32_t kv_cache_block_size_{0};

  /** Maximum per-Worker extent for each named dynamic dimension. */
  std::unordered_map<std::string, int64_t> dynamic_dimension_capacities_;
};

/** Decoder scheduling limits applied when forming one engine step. */
struct SchedulerConfig final {
  /** Maximum packed tokens selected across all DP replicas. */
  int32_t max_batch_tokens_{0};

  /** Maximum sequences selected across all DP replicas. */
  int32_t max_batch_sequences_{0};

  /** Maximum prompt tokens admitted from one sequence in one step. */
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

  /** Decoder scheduler configuration; zero-initialized in embedding mode. */
  SchedulerConfig scheduler_;

  /** Validates configuration-local invariants before Engine construction. */
  void Validate() const;
};

}  // namespace zephyr
