#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "scheduler/sequence.hpp"

namespace zephyr::kv_cache {
class KVCacheManager;
}

namespace zephyr::scheduler {

struct LengthBucketSchedulerConfig final {
  /** Admission limit when waiting and running work coexist, not a limit on every length bucket. */
  size_t max_num_seqs_;
};

struct PagedSchedulerConfig final {
  /** Maximum resident sequences, including rows omitted from the current batch. */
  size_t max_num_seqs_;
  /** New tokens per batch; the first decode row may exceed this budget to ensure progress. */
  size_t max_num_batched_tokens_;
  /** Per-sequence prompt quantum while decode work is resident. */
  size_t max_prefill_chunk_tokens_;
  size_t max_decode_steps_before_prefill_;
};

using SchedulerConfig = std::variant<LengthBucketSchedulerConfig, PagedSchedulerConfig>;

/** Logical request phase. Uncached DECODE still executes a full-history prefill in the executor. */
enum class BatchPhase : uint8_t { PREFILL, DECODE };

/** A slice of the sequence's token history; start_token_ also gives its preceding context length. */
struct ScheduledSequence final {
  sequence_id_t sequence_id_;
  size_t start_token_;
  size_t num_tokens_;
};

/** Ordered logical inputs; execution results retain this sequence order. */
struct ScheduledBatch final {
  BatchPhase phase_{BatchPhase::PREFILL};
  /** Applies only to PREFILL. Intermediate chunks advance KV without producing a final result. */
  bool is_final_prompt_chunk_{false};
  std::vector<ScheduledSequence> sequences_;
};

/** Nonempty batches in execution order: length buckets put decode first; paged scheduling returns at most one. */
struct SchedulePlan final {
  std::vector<ScheduledBatch> batches_;
};

/** Queue occupancy; sequence execution states do not imply queue membership. */
struct SchedulerStatistics final {
  size_t num_running_sequences_{0};
  size_t num_waiting_sequences_{0};
};

/**
 * Control-thread interface over engine-owned sequences and optional paged KV storage.
 * Borrowed objects outlive the scheduler. Finish every returned batch before scheduling again or
 * removing its sequences. No token history or cache state may change while ranks are executing.
 *
 * After all ranks succeed, the engine commits executed range ends to num_computed_tokens_ only when
 * KV was stored. A replay does not increase that end. Uncached execution never creates reusable KV.
 * After a final prompt or decode, the engine consumes the result, then either appends a sampled token
 * and sets DECODE, or marks FINISHED. Errors set ERROR without committing failed computation.
 *
 * Schedule cleans terminal registrations before selecting work. Remove must precede erasing a sequence
 * from the table. Before destroying a scheduler, remove its registered sequences unless their table and
 * cache manager are being discarded together. All rank accesses must have stopped before page release.
 */
class Scheduler {
 public:
  /**
   * Construct a scheduling policy. Without a cache manager, paged configuration falls back to length
   * buckets using only max_num_seqs_. A length-bucket configuration requires uncached execution.
   * supports_packed_prefill comes from the executor's capabilities and applies only to paged scheduling.
   */
  [[nodiscard]] static auto Create(const SchedulerConfig &config, SequenceTable &sequences,
                                   kv_cache::KVCacheManager *cache_manager = nullptr,
                                   bool supports_packed_prefill = false) -> std::unique_ptr<Scheduler>;

  Scheduler(const Scheduler &) = delete;
  auto operator=(const Scheduler &) -> Scheduler & = delete;
  Scheduler(Scheduler &&) = delete;
  auto operator=(Scheduler &&) -> Scheduler & = delete;
  virtual ~Scheduler() = default;

  /** Register a sequence already present in the engine's table. */
  virtual void Add(sequence_id_t sequence_id) = 0;
  /** Select and prepare work without committing execution progress. An empty plan submits no work. */
  [[nodiscard]] virtual auto Schedule() -> SchedulePlan = 0;
  /** Release registration and resources; active removal is cancellation. Safe to repeat. */
  virtual void Remove(sequence_id_t sequence_id) = 0;
  /** Read the scheduler's queues on the owning control thread. */
  [[nodiscard]] virtual auto GetStatistics() const noexcept -> SchedulerStatistics = 0;

 protected:
  Scheduler() = default;
};

}  // namespace zephyr::scheduler
