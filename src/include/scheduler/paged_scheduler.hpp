#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache/manager.hpp"
#include "scheduler/scheduler.hpp"

namespace zephyr::scheduler {

/**
 * Token-budgeted scheduling with prompt reservations, chunked prefill and decode preemption.
 * Borrows the engine's logical page manager. This execution path accepts PROMPT_AND_DECODE inputs;
 * whole-input tasks currently execute without KV through LengthBucketScheduler.
 */
class PagedScheduler final : public Scheduler {
 public:
  PagedScheduler(PagedSchedulerConfig config, SequenceTable &sequences, kv_cache::KVCacheManager &cache_manager,
                 bool supports_packed_prefill);

  /** Register a fresh PROMPT_AND_DECODE sequence already present in the engine's table. */
  void Add(sequence_id_t sequence_id) override;
  /** Clean terminal sequences, then select work and allocate pages without committing progress. */
  [[nodiscard]] auto Schedule() -> SchedulePlan override;
  /** Release a registered sequence; removing an active sequence cancels it. Safe to repeat. */
  void Remove(sequence_id_t sequence_id) override;
  [[nodiscard]] auto GetStatistics() const noexcept -> SchedulerStatistics override {
    return {.num_running_sequences_ = running_.size(), .num_waiting_sequences_ = waiting_.size()};
  }

 private:
  struct SequenceEntry final {
    size_t arrival_order_;
    size_t waiting_count_{0};
    std::vector<kv_cache::block_hash_t> block_hashes_;
  };

  [[nodiscard]] auto GetPrefillBudget() const -> size_t;
  [[nodiscard]] auto IsDecodeDue() const -> bool;
  void AdmitWaiting();
  void RejectWaiting(SequenceState state, std::string reason);
  void CacheComputedBlocks(sequence_id_t sequence_id);
  /** Caller first removes the victim from running_. */
  void Preempt(sequence_id_t sequence_id);
  [[nodiscard]] auto SchedulePrefill() -> ScheduledBatch;
  [[nodiscard]] auto ScheduleDecode() -> ScheduledBatch;

  PagedSchedulerConfig config_;
  SequenceTable &sequences_;
  kv_cache::KVCacheManager &cache_manager_;
  bool supports_packed_prefill_;
  std::unordered_map<sequence_id_t, SequenceEntry> entries_;
  std::deque<sequence_id_t> waiting_;
  // Admissions append; only decode allocation sorts by the original arrival order.
  std::deque<sequence_id_t> running_;
  size_t next_arrival_order_{0};
  size_t decode_steps_since_prefill_{0};
  bool prompt_admission_epoch_{false};
  std::optional<sequence_id_t> next_prompt_sequence_id_;
  size_t decode_cursor_{0};
};

}  // namespace zephyr::scheduler
