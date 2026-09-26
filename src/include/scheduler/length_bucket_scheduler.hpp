#pragma once

#include <cstddef>
#include <deque>
#include <unordered_map>
#include <vector>

#include "scheduler/scheduler.hpp"

namespace zephyr::scheduler {

/**
 * Schedules complete token histories in equal-length buckets, using length and waiting urgency to choose work.
 * The sequence limit applies when admitting waiting work alongside running work. With only waiting work,
 * all sequences enter length buckets and the shortest bucket runs, without a token-count or sequence limit.
 */
class LengthBucketScheduler final : public Scheduler {
 public:
  LengthBucketScheduler(size_t max_num_seqs, SequenceTable &sequences);

  /** Register a waiting or already running sequence with a nonempty token history. */
  void Add(sequence_id_t sequence_id) override;
  /** Select one length bucket; decode precedes prefill, and every row uses its complete token history. */
  [[nodiscard]] auto Schedule() -> SchedulePlan override;
  /** Remove registration; an active sequence becomes FINISHED, while terminal states are preserved. */
  void Remove(sequence_id_t sequence_id) override;

 private:
  /** Choose the shortest bucket in discrete mode, otherwise the greatest summed scheduling priority. */
  void BucketAndWaitlist(bool discrete);

  size_t max_num_seqs_;
  SequenceTable &sequences_;
  std::vector<sequence_id_t> running_;
  std::deque<sequence_id_t> waiting_;
  std::unordered_map<sequence_id_t, size_t> scheduling_urgency_;
};

}  // namespace zephyr::scheduler
