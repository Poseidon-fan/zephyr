#include "scheduler/length_bucket_scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

#include "common/exception.hpp"

namespace zephyr::scheduler {

LengthBucketScheduler::LengthBucketScheduler(size_t max_num_seqs, SequenceTable &sequences)
    : max_num_seqs_(max_num_seqs), sequences_(sequences) {
  if (max_num_seqs == 0) {
    throw ConfigurationException("Length bucket scheduler sequence limit must be positive");
  }
}

void LengthBucketScheduler::Add(sequence_id_t sequence_id) {
  const auto &sequence = sequences_.at(sequence_id);
  if (sequence.id_ != sequence_id || sequence.token_ids_.empty() ||
      (sequence.state_ != SequenceState::WAITING && !sequence.IsRunning()) ||
      scheduling_urgency_.contains(sequence_id)) {
    throw InvalidArgumentException("Length bucket scheduler requires a unique waiting or running sequence with tokens");
  }
  scheduling_urgency_.emplace(sequence_id, 0);
  if (sequence.IsRunning()) {
    running_.push_back(sequence_id);
  } else {
    waiting_.push_back(sequence_id);
  }
}

void LengthBucketScheduler::BucketAndWaitlist(bool discrete) {
  struct Bucket final {
    std::vector<sequence_id_t> sequences_;
    double priority_{0.0};
  };
  std::map<size_t, Bucket> buckets;
  for (const auto id : running_) {
    const auto length = sequences_.at(id).token_ids_.size();
    auto &bucket = buckets[length];
    bucket.sequences_.push_back(id);
    if (!discrete) {
      bucket.priority_ += static_cast<double>(scheduling_urgency_.at(id)) + std::log2(static_cast<double>(length));
    }
  }
  if (buckets.empty()) {
    return;
  }

  auto selected = buckets.begin();
  if (!discrete) {
    // Ordered lengths make equal priorities deterministic: the shorter bucket wins.
    selected = std::ranges::max_element(
        buckets, [](const auto &left, const auto &right) { return left.second.priority_ < right.second.priority_; });
  }
  running_ = std::move(selected->second.sequences_);
  for (const auto id : running_) {
    scheduling_urgency_.at(id) = 0;
  }
  for (const auto &[length, bucket] : buckets) {
    if (length == selected->first) {
      continue;
    }
    for (const auto id : bucket.sequences_) {
      ++scheduling_urgency_.at(id);
      waiting_.push_back(id);
    }
  }
}

auto LengthBucketScheduler::Schedule() -> SchedulePlan {
  const auto finished = [&](auto id) {
    if (!sequences_.at(id).IsTerminal()) {
      return false;
    }
    scheduling_urgency_.erase(id);
    return true;
  };
  std::erase_if(running_, finished);
  std::erase_if(waiting_, finished);

  const bool prompt_only = running_.empty();
  const bool completion_only = waiting_.empty();
  if (prompt_only) {
    for (const auto id : waiting_) {
      sequences_.at(id).state_ = SequenceState::PREFILL;
      running_.push_back(id);
    }
    waiting_.clear();
    BucketAndWaitlist(true);
  } else if (completion_only) {
    BucketAndWaitlist(true);
  } else {
    std::ranges::sort(waiting_);
    while (!waiting_.empty() && running_.size() < max_num_seqs_) {
      const auto id = waiting_.front();
      waiting_.pop_front();
      auto &sequence = sequences_.at(id);
      if (sequence.state_ == SequenceState::WAITING) {
        sequence.state_ = SequenceState::PREFILL;
      }
      running_.push_back(id);
    }
    BucketAndWaitlist(false);
  }

  ScheduledBatch completion{.phase_ = BatchPhase::DECODE, .sequences_ = {}};
  ScheduledBatch prompt{.phase_ = BatchPhase::PREFILL, .is_final_prompt_chunk_ = true, .sequences_ = {}};
  for (const auto id : running_) {
    const auto &sequence = sequences_.at(id);
    auto &batch = !prompt_only && (completion_only || sequence.state_ == SequenceState::DECODE) ? completion : prompt;
    batch.sequences_.push_back({.sequence_id_ = id, .start_token_ = 0, .num_tokens_ = sequence.token_ids_.size()});
  }
  SchedulePlan plan;
  if (!completion.sequences_.empty()) {
    plan.batches_.push_back(std::move(completion));
  }
  if (!prompt.sequences_.empty()) {
    plan.batches_.push_back(std::move(prompt));
  }
  return plan;
}

void LengthBucketScheduler::Remove(sequence_id_t sequence_id) {
  if (!scheduling_urgency_.contains(sequence_id)) {
    return;
  }
  auto &sequence = sequences_.at(sequence_id);
  if (!sequence.IsTerminal()) {
    sequence.state_ = SequenceState::FINISHED;
  }
  std::erase(running_, sequence_id);
  std::erase(waiting_, sequence_id);
  scheduling_urgency_.erase(sequence_id);
}

}  // namespace zephyr::scheduler
