#include "scheduler/scheduler.hpp"

#include <memory>
#include <variant>

#include "common/exception.hpp"
#include "scheduler/length_bucket_scheduler.hpp"
#include "scheduler/paged_scheduler.hpp"

namespace zephyr::scheduler {

auto Scheduler::Create(const SchedulerConfig &config, SequenceTable &sequences, kv_cache::KVCacheManager *cache_manager,
                       bool supports_packed_prefill) -> std::unique_ptr<Scheduler> {
  if (const auto *paged = std::get_if<PagedSchedulerConfig>(&config); paged != nullptr) {
    if (cache_manager != nullptr) {
      return std::make_unique<PagedScheduler>(*paged, sequences, *cache_manager, supports_packed_prefill);
    }
    return std::make_unique<LengthBucketScheduler>(paged->max_num_seqs_, sequences);
  }
  if (cache_manager != nullptr) {
    throw ConfigurationException("Length-bucket scheduling cannot manage paged KV storage");
  }
  return std::make_unique<LengthBucketScheduler>(std::get<LengthBucketSchedulerConfig>(config).max_num_seqs_,
                                                 sequences);
}

}  // namespace zephyr::scheduler
