#pragma once

#include <source_location>
#include <utility>

#include "ttl/internal/runtime/memory/allocation_budget.hpp"
#include "ttl/internal/runtime/memory/pinned/allocation.hpp"
#include "ttl/internal/runtime/memory/pinned/stream_usage.hpp"

namespace ttl::internal {

/** Pinned allocation and per-device events retained until every recorded stream completes. */
struct PinnedRetirement final {
  PinnedRetirement(PinnedAllocation allocation, PinnedStreamUsageSnapshot usage, AllocationBudget::Reservation budget,
                   std::source_location location, bool poisoned = false) noexcept
      : allocation_(allocation),
        usage_(std::move(usage)),
        budget_(std::move(budget)),
        location_(location),
        poisoned_(poisoned) {}

  PinnedRetirement(const PinnedRetirement &) = delete;
  auto operator=(const PinnedRetirement &) -> PinnedRetirement & = delete;
  PinnedRetirement(PinnedRetirement &&) noexcept = default;
  auto operator=(PinnedRetirement &&) noexcept -> PinnedRetirement & = default;

  PinnedAllocation allocation_;
  PinnedStreamUsageSnapshot usage_;
  AllocationBudget::Reservation budget_;
  std::source_location location_;
  bool poisoned_;
};

}  // namespace ttl::internal
