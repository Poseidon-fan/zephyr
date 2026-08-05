#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <span>

#include "ttl/internal/runtime/memory/allocation_budget.hpp"
#include "ttl/internal/runtime/memory/pinned/allocation.hpp"
#include "ttl/internal/runtime/memory/retirement_ticket.hpp"
#include "ttl/runtime/memory.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl {

class ErrorSink;

}  // namespace ttl

namespace ttl::internal {

class EventPool;
class PinnedBlock;
class PinnedAllocatorImpl;
class StreamState;
struct PinnedStreamUsageSnapshot;

/** Snapshot of process-wide pinned allocator counters, including fail-stop quarantine telemetry. */
struct PinnedAllocatorStats final {
  uint64_t logical_live_bytes_;
  uint64_t live_capacity_bytes_;
  uint64_t retiring_capacity_bytes_;
  uint64_t cached_capacity_bytes_;
  uint64_t budgeted_capacity_bytes_;
  uint64_t peak_budgeted_capacity_bytes_;
  uint64_t physical_bytes_;
  uint64_t peak_physical_bytes_;
  uint64_t host_allocation_count_;
  uint64_t host_free_count_;
  uint64_t cache_hit_count_;
  uint64_t retirement_count_;
  uint64_t pending_retirement_count_;
  uint64_t quarantined_retirement_count_;
  uint64_t quarantined_bytes_;
  uint64_t outstanding_buffer_count_;
};

/**
 * @brief Thread-safe process-wide cache of CUDA page-locked host allocations.
 *
 * Live blocks retain this allocator. Shutdown rejects outstanding public buffers, drains retirement fences, and frees
 * every cached allocation.
 */
class PinnedAllocator final : public std::enable_shared_from_this<PinnedAllocator> {
 public:
  [[nodiscard]] static auto Create(const std::shared_ptr<ErrorSink> &error_sink,
                                   std::span<const std::shared_ptr<EventPool>> event_pools,
                                   PinnedMemoryOptions options = {},
                                   std::source_location location = std::source_location::current())
      -> std::shared_ptr<PinnedAllocator>;

  PinnedAllocator(const PinnedAllocator &) = delete;
  auto operator=(const PinnedAllocator &) -> PinnedAllocator & = delete;
  PinnedAllocator(PinnedAllocator &&) = delete;
  auto operator=(PinnedAllocator &&) -> PinnedAllocator & = delete;

  ~PinnedAllocator() noexcept;

  [[nodiscard]] auto Allocate(size_t bytes, std::source_location location = std::source_location::current())
      -> PinnedBuffer;
  void Poll() noexcept;
  void Trim(std::source_location location = std::source_location::current());
  void Shutdown(std::source_location location = std::source_location::current());

  [[nodiscard]] auto GetStats() const noexcept -> PinnedAllocatorStats;

 private:
  friend class PinnedBlock;

  explicit PinnedAllocator(std::unique_ptr<PinnedAllocatorImpl> impl) noexcept;
  void Retire(PinnedAllocation allocation, size_t logical_size_bytes, AllocationBudget::Reservation budget,
              RetirementTicket retirement_ticket, PinnedStreamUsageSnapshot usage,
              std::source_location location) noexcept;

  std::unique_ptr<PinnedAllocatorImpl> impl_;
};

}  // namespace ttl::internal
