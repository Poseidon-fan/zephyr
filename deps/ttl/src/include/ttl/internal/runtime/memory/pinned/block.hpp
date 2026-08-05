#pragma once

#include <cstddef>
#include <memory>
#include <source_location>

#include "ttl/internal/runtime/memory/allocation_budget.hpp"
#include "ttl/internal/runtime/memory/pinned/allocation.hpp"
#include "ttl/internal/runtime/memory/pinned/stream_usage.hpp"
#include "ttl/internal/runtime/memory/retirement_ticket.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {

class PinnedAllocator;
class PinnedAllocatorImpl;
class StreamState;

class PinnedBlock final {
 public:
  PinnedBlock(const PinnedBlock &) = delete;
  auto operator=(const PinnedBlock &) -> PinnedBlock & = delete;
  PinnedBlock(PinnedBlock &&) = delete;
  auto operator=(PinnedBlock &&) -> PinnedBlock & = delete;

  PinnedBlock(PinnedAllocation allocation, size_t size_bytes, std::shared_ptr<PinnedAllocator> allocator,
              AllocationBudget::Reservation budget, RetirementTicket retirement_ticket,
              std::source_location location) noexcept;
  ~PinnedBlock() noexcept;

  [[nodiscard]] auto GetData() const noexcept -> void *;
  [[nodiscard]] auto GetSizeBytes() const noexcept -> size_t;
  void RecordUsage(const Stream &stream, std::source_location location);

 private:
  friend class PinnedAllocator;
  friend class PinnedAllocatorImpl;

  PinnedAllocation allocation_;
  size_t size_bytes_;
  std::shared_ptr<PinnedAllocator> allocator_;
  AllocationBudget::Reservation budget_;
  RetirementTicket retirement_ticket_;
  std::source_location location_;
  PinnedStreamUsage usage_;
};

class PinnedBufferAccess final {
 public:
  [[nodiscard]] static auto GetBlock(const PinnedBuffer &buffer, std::source_location location) -> PinnedBlock &;
};

}  // namespace ttl::internal
