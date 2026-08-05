#include "ttl/internal/runtime/memory/pinned/block.hpp"

#include <cstddef>
#include <memory>
#include <source_location>
#include <utility>

#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/memory/pinned/allocator.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {

PinnedBlock::PinnedBlock(PinnedAllocation allocation, size_t size_bytes, std::shared_ptr<PinnedAllocator> allocator,
                         AllocationBudget::Reservation budget, RetirementTicket retirement_ticket,
                         std::source_location location) noexcept
    : allocation_(allocation),
      size_bytes_(size_bytes),
      allocator_(std::move(allocator)),
      budget_(std::move(budget)),
      retirement_ticket_(std::move(retirement_ticket)),
      location_(location) {}

PinnedBlock::~PinnedBlock() noexcept {
  allocator_->Retire(allocation_, size_bytes_, std::move(budget_), std::move(retirement_ticket_),
                     std::move(usage_).TakeSnapshot(), location_);
}

auto PinnedBlock::GetData() const noexcept -> void * { return allocation_.pointer_; }

auto PinnedBlock::GetSizeBytes() const noexcept -> size_t { return size_bytes_; }

void PinnedBlock::RecordUsage(const Stream &stream, std::source_location location) {
  usage_.Record(StreamAccess::GetState(stream), location);
}

auto PinnedBufferAccess::GetBlock(const PinnedBuffer &buffer, std::source_location location) -> PinnedBlock & {
  if (buffer.block_ == nullptr) {
    throw InvalidArgumentError("pinned buffer is in a moved-from state", location);
  }
  return *buffer.block_;
}

}  // namespace ttl::internal
