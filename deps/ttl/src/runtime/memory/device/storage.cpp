#include "ttl/internal/runtime/memory/device/storage.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include "ttl/common/device.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/memory/device/allocator.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {

Storage::Storage(Allocation allocation, RetirementTicket retirement_ticket, Device device,
                 std::shared_ptr<DeviceAllocator> allocator, std::shared_ptr<StreamState> allocation_stream) noexcept
    : allocation_(std::move(allocation)),
      retirement_ticket_(std::move(retirement_ticket)),
      device_(device),
      allocator_(std::move(allocator)),
      usage_(std::move(allocation_stream)) {}

Storage::~Storage() noexcept {
  allocator_->Retire(std::move(allocation_), std::move(retirement_ticket_), std::move(usage_).TakeSnapshot());
}

auto Storage::GetBasePointer() const noexcept -> void * { return allocation_.pointer_; }

auto Storage::GetCapacityBytes() const noexcept -> size_t { return allocation_.capacity_bytes_; }

auto Storage::GetDevice() const noexcept -> Device { return device_; }

auto Storage::GetAllocationKind() const noexcept -> AllocationKind { return allocation_.kind_; }

void Storage::RecordUsage(const Stream &stream) { usage_.Record(StreamAccess::GetState(stream)); }

}  // namespace ttl::internal
