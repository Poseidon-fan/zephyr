#include "ttl/internal/storage.hpp"

#include <cstddef>
#include <memory>
#include <utility>

#include "ttl/device.hpp"
#include "ttl/internal/device_allocator.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {

Storage::Storage(Allocation allocation, Device device, std::shared_ptr<DeviceAllocator> allocator,
                 std::shared_ptr<StreamState> allocation_stream)
    : allocation_(std::move(allocation)),
      device_(device),
      allocator_(std::move(allocator)),
      usage_(std::move(allocation_stream), allocation_.location_) {}

Storage::~Storage() noexcept {
  allocator_->Retire(std::move(allocation_), usage_.GetAllocationStream(), std::move(usage_).TakeSideStreams());
}

auto Storage::GetBasePointer() const noexcept -> void * { return allocation_.pointer_; }

auto Storage::GetCapacityBytes() const noexcept -> size_t { return allocation_.capacity_bytes_; }

auto Storage::GetDevice() const noexcept -> Device { return device_; }

auto Storage::GetAllocationKind() const noexcept -> AllocationKind { return allocation_.kind_; }

void Storage::RecordUsage(const Stream &stream) { usage_.Record(StreamAccess::GetState(stream)); }

}  // namespace ttl::internal
