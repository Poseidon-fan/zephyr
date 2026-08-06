#pragma once

#include <cstddef>
#include <memory>

#include "ttl/common/device.hpp"
#include "ttl/internal/runtime/memory/device/allocation.hpp"
#include "ttl/internal/runtime/memory/device/stream_usage.hpp"
#include "ttl/internal/runtime/memory/retirement_ticket.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {

class DeviceAllocator;
class DeviceAllocatorImpl;
class StreamState;
class TensorFactory;

/** Shared, fixed-capacity owner of one device allocation. */
class Storage final {
 public:
  Storage(const Storage &) = delete;
  auto operator=(const Storage &) -> Storage & = delete;
  Storage(Storage &&) = delete;
  auto operator=(Storage &&) -> Storage & = delete;

  ~Storage() noexcept;

  [[nodiscard]] auto GetBasePointer() const noexcept -> void *;
  [[nodiscard]] auto GetCapacityBytes() const noexcept -> size_t;
  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetAllocationKind() const noexcept -> AllocationKind;

  /** Retain the stream until this allocation has been safely retired. */
  void RecordUsage(const Stream &stream);

 private:
  friend class DeviceAllocator;
  friend class DeviceAllocatorImpl;
  friend class TensorFactory;

  Storage(Allocation allocation, RetirementTicket retirement_ticket, Device device,
          std::shared_ptr<DeviceAllocator> allocator, std::shared_ptr<StreamState> allocation_stream) noexcept;

  Allocation allocation_;
  RetirementTicket retirement_ticket_;
  Device device_;
  std::shared_ptr<DeviceAllocator> allocator_;
  DeviceStreamUsage usage_;
};

}  // namespace ttl::internal
