#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <utility>

namespace ttl::internal {

enum class AllocationKind : uint8_t {
  DEVICE_POOL,
  EXTERNAL_BORROWED,
  EXTERNAL_OWNED,
};

enum class ExternalOwnership : uint8_t {
  BORROWED,
  SHARED_OWNER,
};

/**
 * Move-only resource record transferred from Storage to allocator retirement.
 *
 * Allocation does not release its pointer. DeviceAllocator is the only component that consumes the record and applies
 * the release policy described by kind_.
 */
class Allocation final {
 public:
  Allocation(void *pointer, size_t capacity_bytes, AllocationKind kind, std::shared_ptr<void> external_owner,
             std::source_location location) noexcept
      : pointer_(pointer),
        capacity_bytes_(capacity_bytes),
        kind_(kind),
        external_owner_(std::move(external_owner)),
        location_(location) {}

  Allocation(const Allocation &) = delete;
  auto operator=(const Allocation &) -> Allocation & = delete;
  Allocation(Allocation &&) noexcept = default;
  auto operator=(Allocation &&) noexcept -> Allocation & = default;

  void *pointer_;
  size_t capacity_bytes_;
  AllocationKind kind_;
  std::shared_ptr<void> external_owner_;
  std::source_location location_;
};

}  // namespace ttl::internal
