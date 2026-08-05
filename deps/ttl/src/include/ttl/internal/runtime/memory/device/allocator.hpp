#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <string_view>
#include <vector>

#include "ttl/common/device.hpp"
#include "ttl/internal/runtime/memory/device/allocation.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl {

class ErrorSink;

}  // namespace ttl

namespace ttl::internal {

class DeviceAllocatorImpl;
class EventPool;
class Storage;
class StreamState;
struct DeviceStreamUsageSnapshot;

struct DeviceAllocatorOptions final {
  uint64_t release_threshold_bytes_{std::numeric_limits<uint64_t>::max()};
  uint64_t max_live_bytes_{0};
  bool enable_maintenance_thread_{true};
};

struct AllocationContext final {
  std::string_view operation_;
  std::optional<Shape> output_shape_;
  std::optional<DType> dtype_;
  std::source_location location_;
};

struct DeviceAllocatorStats final {
  uint64_t logical_live_bytes_;
  uint64_t retiring_bytes_;
  uint64_t peak_physical_in_use_bytes_;
  uint64_t allocation_count_;
  uint64_t retirement_count_;
  uint64_t retry_count_;
  uint64_t oom_count_;
  uint64_t trim_count_;
  uint64_t pending_retirement_count_;
  uint64_t quarantined_retirement_count_;
  uint64_t quarantined_bytes_;
  uint64_t pool_used_bytes_;
  uint64_t pool_reserved_bytes_;
  uint64_t outstanding_storage_count_;
};

/**
 * @brief Per-device stream-ordered allocator backed by one private CUDA memory pool.
 *
 * Create is the only construction path because every Storage must retain a shared allocator owner. Allocate and
 * WrapExternal are thread-safe. Shutdown is an explicit, potentially blocking lifecycle boundary.
 */
class DeviceAllocator final : public std::enable_shared_from_this<DeviceAllocator> {
 public:
  [[nodiscard]] static auto Create(Device device, const std::shared_ptr<ErrorSink> &error_sink,
                                   const std::shared_ptr<EventPool> &event_pool, DeviceAllocatorOptions options = {},
                                   std::source_location location = std::source_location::current())
      -> std::shared_ptr<DeviceAllocator>;

  DeviceAllocator(const DeviceAllocator &) = delete;
  auto operator=(const DeviceAllocator &) -> DeviceAllocator & = delete;
  DeviceAllocator(DeviceAllocator &&) = delete;
  auto operator=(DeviceAllocator &&) -> DeviceAllocator & = delete;

  ~DeviceAllocator() noexcept;

  [[nodiscard]] auto Allocate(const Stream &stream, size_t bytes, size_t alignment, const AllocationContext &context)
      -> std::shared_ptr<Storage>;

  [[nodiscard]] auto WrapExternal(const Stream &allocation_stream, void *pointer, size_t capacity_bytes,
                                  ExternalOwnership ownership, std::shared_ptr<void> owner,
                                  std::source_location location = std::source_location::current())
      -> std::shared_ptr<Storage>;

  void SetPeerAccess(Device peer, bool enabled, std::source_location location = std::source_location::current());
  void Poll() noexcept;
  void TrimTo(size_t target_reserved_bytes, std::source_location location = std::source_location::current());
  [[nodiscard]] auto GetStats(std::source_location location = std::source_location::current()) const
      -> DeviceAllocatorStats;
  void Shutdown(std::source_location location = std::source_location::current());

  [[nodiscard]] auto GetDevice() const noexcept -> Device;

 private:
  friend class Storage;

  explicit DeviceAllocator(std::unique_ptr<DeviceAllocatorImpl> impl) noexcept;

  void Retire(Allocation allocation, DeviceStreamUsageSnapshot usage) noexcept;

  std::unique_ptr<DeviceAllocatorImpl> impl_;
};

}  // namespace ttl::internal
