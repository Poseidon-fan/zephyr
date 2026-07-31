#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <source_location>
#include <span>
#include <vector>

#include "ttl/internal/event_pool.hpp"
#include "ttl/pinned_buffer.hpp"
#include "ttl/runtime.hpp"
#include "ttl/stream.hpp"

namespace ttl {

class ErrorSink;

}  // namespace ttl

namespace ttl::internal {

class EventPool;
class PinnedAllocatorImpl;
class StreamState;

struct PinnedAllocation final {
  void *pointer_;
  size_t capacity_bytes_;
};

/** Snapshot of process-wide pinned allocator counters. */
struct PinnedAllocatorStats final {
  uint64_t live_bytes_;
  uint64_t pending_bytes_;
  uint64_t cached_bytes_;
  uint64_t physical_bytes_;
  uint64_t peak_physical_bytes_;
  uint64_t host_allocation_count_;
  uint64_t host_free_count_;
  uint64_t cache_hit_count_;
  uint64_t retirement_count_;
  uint64_t pending_retirement_count_;
  uint64_t outstanding_buffer_count_;
};

/** Shared live allocation behind one or more public PinnedBuffer handles. */
class PinnedBlock final {
 public:
  PinnedBlock(const PinnedBlock &) = delete;
  auto operator=(const PinnedBlock &) -> PinnedBlock & = delete;
  PinnedBlock(PinnedBlock &&) = delete;
  auto operator=(PinnedBlock &&) -> PinnedBlock & = delete;

  PinnedBlock(PinnedAllocation allocation, size_t size_bytes, std::shared_ptr<PinnedAllocator> allocator,
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
  std::source_location location_;
  std::mutex usage_latch_;
  std::vector<std::shared_ptr<StreamState>> streams_;
  std::vector<PooledEvent> retirement_events_;
};

/** Private gateway for validating and recording public pinned buffers. */
class PinnedBufferAccess final {
 public:
  [[nodiscard]] static auto GetBlock(const PinnedBuffer &buffer, std::source_location location) -> PinnedBlock &;
};

/**
 * Thread-safe process-wide cache of CUDA page-locked host allocations.
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
  void Retire(PinnedAllocation allocation, std::vector<std::shared_ptr<StreamState>> streams,
              std::vector<PooledEvent> events, std::source_location location) noexcept;

  std::unique_ptr<PinnedAllocatorImpl> impl_;
};

}  // namespace ttl::internal
