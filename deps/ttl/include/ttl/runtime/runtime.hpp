#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <source_location>
#include <span>
#include <vector>

#include <driver_types.h>

#include "ttl/common/device.hpp"
#include "ttl/runtime/device_properties.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl {

class ErrorSink;
class Tensor;

namespace internal {

class RuntimeAccess;

}  // namespace internal

/** @brief Configures one CUDA device's stream-ordered memory pool. */
struct DeviceMemoryOptions final {
  /** Bytes the CUDA pool may retain after frees; defaults to no automatic release. */
  uint64_t release_threshold_bytes_{std::numeric_limits<uint64_t>::max()};
  /** Maximum bytes allocated or asynchronously retiring; zero disables the TTL budget. */
  uint64_t max_live_bytes_{0};
  /** Whether a host thread polls completed asynchronous retirements. */
  bool enable_maintenance_thread_{true};
};

/** @brief Configures the process-wide page-locked host-memory cache and admission budget. */
struct PinnedMemoryOptions final {
  size_t max_cached_bytes_{256U * 1024U * 1024U};
  size_t max_live_bytes_{512U * 1024U * 1024U};
};

/** @brief Configures devices and process-local services owned by a `Runtime`. */
struct RuntimeOptions final {
  /** Unique CUDA devices registered with the runtime. */
  std::vector<Device> devices_;
  DeviceMemoryOptions device_memory_;
  /** Per-handle cuBLAS workspace bytes; zero selects an architecture-dependent default. */
  size_t blas_workspace_bytes_{0};
  size_t event_pool_capacity_per_device_{256};
  size_t event_pool_reserve_per_device_{0};
  std::shared_ptr<ErrorSink> error_sink_;
  PinnedMemoryOptions pinned_memory_;
};

/** @brief Public lifecycle state of a runtime. */
enum class RuntimeStatus : uint8_t {
  RUNNING,
  CLOSING,
  CLOSED,
};

/** @brief Snapshot of allocator, event, and cuBLAS resources for one registered device. */
struct DeviceMemoryStatistics final {
  Device device_;
  uint64_t logical_live_bytes_;
  uint64_t retiring_bytes_;
  uint64_t peak_physical_in_use_bytes_;
  uint64_t allocation_count_;
  uint64_t retirement_count_;
  uint64_t retry_count_;
  uint64_t oom_count_;
  uint64_t trim_count_;
  uint64_t pending_retirement_count_;
  /** Retirements whose native completion path failed and are awaiting explicit Shutdown drain. */
  uint64_t quarantined_retirement_count_;
  uint64_t quarantined_bytes_;
  uint64_t pool_used_bytes_;
  uint64_t pool_reserved_bytes_;
  uint64_t outstanding_storage_count_;
  size_t cached_event_count_;
  size_t outstanding_event_count_;
  size_t event_cache_capacity_;
  size_t blas_workspace_bytes_;
};

/** @brief Snapshot of process-wide page-locked host-memory accounting. */
struct PinnedMemoryStatistics final {
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
  /** Retirements whose native completion path failed and are awaiting explicit Shutdown drain. */
  uint64_t quarantined_retirement_count_;
  uint64_t quarantined_bytes_;
  uint64_t outstanding_buffer_count_;
};

/** Aggregate host-side snapshot for serving telemetry and memory admission control. */
struct RuntimeStatistics final {
  RuntimeStatus status_;
  size_t execution_context_count_;
  size_t captured_graph_count_;
  size_t active_capture_count_;
  std::vector<DeviceMemoryStatistics> devices_;
  PinnedMemoryStatistics pinned_memory_;
};

/**
 * @brief Description of an existing CUDA device allocation.
 *
 * A null owner means borrowed memory. A non-null owner is retained until all recorded stream usage has completed.
 */
struct ExternalMemory final {
  void *pointer_;
  size_t capacity_bytes_;
  Device device_;
  std::shared_ptr<void> owner_;
};

/**
 * @brief Process-local owner of the CUDA devices and memory services registered with TTL.
 *
 * Public methods may be called concurrently. Shutdown closes admission before checking child registrations, so a
 * racing resource-creation call either commits before that transition or fails and releases its reservation.
 */
class Runtime final {
 public:
  explicit Runtime(RuntimeOptions options, std::source_location location = std::source_location::current());
  Runtime(const Runtime &) = delete;
  auto operator=(const Runtime &) -> Runtime & = delete;
  Runtime(Runtime &&) = delete;
  auto operator=(Runtime &&) -> Runtime & = delete;
  ~Runtime() noexcept;

  [[nodiscard]] auto GetDevices() const noexcept -> std::span<const Device>;
  [[nodiscard]] auto GetDeviceProperties(Device device,
                                         std::source_location location = std::source_location::current()) const
      -> const DeviceProperties &;
  [[nodiscard]] auto CanAccessPeer(Device device, Device peer_device,
                                   std::source_location location = std::source_location::current()) const -> bool;
  [[nodiscard]] auto GetStatus() const noexcept -> RuntimeStatus;
  [[nodiscard]] auto GetStatistics(std::source_location location = std::source_location::current()) const
      -> RuntimeStatistics;

  /** @brief Create an execution context backed by a new non-default CUDA stream. */
  [[nodiscard]] auto CreateExecutionContext(Device device, const ExecutionContextOptions &options = {},
                                            std::source_location location = std::source_location::current())
      -> ExecutionContext;

  /**
   * @brief Create an execution context around a caller-owned non-default CUDA stream.
   *
   * A non-null `owner` is retained for the context lifetime and is required for CUDA Graph capture.
   */
  [[nodiscard]] auto WrapExternalStream(Device device, cudaStream_t stream, std::shared_ptr<void> owner = nullptr,
                                        const ExecutionContextOptions &options = {},
                                        std::source_location location = std::source_location::current())
      -> ExecutionContext;

  /**
   * @brief Wrap an existing CUDA device allocation in immutable tensor metadata.
   *
   * A null `memory.owner_` creates borrowed storage; otherwise the owner is retained through asynchronous retirement.
   */
  [[nodiscard]] auto FromBlob(ExecutionContext &context, ExternalMemory memory, const Shape &shape,
                              const Strides &strides, DType dtype, int64_t storage_offset = 0,
                              std::source_location location = std::source_location::current()) -> Tensor;

  /** @brief Allocate a page-locked host buffer from the runtime cache. */
  [[nodiscard]] auto AllocatePinned(size_t bytes, std::source_location location = std::source_location::current())
      -> PinnedBuffer;

  /** @brief Poll retirements and ask one CUDA memory pool to release cached pages down to the target. */
  void TrimMemory(Device device, size_t target_reserved_bytes,
                  std::source_location location = std::source_location::current());

  /** @brief Poll retirements and release every cached page-locked host allocation. */
  void TrimPinnedMemory(std::source_location location = std::source_location::current());

  /** @brief Advance nonblocking cleanup and asynchronous NCCL error polling without throwing. */
  void Poll() noexcept;

  /**
   * @brief Close all runtime services after callers have released contexts, graphs, tensors, and communicators.
   *
   * Shutdown is retryable after an ordering error or native cleanup failure; completed stages are not repeated.
   */
  void Shutdown(std::source_location location = std::source_location::current());

 private:
  friend class internal::RuntimeAccess;

  class Impl;

  std::unique_ptr<Impl> impl_;
};

static_assert(!std::copy_constructible<Runtime>);
static_assert(!std::move_constructible<Runtime>);

}  // namespace ttl
