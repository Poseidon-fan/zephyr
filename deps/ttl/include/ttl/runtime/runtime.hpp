#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <span>
#include <vector>

#include <driver_types.h>

#include "ttl/common/device.hpp"
#include "ttl/runtime/device_properties.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/memory.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl {

class ErrorSink;
class Tensor;

namespace internal {

class RuntimeAccess;
class RuntimeState;

}  // namespace internal

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
  /** @brief Query the registered device's CUDA memory without synchronizing or trimming its pool. */
  [[nodiscard]] auto GetDeviceMemoryInfo(Device device,
                                         std::source_location location = std::source_location::current()) const
      -> DeviceMemoryInfo;
  [[nodiscard]] auto CanAccessPeer(Device device, Device peer_device,
                                   std::source_location location = std::source_location::current()) const -> bool;
  [[nodiscard]] auto GetStatus() const noexcept -> RuntimeStatus;
  [[nodiscard]] auto GetStatistics(std::source_location location = std::source_location::current()) const
      -> RuntimeStatistics;

  /**
   * @brief Start a device-wide peak window at the allocator's currently charged live and retiring capacity.
   *
   * The window covers every context sharing this runtime/device. Callers coordinate measurement boundaries; this
   * method neither synchronizes streams nor polls retirements. Read the peak with GetStatistics().
   */
  void ResetPeakMemoryStatistics(Device device, std::source_location location = std::source_location::current());

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
  [[nodiscard]] auto FromBlob(ExecutionContext &context, ExternalDeviceMemory memory, const Shape &shape,
                              const Strides &strides, DType dtype, int64_t storage_offset = 0,
                              std::source_location location = std::source_location::current()) -> Tensor;

  /** @brief Allocate a page-locked host buffer from the runtime cache. */
  [[nodiscard]] auto AllocatePinned(size_t bytes, std::source_location location = std::source_location::current())
      -> PinnedBuffer;

  /**
   * @brief Wait for released device storage to finish retiring, without trimming cached pool pages.
   *
   * Callers must synchronize the contexts that used the storage, finish releasing its owners, and exclude new
   * retirements on this device until this returns. Live storage is not synchronized. A failed allocator or retirement
   * wait is reported to the caller.
   */
  void SynchronizeMemory(Device device, std::source_location location = std::source_location::current());

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

  std::shared_ptr<internal::RuntimeState> state_;
};

static_assert(!std::copy_constructible<Runtime>);
static_assert(!std::move_constructible<Runtime>);

}  // namespace ttl
