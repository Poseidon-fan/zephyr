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

#include "ttl/device.hpp"
#include "ttl/device_properties.hpp"
#include "ttl/dtype.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/pinned_buffer.hpp"
#include "ttl/shape.hpp"

namespace ttl {

class ErrorSink;
class Tensor;

namespace internal {

class RuntimeAccess;

}  // namespace internal

struct DeviceMemoryOptions final {
  uint64_t release_threshold_bytes_{std::numeric_limits<uint64_t>::max()};
  uint64_t max_live_bytes_{0};
  uint64_t max_reserved_bytes_{0};
  bool enable_maintenance_thread_{true};
};

struct PinnedMemoryOptions final {
  size_t max_cached_bytes_{256U * 1024U * 1024U};
  size_t max_live_bytes_{512U * 1024U * 1024U};
};

struct RuntimeOptions final {
  std::vector<Device> devices_;
  DeviceMemoryOptions device_memory_;
  size_t blas_workspace_bytes_{0};
  size_t event_pool_capacity_per_device_{256};
  size_t event_pool_reserve_per_device_{0};
  std::shared_ptr<ErrorSink> error_sink_;
  PinnedMemoryOptions pinned_memory_;
};

enum class RuntimeStatus : uint8_t {
  RUNNING,
  CLOSING,
  CLOSED,
};

/**
 * Description of an existing CUDA device allocation.
 *
 * A null owner means borrowed memory. A non-null owner is retained until all recorded stream usage has completed.
 */
struct ExternalMemory final {
  void *pointer_;
  size_t capacity_bytes_;
  Device device_;
  std::shared_ptr<void> owner_;
};

/** Process-local owner of the CUDA devices and memory services registered with TTL. */
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

  [[nodiscard]] auto CreateExecutionContext(Device device, const ExecutionContextOptions &options = {},
                                            std::source_location location = std::source_location::current())
      -> ExecutionContext;
  [[nodiscard]] auto WrapExternalStream(Device device, cudaStream_t stream, std::shared_ptr<void> owner = nullptr,
                                        const ExecutionContextOptions &options = {},
                                        std::source_location location = std::source_location::current())
      -> ExecutionContext;

  [[nodiscard]] auto FromBlob(ExecutionContext &context, ExternalMemory memory, const Shape &shape,
                              const Strides &strides, DType dtype, int64_t storage_offset = 0,
                              std::source_location location = std::source_location::current()) -> Tensor;

  [[nodiscard]] auto AllocatePinned(size_t bytes, std::source_location location = std::source_location::current())
      -> PinnedBuffer;
  void TrimMemory(Device device, size_t target_reserved_bytes,
                  std::source_location location = std::source_location::current());
  void TrimPinnedMemory(std::source_location location = std::source_location::current());
  void Poll() noexcept;
  void Shutdown(std::source_location location = std::source_location::current());

 private:
  friend class internal::RuntimeAccess;

  class Impl;

  std::unique_ptr<Impl> impl_;
};

static_assert(!std::copy_constructible<Runtime>);
static_assert(!std::move_constructible<Runtime>);

}  // namespace ttl
