#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <source_location>
#include <span>
#include <vector>

#include "ttl/common/device.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/ops/matmul_plan.hpp"
#include "ttl/internal/runtime/execution/event_pool.hpp"
#include "ttl/internal/runtime/library/blas_handle_pool.hpp"
#include "ttl/internal/runtime/memory/device/allocator.hpp"
#include "ttl/internal/runtime/memory/pinned/allocator.hpp"
#include "ttl/runtime/device_properties.hpp"
#include "ttl/runtime/runtime.hpp"

namespace ttl {

struct NcclOptions;

}  // namespace ttl

namespace ttl::internal {

class CommunicatorGroupState;
class CaptureSessionState;
class ExecutionContextRegistration;
class GraphCleanupState;

/**
 * @brief Runtime-owned services and immutable properties for one registered CUDA device.
 *
 * The event pool, allocator, and cuBLAS handle pool are shared with resources that may finish asynchronously. Runtime
 * shutdown closes them in dependency order; DeviceContext itself only groups their ownership and the device-local
 * matmul algorithm cache.
 */
class DeviceContext final {
 public:
  DeviceContext(DeviceProperties properties, std::shared_ptr<EventPool> event_pool,
                std::shared_ptr<DeviceAllocator> allocator, std::shared_ptr<BlasHandlePool> blas_handle_pool) noexcept;

  DeviceContext(const DeviceContext &) = delete;
  auto operator=(const DeviceContext &) -> DeviceContext & = delete;
  DeviceContext(DeviceContext &&) = delete;
  auto operator=(DeviceContext &&) -> DeviceContext & = delete;

  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetProperties() const noexcept -> const DeviceProperties &;
  [[nodiscard]] auto GetEventPool() const noexcept -> const std::shared_ptr<EventPool> &;
  [[nodiscard]] auto GetAllocator() const noexcept -> const std::shared_ptr<DeviceAllocator> &;
  [[nodiscard]] auto GetBlasHandlePool() const noexcept -> const std::shared_ptr<BlasHandlePool> &;
  [[nodiscard]] auto GetMatmulAlgorithmCache() noexcept -> MatmulAlgorithmCache &;

 private:
  DeviceProperties properties_;
  std::shared_ptr<EventPool> event_pool_;
  std::shared_ptr<DeviceAllocator> allocator_;
  std::shared_ptr<BlasHandlePool> blas_handle_pool_;
  MatmulAlgorithmCache matmul_algorithm_cache_;
};

/**
 * @brief Shared implementation of the process-local Runtime and its lifecycle state machine.
 *
 * Execution contexts, graph captures, and communicators retain this object while they use runtime services. The
 * lifecycle latch makes registration atomic with respect to shutdown, while the counters prevent shutdown from
 * invalidating live execution resources. Poll performs non-throwing deferred progress. Shutdown is ordered and
 * retryable: a failed component remains open and a later call resumes from the first incomplete stage.
 */
class RuntimeState final : public std::enable_shared_from_this<RuntimeState> {
 public:
  RuntimeState(RuntimeOptions options, std::source_location location);

  RuntimeState(const RuntimeState &) = delete;
  auto operator=(const RuntimeState &) -> RuntimeState & = delete;
  RuntimeState(RuntimeState &&) = delete;
  auto operator=(RuntimeState &&) -> RuntimeState & = delete;

  [[nodiscard]] auto GetDevices() const noexcept -> std::span<const Device>;
  [[nodiscard]] auto GetDeviceContext(Device device, std::source_location location) const
      -> const std::shared_ptr<DeviceContext> &;
  [[nodiscard]] auto CanAccessPeer(Device device, Device peer_device, std::source_location location) const -> bool;
  [[nodiscard]] auto GetErrorSink() const noexcept -> const std::shared_ptr<ErrorSink> &;
  [[nodiscard]] auto GetPinnedAllocator() const noexcept -> const std::shared_ptr<PinnedAllocator> &;
  [[nodiscard]] auto GetStatus() const noexcept -> RuntimeStatus;
  [[nodiscard]] auto GetStatistics(std::source_location location) const -> RuntimeStatistics;

  void EnsureRunning(std::source_location location) const;
  [[nodiscard]] auto BeginExecutionContextCreation(std::source_location location) -> ExecutionContextRegistration;
  void UnregisterExecutionContext() noexcept;
  void RegisterGraph(std::source_location location);
  void UnregisterGraph() noexcept;
  void BeginCapture(std::source_location location);
  void EndCapture() noexcept;
  void TrackCaptureSession(const std::shared_ptr<CaptureSessionState> &state, std::source_location location);
  void EnqueueGraphCleanup(GraphCleanupState *state) noexcept;
  [[nodiscard]] auto HasActiveCapture() const noexcept -> bool;
  [[nodiscard]] auto CreateCommunicatorGroup(std::span<const Device> rank_order, const NcclOptions &options,
                                             std::source_location location) -> std::shared_ptr<CommunicatorGroupState>;
  [[nodiscard]] auto AllocatePinned(size_t bytes, std::source_location location) -> PinnedBuffer;
  void TrimMemory(Device device, size_t target_reserved_bytes, std::source_location location);
  void TrimPinnedMemory(std::source_location location);
  void Poll() noexcept;
  void Shutdown(std::source_location location);
  void Abandon() noexcept;

 private:
  [[nodiscard]] auto FindDeviceIndex(Device device, std::source_location location) const -> size_t;
  [[nodiscard]] auto HasOpenCommunicatorGroups() noexcept -> bool;
  void DrainGraphCleanups(std::source_location location);
  void PollGraphCleanupsNoexcept() noexcept;

  std::vector<Device> devices_;
  std::vector<std::shared_ptr<DeviceContext>> device_contexts_;
  // Row-major [accessing device][peer device] capability matrix.
  std::vector<uint8_t> peer_access_;
  std::shared_ptr<ErrorSink> error_sink_;
  std::shared_ptr<PinnedAllocator> pinned_allocator_;
  std::vector<std::weak_ptr<CommunicatorGroupState>> communicator_groups_;
  std::vector<std::weak_ptr<CaptureSessionState>> capture_sessions_;
  // Intrusive ownership list populated by noexcept graph destructors and drained by Poll or Shutdown.
  GraphCleanupState *pending_graph_cleanup_head_{nullptr};
  std::source_location location_;

  // Shutdown progress is retained after a component failure so a retry only resumes at the failed component.
  std::vector<uint8_t> blas_shutdown_;
  std::vector<uint8_t> allocator_shutdown_;
  std::vector<uint8_t> event_pool_shutdown_;
  bool pinned_allocator_shutdown_{false};

  // Serializes lifecycle transitions with creation, capture, graph, and communicator registration.
  mutable std::mutex lifecycle_latch_;
  std::mutex graph_cleanup_latch_;
  std::atomic<RuntimeStatus> status_{RuntimeStatus::RUNNING};
  std::atomic<size_t> execution_context_count_{0};
  std::atomic<size_t> graph_count_{0};
  std::atomic<size_t> active_capture_count_{0};
};

class RuntimeAccess final {
 public:
  [[nodiscard]] static auto GetState(Runtime &runtime, std::source_location location)
      -> const std::shared_ptr<RuntimeState> &;
};

}  // namespace ttl::internal
