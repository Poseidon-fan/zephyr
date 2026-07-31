#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <source_location>
#include <span>
#include <vector>

#include "ttl/device.hpp"
#include "ttl/device_properties.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/blas_handle_pool.hpp"
#include "ttl/internal/device_allocator.hpp"
#include "ttl/internal/event_pool.hpp"
#include "ttl/internal/matmul_plan.hpp"
#include "ttl/internal/pinned_allocator.hpp"
#include "ttl/runtime.hpp"

namespace ttl {

struct NcclOptions;

}  // namespace ttl

namespace ttl::internal {

class CommunicatorGroupState;

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

  void EnsureRunning(std::source_location location) const;
  void RegisterExecutionContext(std::source_location location);
  void UnregisterExecutionContext() noexcept;
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

  std::vector<Device> devices_;
  std::vector<std::shared_ptr<DeviceContext>> device_contexts_;
  std::vector<uint8_t> peer_access_;
  std::shared_ptr<ErrorSink> error_sink_;
  std::shared_ptr<PinnedAllocator> pinned_allocator_;
  std::vector<std::weak_ptr<CommunicatorGroupState>> communicator_groups_;
  std::source_location location_;

  mutable std::mutex lifecycle_latch_;
  std::atomic<RuntimeStatus> status_{RuntimeStatus::RUNNING};
  std::atomic<size_t> execution_context_count_{0};
};

class RuntimeAccess final {
 public:
  [[nodiscard]] static auto GetState(Runtime &runtime, std::source_location location)
      -> const std::shared_ptr<RuntimeState> &;
};

}  // namespace ttl::internal
