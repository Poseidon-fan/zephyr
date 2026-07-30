#pragma once

#include <atomic>
#include <cstddef>
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
#include "ttl/runtime.hpp"

namespace ttl::internal {

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

 private:
  DeviceProperties properties_;
  std::shared_ptr<EventPool> event_pool_;
  std::shared_ptr<DeviceAllocator> allocator_;
  std::shared_ptr<BlasHandlePool> blas_handle_pool_;
};

class RuntimeState final {
 public:
  RuntimeState(RuntimeOptions options, std::source_location location);

  RuntimeState(const RuntimeState &) = delete;
  auto operator=(const RuntimeState &) -> RuntimeState & = delete;
  RuntimeState(RuntimeState &&) = delete;
  auto operator=(RuntimeState &&) -> RuntimeState & = delete;

  [[nodiscard]] auto GetDevices() const noexcept -> std::span<const Device>;
  [[nodiscard]] auto GetDeviceContext(Device device, std::source_location location) const
      -> const std::shared_ptr<DeviceContext> &;
  [[nodiscard]] auto CanAccessPeer(Device source, Device destination, std::source_location location) const -> bool;
  [[nodiscard]] auto GetErrorSink() const noexcept -> const std::shared_ptr<ErrorSink> &;
  [[nodiscard]] auto GetStatus() const noexcept -> RuntimeStatus;

  void EnsureRunning(std::source_location location) const;
  void RegisterExecutionContext(std::source_location location);
  void UnregisterExecutionContext() noexcept;
  void TrimMemory(Device device, size_t target_reserved_bytes, std::source_location location);
  void Poll() noexcept;
  void Shutdown(std::source_location location);
  void Abandon() noexcept;

 private:
  [[nodiscard]] auto FindDeviceIndex(Device device, std::source_location location) const -> size_t;

  std::vector<Device> devices_;
  std::vector<std::shared_ptr<DeviceContext>> device_contexts_;
  std::vector<uint8_t> peer_access_;
  std::shared_ptr<ErrorSink> error_sink_;
  std::source_location location_;

  mutable std::mutex lifecycle_latch_;
  std::atomic<RuntimeStatus> status_{RuntimeStatus::RUNNING};
  std::atomic<size_t> execution_context_count_{0};
};

}  // namespace ttl::internal
