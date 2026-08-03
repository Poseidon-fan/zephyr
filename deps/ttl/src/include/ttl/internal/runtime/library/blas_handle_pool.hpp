#pragma once

#include <cstddef>
#include <memory>
#include <source_location>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "ttl/runtime/device.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl {

class ErrorSink;

}  // namespace ttl

namespace ttl::internal {

class BlasHandlePoolState;
class DeviceAllocator;
class EventPool;
class Storage;
class StreamState;

/**
 * Move-only lease of one stream-bound cuBLAS handle and its private device workspace.
 *
 * Returning a lease records completion on its stream. The resource is not reused until that event completes.
 */
class BlasHandleLease final {
 public:
  BlasHandleLease(const BlasHandleLease &) = delete;
  auto operator=(const BlasHandleLease &) -> BlasHandleLease & = delete;

  BlasHandleLease(BlasHandleLease &&other) noexcept;
  auto operator=(BlasHandleLease &&other) noexcept -> BlasHandleLease &;

  ~BlasHandleLease() noexcept;

  [[nodiscard]] auto GetCublasHandle() const noexcept -> cublasHandle_t;
  [[nodiscard]] auto GetCublasLtHandle() const noexcept -> cublasLtHandle_t;
  [[nodiscard]] auto GetWorkspace() const noexcept -> Storage &;
  [[nodiscard]] auto GetWorkspaceStorage() const noexcept -> const std::shared_ptr<Storage> &;

 private:
  friend class BlasHandlePoolState;

  BlasHandleLease(cublasHandle_t handle, cublasLtHandle_t lt_handle, std::shared_ptr<Storage> workspace,
                  std::shared_ptr<StreamState> stream, std::shared_ptr<BlasHandlePoolState> pool) noexcept;
  void Reset() noexcept;

  cublasHandle_t handle_;
  cublasLtHandle_t lt_handle_;
  std::shared_ptr<Storage> workspace_;
  std::shared_ptr<StreamState> stream_;
  std::shared_ptr<BlasHandlePoolState> pool_;
};

/**
 * Per-device pool of cuBLAS handles and fixed workspaces.
 *
 * Acquire is thread-safe. Shutdown is an explicit blocking lifecycle boundary and must run before allocator shutdown.
 */
class BlasHandlePool final {
 public:
  BlasHandlePool(Device device, size_t workspace_bytes, std::shared_ptr<ErrorSink> error_sink,
                 std::shared_ptr<EventPool> event_pool, std::shared_ptr<DeviceAllocator> allocator,
                 std::source_location location = std::source_location::current());

  BlasHandlePool(const BlasHandlePool &) = delete;
  auto operator=(const BlasHandlePool &) -> BlasHandlePool & = delete;
  BlasHandlePool(BlasHandlePool &&) = delete;
  auto operator=(BlasHandlePool &&) -> BlasHandlePool & = delete;

  ~BlasHandlePool() noexcept;

  [[nodiscard]] auto Acquire(const Stream &stream, std::source_location location = std::source_location::current())
      -> BlasHandleLease;

  void Poll() noexcept;
  void Shutdown(std::source_location location = std::source_location::current());

  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetWorkspaceBytes() const noexcept -> size_t;

 private:
  std::shared_ptr<BlasHandlePoolState> state_;
};

}  // namespace ttl::internal
