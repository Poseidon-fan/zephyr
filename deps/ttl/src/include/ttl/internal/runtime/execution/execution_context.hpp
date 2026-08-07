#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <vector>

#include <driver_types.h>

#include "ttl/common/device.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/execution/event_pool.hpp"
#include "ttl/internal/runtime/execution/execution_lane.hpp"
#include "ttl/internal/runtime/memory/device/allocator.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl {

struct DeviceProperties;
class PinnedBuffer;
class Runtime;

}  // namespace ttl

namespace ttl::internal {

class DeviceContext;
class DeviceErrorState;
class CaptureSessionState;
class MatmulAlgorithmCache;
class PinnedAllocator;
class RuntimeState;

enum class ExecutionContextStatus : uint8_t {
  READY,
  CAPTURING,
  FAILED,
};

enum class ContextUseMode : uint8_t {
  SUBMIT,
  CLEANUP,
};

/**
 * @brief Move-only reservation that prevents Runtime shutdown while an ExecutionContext is being constructed or alive.
 *
 * Creation acquires the reservation before constructing native resources. Any construction failure releases it via
 * RAII; a successful construction moves it into ExecutionContextState for the complete context lifetime.
 */
class ExecutionContextRegistration final {
 public:
  ExecutionContextRegistration(const ExecutionContextRegistration &) = delete;
  auto operator=(const ExecutionContextRegistration &) -> ExecutionContextRegistration & = delete;
  ExecutionContextRegistration(ExecutionContextRegistration &&other) noexcept;
  auto operator=(ExecutionContextRegistration &&) -> ExecutionContextRegistration & = delete;
  ~ExecutionContextRegistration() noexcept;

 private:
  friend class RuntimeState;

  explicit ExecutionContextRegistration(std::shared_ptr<RuntimeState> runtime_state) noexcept;

  std::shared_ptr<RuntimeState> runtime_state_;
};

/** Shared lifetime state retained by execution contexts and active CUDA graph captures. */
class ExecutionContextState final {
 public:
  ExecutionContextState(std::shared_ptr<RuntimeState> runtime_state, ExecutionContextRegistration registration,
                        std::shared_ptr<DeviceContext> device_context, ExecutionLane primary_lane,
                        std::vector<ExecutionLane> auxiliary_lanes, std::optional<PooledEvent> fork_event,
                        std::vector<PooledEvent> join_events,
                        std::unique_ptr<DeviceErrorState> device_error_state) noexcept;

  ExecutionContextState(const ExecutionContextState &) = delete;
  auto operator=(const ExecutionContextState &) -> ExecutionContextState & = delete;
  ExecutionContextState(ExecutionContextState &&) = delete;
  auto operator=(ExecutionContextState &&) -> ExecutionContextState & = delete;

  ~ExecutionContextState() noexcept;

  std::shared_ptr<RuntimeState> runtime_state_;
  ExecutionContextRegistration registration_;
  std::shared_ptr<DeviceContext> device_context_;
  ExecutionLane primary_lane_;
  std::vector<ExecutionLane> auxiliary_lanes_;
  std::optional<PooledEvent> fork_event_;
  std::vector<PooledEvent> join_events_;
  std::unique_ptr<DeviceErrorState> device_error_state_;
  std::weak_ptr<CaptureSessionState> capture_state_;
  std::atomic_flag in_use_ = ATOMIC_FLAG_INIT;
  std::atomic<ExecutionContextStatus> status_{ExecutionContextStatus::READY};
};

/** Reject concurrent host use of one ExecutionContext without silently serializing it. */
class ContextUseGuard final {
 public:
  ContextUseGuard(ExecutionContext &context, ContextUseMode mode, std::source_location location);

  ContextUseGuard(const ContextUseGuard &) = delete;
  auto operator=(const ContextUseGuard &) -> ContextUseGuard & = delete;
  ContextUseGuard(ContextUseGuard &&) = delete;
  auto operator=(ContextUseGuard &&) -> ContextUseGuard & = delete;

  ~ContextUseGuard() noexcept;

 private:
  ExecutionContextState &ctx_state_;
};

/** Private construction and resource gateway for Runtime and operator implementations. */
class ContextAccess final {
 public:
  [[nodiscard]] static auto Create(const std::shared_ptr<RuntimeState> &runtime_state,
                                   ExecutionContextRegistration registration,
                                   std::shared_ptr<DeviceContext> device_context, Stream stream,
                                   const ExecutionContextOptions &options, std::source_location location)
      -> ExecutionContext;
  [[nodiscard]] static auto GetState(ExecutionContext &context, std::source_location location)
      -> ExecutionContextState &;
  [[nodiscard]] static auto GetStateOwner(ExecutionContext &context, std::source_location location)
      -> const std::shared_ptr<ExecutionContextState> &;
  [[nodiscard]] static auto GetRuntimeState(ExecutionContext &context, std::source_location location)
      -> const std::shared_ptr<RuntimeState> &;
  [[nodiscard]] static auto GetDeviceContext(ExecutionContext &context, std::source_location location)
      -> const std::shared_ptr<DeviceContext> &;
  [[nodiscard]] static auto GetAllocator(ExecutionContext &context, std::source_location location)
      -> const std::shared_ptr<DeviceAllocator> &;
  [[nodiscard]] static auto GetDeviceProperties(ExecutionContext &context, std::source_location location)
      -> const DeviceProperties &;
  [[nodiscard]] static auto CanAccessPeer(ExecutionContext &context, Device peer_device, std::source_location location)
      -> bool;
  [[nodiscard]] static auto GetErrorSink(ExecutionContext &context, std::source_location location)
      -> const std::shared_ptr<ErrorSink> &;
  [[nodiscard]] static auto GetPinnedAllocator(ExecutionContext &context, std::source_location location)
      -> const std::shared_ptr<PinnedAllocator> &;
  [[nodiscard]] static auto AllocatePinned(ExecutionContext &context, size_t bytes, std::source_location location)
      -> PinnedBuffer;
  [[nodiscard]] static auto GetStream(ExecutionContext &context, std::source_location location) -> const Stream &;
  [[nodiscard]] static auto GetNativeStream(ExecutionContext &context, std::source_location location) -> cudaStream_t;
  [[nodiscard]] static auto GetPrimaryLane(ExecutionContext &context, std::source_location location) -> ExecutionLane &;
  [[nodiscard]] static auto GetDeviceErrorState(ExecutionContext &context, std::source_location location)
      -> DeviceErrorState &;
  [[nodiscard]] static auto GetMatmulAlgorithmCache(ExecutionContext &context, std::source_location location)
      -> MatmulAlgorithmCache &;
};

}  // namespace ttl::internal
