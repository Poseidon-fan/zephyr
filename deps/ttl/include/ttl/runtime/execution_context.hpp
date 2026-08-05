#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <type_traits>

#include "ttl/common/device.hpp"
#include "ttl/runtime/event.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {

class ContextAccess;
class ExecutionContextImpl;

}  // namespace ttl::internal

namespace ttl {

class CaptureSession;
struct GraphCaptureOptions;

/** @brief Configures the primary stream and structured auxiliary lanes owned by an execution context. */
struct ExecutionContextOptions final {
  /** CUDA stream priority applied to the primary and auxiliary streams. */
  int32_t stream_priority_{0};
  /** Maximum auxiliary lanes available to checked multi-stream submissions. */
  size_t max_auxiliary_stream_count_{0};
};

/**
 * Move-only execution lane bound to one CUDA device and one non-default stream.
 *
 * Work submitted through one context is ordered by its stream. Different contexts have no implicit data dependency;
 * callers must connect them with RecordEvent and Wait. A context may be used by only one host thread at a time.
 */
class ExecutionContext final {
 public:
  ExecutionContext() = delete;
  ExecutionContext(const ExecutionContext &) = delete;
  auto operator=(const ExecutionContext &) -> ExecutionContext & = delete;
  ExecutionContext(ExecutionContext &&) noexcept = default;
  auto operator=(ExecutionContext &&) noexcept -> ExecutionContext & = default;
  ~ExecutionContext() noexcept = default;

  [[nodiscard]] auto GetDevice(std::source_location location = std::source_location::current()) const -> Device;
  [[nodiscard]] auto GetStream(std::source_location location = std::source_location::current()) const -> const Stream &;
  [[nodiscard]] auto GetAuxiliaryStreamCount(std::source_location location = std::source_location::current()) const
      -> size_t;
  [[nodiscard]] auto IsExternalStream(std::source_location location = std::source_location::current()) const -> bool;

  /** @brief Record an event after all work currently enqueued on the primary stream. */
  [[nodiscard]] auto RecordEvent(std::source_location location = std::source_location::current()) -> Event;

  /** @brief Make the primary stream wait for `event` without blocking the host. */
  void Wait(const Event &event, std::source_location location = std::source_location::current());

  /** @brief Begin relaxed CUDA stream capture on the primary stream. */
  [[nodiscard]] auto BeginCapture(std::source_location location = std::source_location::current()) -> CaptureSession;

  /** @brief Begin relaxed CUDA stream capture using the supplied graph metadata. */
  [[nodiscard]] auto BeginCapture(const GraphCaptureOptions &options,
                                  std::source_location location = std::source_location::current()) -> CaptureSession;

  /** @brief Synchronize submitted work and surface the first pending device-side semantic error. */
  void CheckAsyncErrors(std::source_location location = std::source_location::current());

  /** @brief Synchronize submitted work, check asynchronous errors, and reclaim completed runtime resources. */
  void Synchronize(std::source_location location = std::source_location::current());

  /** @brief Reclaim completed resources without synchronizing GPU work. */
  void Poll(std::source_location location = std::source_location::current());

 private:
  friend class internal::ContextAccess;

  explicit ExecutionContext(std::shared_ptr<internal::ExecutionContextImpl> impl) noexcept;

  std::shared_ptr<internal::ExecutionContextImpl> impl_;
};

static_assert(!std::default_initializable<ExecutionContext>);
static_assert(!std::copy_constructible<ExecutionContext>);
static_assert(std::is_nothrow_move_constructible_v<ExecutionContext>);
static_assert(std::is_nothrow_move_assignable_v<ExecutionContext>);

}  // namespace ttl
