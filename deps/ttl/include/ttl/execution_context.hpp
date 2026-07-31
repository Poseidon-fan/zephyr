#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <type_traits>

#include "ttl/device.hpp"
#include "ttl/event.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {

class ContextAccess;
class ExecutionContextImpl;

}  // namespace ttl::internal

namespace ttl {

class CaptureSession;
struct GraphCaptureOptions;

struct ExecutionContextOptions final {
  int32_t stream_priority_{0};
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
  ExecutionContext(ExecutionContext &&) noexcept;
  auto operator=(ExecutionContext &&) noexcept -> ExecutionContext &;
  ~ExecutionContext() noexcept;

  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetStream() const noexcept -> const Stream &;
  [[nodiscard]] auto GetAuxiliaryStreamCount() const noexcept -> size_t;
  [[nodiscard]] auto IsExternalStream() const noexcept -> bool;

  [[nodiscard]] auto RecordEvent(std::source_location location = std::source_location::current()) -> Event;
  void Wait(const Event &event, std::source_location location = std::source_location::current());

  [[nodiscard]] auto BeginCapture(std::source_location location = std::source_location::current()) -> CaptureSession;
  [[nodiscard]] auto BeginCapture(const GraphCaptureOptions &options,
                                  std::source_location location = std::source_location::current()) -> CaptureSession;

  /** Synchronize submitted work and surface the first pending device-side semantic error. */
  void CheckAsyncErrors(std::source_location location = std::source_location::current());
  void Synchronize(std::source_location location = std::source_location::current());
  void Poll(std::source_location location = std::source_location::current());

 private:
  friend class internal::ContextAccess;

  explicit ExecutionContext(std::unique_ptr<internal::ExecutionContextImpl> impl) noexcept;

  std::unique_ptr<internal::ExecutionContextImpl> impl_;
};

static_assert(!std::default_initializable<ExecutionContext>);
static_assert(!std::copy_constructible<ExecutionContext>);
static_assert(std::is_nothrow_move_constructible_v<ExecutionContext>);
static_assert(std::is_nothrow_move_assignable_v<ExecutionContext>);

}  // namespace ttl
