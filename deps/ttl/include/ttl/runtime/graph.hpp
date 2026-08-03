#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "ttl/runtime/device.hpp"
#include "ttl/runtime/execution_context.hpp"

namespace ttl::internal {

class CaptureSessionState;
class CapturedGraphState;

}  // namespace ttl::internal

namespace ttl {

struct GraphCaptureOptions final {
  std::string name_;
};

class CapturedGraph;

/**
 * Explicit owner of one active stream-capture transaction.
 *
 * Operations, Finish, and Abort must not run concurrently. The transaction may move between host threads because TTL
 * uses relaxed CUDA stream capture so a noexcept destructor can always terminate the native capture. Destruction
 * aborts an active session without throwing.
 */
class CaptureSession final {
 public:
  CaptureSession() = delete;
  CaptureSession(const CaptureSession &) = delete;
  auto operator=(const CaptureSession &) -> CaptureSession & = delete;
  CaptureSession(CaptureSession &&other) noexcept;
  auto operator=(CaptureSession &&other) noexcept -> CaptureSession &;
  ~CaptureSession() noexcept;

  [[nodiscard]] auto Finish(std::source_location location = std::source_location::current()) -> CapturedGraph;
  void Abort() noexcept;
  [[nodiscard]] auto IsActive() const noexcept -> bool;

 private:
  friend class ExecutionContext;

  explicit CaptureSession(std::shared_ptr<internal::CaptureSessionState> state) noexcept;

  std::shared_ptr<internal::CaptureSessionState> state_;
};

/** Immutable, fixed-address CUDA graph executable bound to its capture context stream. */
class CapturedGraph final {
 public:
  CapturedGraph() = delete;
  CapturedGraph(const CapturedGraph &) = delete;
  auto operator=(const CapturedGraph &) -> CapturedGraph & = delete;
  CapturedGraph(CapturedGraph &&) noexcept;
  auto operator=(CapturedGraph &&) noexcept -> CapturedGraph &;
  ~CapturedGraph() noexcept;

  void Launch(ExecutionContext &context, std::source_location location = std::source_location::current());
  void DebugDumpDot(std::string_view path, std::source_location location = std::source_location::current()) const;

  [[nodiscard]] auto GetDevice(std::source_location location = std::source_location::current()) const -> Device;
  [[nodiscard]] auto GetStreamId(std::source_location location = std::source_location::current()) const -> uint64_t;
  [[nodiscard]] auto GetNodeCount(std::source_location location = std::source_location::current()) const -> size_t;
  [[nodiscard]] auto GetLaunchCount(std::source_location location = std::source_location::current()) const -> uint64_t;
  [[nodiscard]] auto GetName(std::source_location location = std::source_location::current()) const -> std::string_view;

 private:
  friend class CaptureSession;

  explicit CapturedGraph(std::unique_ptr<internal::CapturedGraphState> state) noexcept;

  std::unique_ptr<internal::CapturedGraphState> state_;
};

using GraphCaptureFunction = std::function<void(size_t, ExecutionContext &)>;

struct GraphGroupCaptureOptions final {
  std::string name_;
};

/**
 * Process-local multi-GPU graph owner.
 *
 * Capture takes ownership of one warmed-up context per rank. A fixed worker thread per rank performs capture and every
 * replay so NCCL graph launches are never issued sequentially by one host thread.
 */
class CapturedGraphGroup final {
 public:
  [[nodiscard]] static auto Capture(std::vector<ExecutionContext> contexts, GraphCaptureFunction capture_function,
                                    const GraphGroupCaptureOptions &options = {},
                                    std::source_location location = std::source_location::current())
      -> CapturedGraphGroup;

  CapturedGraphGroup() = delete;
  CapturedGraphGroup(const CapturedGraphGroup &) = delete;
  auto operator=(const CapturedGraphGroup &) -> CapturedGraphGroup & = delete;
  CapturedGraphGroup(CapturedGraphGroup &&) noexcept;
  auto operator=(CapturedGraphGroup &&) noexcept -> CapturedGraphGroup &;
  ~CapturedGraphGroup() noexcept;

  void Launch(std::source_location location = std::source_location::current());
  void Synchronize(std::source_location location = std::source_location::current());

  [[nodiscard]] auto GetWorldSize(std::source_location location = std::source_location::current()) const -> size_t;
  [[nodiscard]] auto GetContext(size_t rank, std::source_location location = std::source_location::current())
      -> ExecutionContext &;
  [[nodiscard]] auto GetDevice(size_t rank, std::source_location location = std::source_location::current()) const
      -> Device;
  [[nodiscard]] auto GetNodeCount(size_t rank, std::source_location location = std::source_location::current()) const
      -> size_t;
  [[nodiscard]] auto GetLaunchCount(std::source_location location = std::source_location::current()) const -> uint64_t;
  [[nodiscard]] auto GetName(std::source_location location = std::source_location::current()) const -> std::string_view;

 private:
  class Impl;

  explicit CapturedGraphGroup(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

static_assert(!std::default_initializable<CaptureSession>);
static_assert(!std::copy_constructible<CaptureSession>);
static_assert(std::is_nothrow_move_constructible_v<CaptureSession>);
static_assert(!std::default_initializable<CapturedGraph>);
static_assert(!std::copy_constructible<CapturedGraph>);
static_assert(std::is_nothrow_move_constructible_v<CapturedGraph>);
static_assert(!std::default_initializable<CapturedGraphGroup>);
static_assert(!std::copy_constructible<CapturedGraphGroup>);
static_assert(std::is_nothrow_move_constructible_v<CapturedGraphGroup>);

}  // namespace ttl
