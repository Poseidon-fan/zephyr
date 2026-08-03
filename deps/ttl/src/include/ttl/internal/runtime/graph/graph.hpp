#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

#include "ttl/internal/runtime/execution/event_pool.hpp"
#include "ttl/runtime/device.hpp"

namespace ttl {

class ErrorSink;
class ExecutionContext;
struct GraphCaptureOptions;

}  // namespace ttl

namespace ttl::internal {

class CommunicatorGroupState;
class ExecutionContextImpl;
class RuntimeState;
class Storage;
class StreamState;

enum class CaptureStatus : uint8_t {
  ACTIVE,
  INVALIDATED,
  FINISHED,
  ABORTED,
};

/** Owner and bookkeeping record for one active stream-capture transaction. */
class CaptureSessionState final {
 public:
  [[nodiscard]] static auto Begin(ExecutionContext &context, const GraphCaptureOptions &options,
                                  std::source_location location) -> std::shared_ptr<CaptureSessionState>;

  CaptureSessionState(const CaptureSessionState &) = delete;
  auto operator=(const CaptureSessionState &) -> CaptureSessionState & = delete;
  CaptureSessionState(CaptureSessionState &&) = delete;
  auto operator=(CaptureSessionState &&) -> CaptureSessionState & = delete;

  ~CaptureSessionState() noexcept;

  [[nodiscard]] auto Finish(std::source_location location) -> std::unique_ptr<class CapturedGraphState>;
  void Abort() noexcept;
  void Invalidate() noexcept;

  [[nodiscard]] auto IsActive() const noexcept -> bool;
  [[nodiscard]] auto GetStatus() const noexcept -> CaptureStatus;
  [[nodiscard]] auto GetOperationCount() const noexcept -> uint64_t;

  void BeginOperation(std::string_view operation, std::source_location location);
  void RetainStorage(const std::shared_ptr<Storage> &storage, std::source_location location);
  void RetainCommunicator(const std::shared_ptr<CommunicatorGroupState> &communicator, std::source_location location);
  void RetainAuxiliaryStream(std::shared_ptr<StreamState> stream);

  [[nodiscard]] auto GetForkEvent() noexcept -> PooledEvent &;
  [[nodiscard]] auto GetJoinEvent(size_t index) noexcept -> PooledEvent &;

 private:
  CaptureSessionState(std::shared_ptr<ExecutionContextImpl> context, std::shared_ptr<RuntimeState> runtime_state,
                      std::shared_ptr<StreamState> primary_stream, std::vector<PooledEvent> dependency_events,
                      std::string name, std::source_location location) noexcept;

  void ClearContextRegistration(CaptureStatus final_status) noexcept;
  void ReportCleanupFailure(std::string_view message) noexcept;

  std::shared_ptr<ExecutionContextImpl> context_;
  std::shared_ptr<RuntimeState> runtime_state_;
  std::shared_ptr<StreamState> primary_stream_;
  std::vector<std::shared_ptr<StreamState>> auxiliary_streams_;
  std::vector<PooledEvent> dependency_events_;
  std::map<const Storage *, std::shared_ptr<Storage>> retained_storage_;
  std::map<const CommunicatorGroupState *, std::shared_ptr<CommunicatorGroupState>> retained_communicators_;
  std::string name_;
  std::source_location location_;
  std::atomic<CaptureStatus> status_{CaptureStatus::ACTIVE};
  std::atomic<uint64_t> operation_count_{0};
};

/** Native CUDA graph executable plus every owner required by its captured addresses and library nodes. */
class CapturedGraphState final {
 public:
  CapturedGraphState(Device device, uint64_t stream_id, cudaGraph_t graph, cudaGraphExec_t executable,
                     std::shared_ptr<StreamState> primary_stream,
                     std::vector<std::shared_ptr<StreamState>> auxiliary_streams,
                     std::vector<PooledEvent> dependency_events, std::vector<std::shared_ptr<Storage>> storage,
                     std::vector<std::shared_ptr<CommunicatorGroupState>> communicators,
                     std::shared_ptr<RuntimeState> runtime_state, std::string name, size_t node_count,
                     std::source_location location);

  CapturedGraphState(const CapturedGraphState &) = delete;
  auto operator=(const CapturedGraphState &) -> CapturedGraphState & = delete;
  CapturedGraphState(CapturedGraphState &&) = delete;
  auto operator=(CapturedGraphState &&) -> CapturedGraphState & = delete;
  ~CapturedGraphState() noexcept;

  void Launch(ExecutionContext &context, std::source_location location);
  void DebugDumpDot(std::string_view path, std::source_location location) const;

  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetStreamId() const noexcept -> uint64_t;
  [[nodiscard]] auto GetNodeCount() const noexcept -> size_t;
  [[nodiscard]] auto GetLaunchCount() const noexcept -> uint64_t;
  [[nodiscard]] auto GetName() const noexcept -> std::string_view;

 private:
  Device device_;
  uint64_t stream_id_;
  cudaGraph_t graph_;
  cudaGraphExec_t executable_;
  std::shared_ptr<StreamState> primary_stream_;
  std::vector<std::shared_ptr<StreamState>> auxiliary_streams_;
  std::vector<PooledEvent> dependency_events_;
  std::vector<std::shared_ptr<Storage>> storage_;
  std::vector<std::shared_ptr<CommunicatorGroupState>> communicators_;
  std::shared_ptr<RuntimeState> runtime_state_;
  std::string name_;
  size_t node_count_;
  std::source_location location_;
  std::atomic<uint64_t> launch_count_{0};
};

/** Return the active capture state or null when the context is executing eagerly. */
[[nodiscard]] auto GetCaptureState(ExecutionContext &context, std::source_location location)
    -> std::shared_ptr<CaptureSessionState>;

}  // namespace ttl::internal
