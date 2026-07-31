#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <source_location>
#include <span>
#include <string_view>
#include <vector>

#include <nccl.h>

#include "ttl/communicator.hpp"
#include "ttl/device.hpp"
#include "ttl/internal/event_pool.hpp"
#include "ttl/stream.hpp"

namespace ttl {

class ErrorSink;

}  // namespace ttl

namespace ttl::internal {

class RuntimeState;
class Storage;

class CommunicatorOperationLease final {
 public:
  CommunicatorOperationLease(const CommunicatorOperationLease &) = delete;
  auto operator=(const CommunicatorOperationLease &) -> CommunicatorOperationLease & = delete;
  CommunicatorOperationLease(CommunicatorOperationLease &&other) noexcept;
  auto operator=(CommunicatorOperationLease &&) -> CommunicatorOperationLease & = delete;
  ~CommunicatorOperationLease() noexcept;

  [[nodiscard]] auto GetHandle(size_t rank) const noexcept -> ncclComm_t;
  [[nodiscard]] auto GetRanks() const noexcept -> std::span<const size_t>;

 private:
  friend class CommunicatorGroupState;

  CommunicatorOperationLease(std::shared_ptr<CommunicatorGroupState> state, std::vector<size_t> ranks) noexcept;

  std::shared_ptr<CommunicatorGroupState> state_;
  std::vector<size_t> ranks_;
};

/** Shared process-local state behind one LocalCommunicatorGroup and its rank facades. */
class CommunicatorGroupState final : public std::enable_shared_from_this<CommunicatorGroupState> {
 public:
  [[nodiscard]] static auto Create(const std::shared_ptr<RuntimeState> &runtime_state,
                                   std::span<const Device> rank_order, const NcclOptions &options,
                                   std::source_location location) -> std::shared_ptr<CommunicatorGroupState>;

  CommunicatorGroupState(const CommunicatorGroupState &) = delete;
  auto operator=(const CommunicatorGroupState &) -> CommunicatorGroupState & = delete;
  CommunicatorGroupState(CommunicatorGroupState &&) = delete;
  auto operator=(CommunicatorGroupState &&) -> CommunicatorGroupState & = delete;
  ~CommunicatorGroupState() noexcept;

  [[nodiscard]] auto GetWorldSize() const noexcept -> size_t;
  [[nodiscard]] auto GetDevice(size_t rank) const noexcept -> Device;
  [[nodiscard]] auto GetRankOrder() const noexcept -> std::span<const Device>;
  [[nodiscard]] auto GetStatus() const noexcept -> CommunicatorStatus;
  [[nodiscard]] auto BelongsTo(const std::shared_ptr<RuntimeState> &runtime_state) const noexcept -> bool;
  [[nodiscard]] auto HasNativeResources() const noexcept -> bool;
  [[nodiscard]] auto HasGraphReferences() const noexcept -> bool;

  [[nodiscard]] auto AcquireRank(size_t rank, std::source_location location) -> CommunicatorOperationLease;
  [[nodiscard]] auto AcquireAll(std::source_location location) -> CommunicatorOperationLease;

  void CheckSubmission(std::span<const size_t> ranks, ncclResult_t status, std::string_view operation,
                       std::source_location location);
  void CheckGroupedSubmission(std::span<const size_t> ranks, std::span<const ncclResult_t> statuses,
                              ncclResult_t end_status, std::string_view operation, std::source_location location);
  void MarkFailed() noexcept;
  void RegisterGraph(std::source_location location);
  void UnregisterGraph() noexcept;
  void ValidateGraphLaunch(std::source_location location) const;
  void ReleasePublicOwner() noexcept;

  [[nodiscard]] auto GetBarrierStorage(size_t rank) const noexcept -> const std::shared_ptr<Storage> &;
  void BeginBarrier(size_t rank, const Stream &stream, bool capture_external, std::source_location location);
  void EndBarrier(size_t rank, const Stream &stream, bool capture_external, std::source_location location);

  void Poll(std::source_location location);
  void PollNoexcept() noexcept;
  void Close(std::source_location location);
  void Abort() noexcept;

 private:
  friend class CommunicatorOperationLease;

  CommunicatorGroupState(std::shared_ptr<RuntimeState> runtime_state, std::shared_ptr<ErrorSink> error_sink,
                         std::vector<Device> rank_order, NcclOptions options, std::source_location location) noexcept;

  void Initialize(std::source_location location);
  void InitializeBarrierStorage(std::source_location location);
  void ValidateRank(size_t rank, std::source_location location) const;
  [[nodiscard]] auto Acquire(std::vector<size_t> ranks, std::source_location location) -> CommunicatorOperationLease;
  void Release(std::span<const size_t> ranks) noexcept;
  void WaitForProgress(std::span<const size_t> ranks, std::chrono::milliseconds timeout, std::string_view operation,
                       std::source_location location);
  [[nodiscard]] auto TryAcquireAllForPoll() noexcept -> bool;
  void ReleaseAllFromPoll() noexcept;
  void AbortHandlesNoexcept() noexcept;

  std::shared_ptr<RuntimeState> runtime_state_;
  std::shared_ptr<ErrorSink> error_sink_;
  std::vector<Device> rank_order_;
  std::vector<ncclComm_t> handles_;
  std::vector<std::shared_ptr<Storage>> barrier_storage_;
  std::vector<PooledEvent> barrier_events_;
  NcclOptions options_;
  std::source_location location_;

  mutable std::mutex native_lifecycle_latch_;
  mutable std::mutex lifecycle_latch_;
  std::vector<uint8_t> rank_in_use_;
  size_t active_rank_count_{0};
  bool abort_in_progress_{false};
  bool abort_requested_{false};
  bool public_owner_alive_{true};
  size_t graph_reference_count_{0};
  std::atomic<CommunicatorStatus> status_{CommunicatorStatus::INITIALIZING};
  std::atomic<size_t> native_resource_count_{0};
};

/** Private validation and ownership gateway for public communicator facades and collective operators. */
class CommunicatorAccess final {
 public:
  [[nodiscard]] static auto GetState(NcclCommunicator &communicator, std::source_location location)
      -> const std::shared_ptr<CommunicatorGroupState> &;
  [[nodiscard]] static auto GetState(const NcclCommunicator &communicator, std::source_location location)
      -> const std::shared_ptr<CommunicatorGroupState> &;
  [[nodiscard]] static auto GetRank(const NcclCommunicator &communicator, std::source_location location) -> size_t;
};

}  // namespace ttl::internal
