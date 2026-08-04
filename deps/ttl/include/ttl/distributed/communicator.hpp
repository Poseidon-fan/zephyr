#pragma once

#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <span>
#include <type_traits>
#include <vector>

#include "ttl/common/device.hpp"

namespace ttl::internal {

class CommunicatorAccess;
class CommunicatorGroupState;

}  // namespace ttl::internal

namespace ttl {

class Runtime;

/** Process-local NCCL communicator lifecycle visible to callers. */
enum class CommunicatorStatus : uint8_t {
  INITIALIZING,
  READY,
  FAILED,
  FINALIZING,
  ABORTED,
  CLOSED,
};

/** Bounded host-progress waits for nonblocking NCCL lifecycle and enqueue operations. */
struct NcclOptions final {
  std::chrono::milliseconds initialization_timeout_{30'000};
  std::chrono::milliseconds enqueue_timeout_{30'000};
  std::chrono::milliseconds finalize_timeout_{30'000};
};

/**
 * Move-only rank-local view of one communicator in a process-local group.
 *
 * A communicator is bound to one CUDA device and one rank. Closing or aborting any rank acts on the complete local
 * group because NCCL failures and collective ordering are group-wide.
 */
class NcclCommunicator final {
 public:
  NcclCommunicator() = delete;
  NcclCommunicator(const NcclCommunicator &) = delete;
  auto operator=(const NcclCommunicator &) -> NcclCommunicator & = delete;
  NcclCommunicator(NcclCommunicator &&) noexcept = default;
  auto operator=(NcclCommunicator &&) noexcept -> NcclCommunicator & = default;
  ~NcclCommunicator() noexcept = default;

  [[nodiscard]] auto GetRank(std::source_location location = std::source_location::current()) const -> int32_t;
  [[nodiscard]] auto GetWorldSize(std::source_location location = std::source_location::current()) const -> int32_t;
  [[nodiscard]] auto GetDevice(std::source_location location = std::source_location::current()) const -> Device;
  [[nodiscard]] auto GetStatus(std::source_location location = std::source_location::current()) const
      -> CommunicatorStatus;

  void PollAsyncError(std::source_location location = std::source_location::current());
  void Close(std::source_location location = std::source_location::current());
  void Abort() noexcept;

 private:
  friend class internal::CommunicatorAccess;
  friend class LocalCommunicatorGroup;

  NcclCommunicator(std::shared_ptr<internal::CommunicatorGroupState> state, size_t rank) noexcept;

  std::shared_ptr<internal::CommunicatorGroupState> state_;
  size_t rank_;
};

/**
 * Owner of every rank-local NCCL communicator for one explicit process-local rank order.
 *
 * Construction and close are all-or-nothing. The group has no background progress thread; callers may use Poll, and
 * Runtime::Poll also advances registered groups when they are idle. Runtime::Shutdown rejects an open group.
 */
class LocalCommunicatorGroup final {
 public:
  [[nodiscard]] static auto Create(Runtime &runtime, std::span<const Device> rank_order,
                                   const NcclOptions &options = {},
                                   std::source_location location = std::source_location::current())
      -> LocalCommunicatorGroup;

  LocalCommunicatorGroup() = delete;
  LocalCommunicatorGroup(const LocalCommunicatorGroup &) = delete;
  auto operator=(const LocalCommunicatorGroup &) -> LocalCommunicatorGroup & = delete;
  LocalCommunicatorGroup(LocalCommunicatorGroup &&) noexcept = default;
  auto operator=(LocalCommunicatorGroup &&) noexcept -> LocalCommunicatorGroup & = delete;
  ~LocalCommunicatorGroup() noexcept;

  [[nodiscard]] auto GetWorldSize(std::source_location location = std::source_location::current()) const -> size_t;
  [[nodiscard]] auto GetCommunicator(size_t rank, std::source_location location = std::source_location::current())
      -> NcclCommunicator &;
  [[nodiscard]] auto GetRankOrder(std::source_location location = std::source_location::current()) const
      -> std::span<const Device>;
  [[nodiscard]] auto GetStatus(std::source_location location = std::source_location::current()) const
      -> CommunicatorStatus;

  void Poll(std::source_location location = std::source_location::current());
  void Close(std::source_location location = std::source_location::current());
  void Abort() noexcept;

 private:
  LocalCommunicatorGroup(std::shared_ptr<internal::CommunicatorGroupState> state,
                         std::vector<NcclCommunicator> communicators) noexcept;

  std::shared_ptr<internal::CommunicatorGroupState> state_;
  std::vector<NcclCommunicator> communicators_;
};

static_assert(!std::default_initializable<NcclCommunicator>);
static_assert(!std::copy_constructible<NcclCommunicator>);
static_assert(std::is_nothrow_move_constructible_v<NcclCommunicator>);
static_assert(!std::default_initializable<LocalCommunicatorGroup>);
static_assert(!std::copy_constructible<LocalCommunicatorGroup>);
static_assert(std::is_nothrow_move_constructible_v<LocalCommunicatorGroup>);

}  // namespace ttl
