#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

#include <ttl/common/device.hpp>
#include <ttl/distributed/collective.hpp>
#include <ttl/distributed/communicator.hpp>
#include <ttl/runtime/execution_context.hpp>
#include <ttl/runtime/runtime.hpp>
#include <ttl/tensor/tensor.hpp>

namespace zephyr::parallel {

/** Borrowed rank-local communicator view. The owning TpContext must outlive it. */
class TpRankContext final {
 public:
  TpRankContext(const TpRankContext &) = default;
  auto operator=(const TpRankContext &) -> TpRankContext & = default;
  TpRankContext(TpRankContext &&) noexcept = default;
  auto operator=(TpRankContext &&) noexcept -> TpRankContext & = default;

  [[nodiscard]] auto Rank() const -> size_t;
  [[nodiscard]] auto WorldSize() const -> size_t;
  [[nodiscard]] auto Device() const -> ttl::Device;
  /** Enqueue an in-place or out-of-place all-reduce on this rank's stream. */
  void AllReduce(ttl::ExecutionContext &execution, ttl::Tensor &output, const ttl::Tensor &input,
                 ttl::ReduceOp operation) const;

 private:
  friend class TpContext;

  explicit TpRankContext(ttl::NcclCommunicator &communicator) noexcept : communicator_(&communicator) {}

  ttl::NcclCommunicator *communicator_;
};

/**
 * Owner of one process-local tensor parallel communicator group. The caller's
 * runtime must outlive this object; rank workers must be joined before destruction.
 * Workers keep a value copy of the view returned by GetRank.
 */
class TpContext final {
 public:
  /** Assign devices[i] to rank i. The device span is borrowed only for this call. */
  [[nodiscard]] static auto Create(ttl::Runtime &runtime, std::span<const ttl::Device> devices,
                                   const ttl::NcclOptions &options = {}) -> std::unique_ptr<TpContext>;

  TpContext(const TpContext &) = delete;
  auto operator=(const TpContext &) -> TpContext & = delete;
  TpContext(TpContext &&) = delete;
  auto operator=(TpContext &&) -> TpContext & = delete;
  ~TpContext() noexcept = default;

  [[nodiscard]] auto WorldSize() const noexcept -> size_t { return communicator_group_.GetWorldSize(); }
  /** Return a rank-local view that the worker may store by value. */
  [[nodiscard]] auto GetRank(size_t rank) -> TpRankContext;
  [[nodiscard]] auto GetStatus() const -> ttl::CommunicatorStatus;

  /** Close the group after workers and captured graphs have released all submissions. */
  void Close();
  void Abort() noexcept;

 private:
  explicit TpContext(ttl::LocalCommunicatorGroup communicator_group) noexcept
      : communicator_group_(std::move(communicator_group)) {}

  ttl::LocalCommunicatorGroup communicator_group_;
};

}  // namespace zephyr::parallel
