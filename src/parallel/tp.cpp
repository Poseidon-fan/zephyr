#include "parallel/tp.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

#include <ttl/distributed/collective.hpp>

namespace zephyr::parallel {

auto TpContext::Create(ttl::Runtime &runtime, std::span<const ttl::Device> devices, const ttl::NcclOptions &options)
    -> std::unique_ptr<TpContext> {
  return std::unique_ptr<TpContext>{new TpContext{ttl::LocalCommunicatorGroup::Create(runtime, devices, options)}};
}

auto TpRankContext::Rank() const -> size_t { return static_cast<size_t>(communicator_->GetRank()); }

auto TpRankContext::WorldSize() const -> size_t { return static_cast<size_t>(communicator_->GetWorldSize()); }

auto TpRankContext::Device() const -> ttl::Device { return communicator_->GetDevice(); }

void TpRankContext::AllReduce(ttl::ExecutionContext &execution, ttl::Tensor &output, const ttl::Tensor &input,
                              ttl::ReduceOp operation) const {
  ttl::AllReduceOut(execution, output, input, *communicator_, operation);
}

auto TpContext::GetRank(size_t rank) -> TpRankContext {
  return TpRankContext{communicator_group_.GetCommunicator(rank)};
}

auto TpContext::GetStatus() const -> ttl::CommunicatorStatus { return communicator_group_.GetStatus(); }

void TpContext::Close() { communicator_group_.Close(); }

void TpContext::Abort() noexcept { communicator_group_.Abort(); }

}  // namespace zephyr::parallel
