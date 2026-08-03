#include "ttl/internal/distributed/nccl_api.hpp"

#include <atomic>
#include <exception>

#include <nccl.h>

namespace ttl::internal {
namespace {

constinit const NcclApi NCCL_API{
    .get_version_ = ncclGetVersion,
    .get_unique_id_ = ncclGetUniqueId,
    .comm_init_rank_config_ = ncclCommInitRankConfig,
    .comm_get_async_error_ = ncclCommGetAsyncError,
    .comm_finalize_ = ncclCommFinalize,
    .comm_destroy_ = ncclCommDestroy,
    .comm_abort_ = ncclCommAbort,
    .group_start_ = ncclGroupStart,
    .group_end_ = ncclGroupEnd,
    .all_reduce_ = ncclAllReduce,
    .reduce_ = ncclReduce,
    .all_gather_ = ncclAllGather,
    .reduce_scatter_ = ncclReduceScatter,
    .broadcast_ = ncclBroadcast,
    .send_ = ncclSend,
    .receive_ = ncclRecv,
};

constinit std::atomic<const NcclApi *> active_nccl_api{&NCCL_API};

}  // namespace

auto GetNcclApi() noexcept -> const NcclApi & { return *active_nccl_api.load(std::memory_order_acquire); }

ScopedNcclApiOverride::ScopedNcclApiOverride(const NcclApi &nccl_api) noexcept
    : nccl_api_(&nccl_api), previous_nccl_api_(active_nccl_api.exchange(nccl_api_, std::memory_order_acq_rel)) {}

ScopedNcclApiOverride::~ScopedNcclApiOverride() noexcept {
  if (active_nccl_api.exchange(previous_nccl_api_, std::memory_order_acq_rel) != nccl_api_) {
    std::terminate();
  }
}

}  // namespace ttl::internal
