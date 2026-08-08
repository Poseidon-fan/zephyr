#pragma once

#include <nccl.h>

namespace ttl::internal {

/** Injectable NCCL entry points used by communicator lifecycle and collective submission. */
struct NcclApi final {
  decltype(&ncclGetUniqueId) get_unique_id_;
  decltype(&ncclCommInitRankConfig) comm_init_rank_config_;
  decltype(&ncclCommGetAsyncError) comm_get_async_error_;
  decltype(&ncclCommFinalize) comm_finalize_;
  decltype(&ncclCommDestroy) comm_destroy_;
  decltype(&ncclCommAbort) comm_abort_;
  decltype(&ncclGroupStart) group_start_;
  decltype(&ncclGroupEnd) group_end_;
  decltype(&ncclAllReduce) all_reduce_;
  decltype(&ncclReduce) reduce_;
  decltype(&ncclAllGather) all_gather_;
  decltype(&ncclReduceScatter) reduce_scatter_;
  decltype(&ncclBroadcast) broadcast_;
  decltype(&ncclSend) send_;
  decltype(&ncclRecv) receive_;
};

/** Return the active process-wide table, backed by NCCL outside scoped tests. */
[[nodiscard]] auto GetNcclApi() noexcept -> const NcclApi &;

/**
 * @brief Process-wide NCCL API override for deterministic unit tests.
 *
 * Overrides may be nested only in strict LIFO order and must not overlap across host threads.
 */
class ScopedNcclApiOverride final {
 public:
  explicit ScopedNcclApiOverride(const NcclApi &nccl_api) noexcept;

  ScopedNcclApiOverride(const ScopedNcclApiOverride &) = delete;
  auto operator=(const ScopedNcclApiOverride &) -> ScopedNcclApiOverride & = delete;
  ScopedNcclApiOverride(ScopedNcclApiOverride &&) = delete;
  auto operator=(ScopedNcclApiOverride &&) -> ScopedNcclApiOverride & = delete;

  ~ScopedNcclApiOverride() noexcept;

 private:
  const NcclApi *nccl_api_;
  const NcclApi *previous_nccl_api_;
};

}  // namespace ttl::internal
