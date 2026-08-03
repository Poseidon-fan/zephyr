#include "ttl/internal/runtime/execution/execution_lane.hpp"

#include <memory>
#include <source_location>
#include <utility>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "ttl/internal/runtime/library/blas_handle_pool.hpp"
#include "ttl/internal/runtime/memory/device_allocator.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/internal/tensor/storage.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {

ExecutionLane::ExecutionLane(Stream stream, std::shared_ptr<BlasHandlePool> blas_handle_pool,
                             std::shared_ptr<DeviceAllocator> allocator)
    : stream_(std::move(stream)),
      blas_handle_pool_(std::move(blas_handle_pool)),
      scratch_arena_(std::make_unique<ScratchArena>(stream_, std::move(allocator))) {}

auto ExecutionLane::GetStream() const noexcept -> const Stream & { return stream_; }

auto ExecutionLane::GetCublasHandle(std::source_location location) -> cublasHandle_t {
  return GetBlas(location).GetCublasHandle();
}

auto ExecutionLane::GetCublasLtHandle(std::source_location location) -> cublasLtHandle_t {
  return GetBlas(location).GetCublasLtHandle();
}

auto ExecutionLane::GetBlasWorkspace(std::source_location location) -> Storage & {
  return GetBlas(location).GetWorkspace();
}

auto ExecutionLane::MakeScratchScope(ScratchGrowthPolicy growth_policy, std::source_location location)
    -> ScratchArena::Scope {
  return scratch_arena_->MakeScope(growth_policy, location);
}

void ExecutionLane::ReserveScratch(size_t capacity_bytes, std::source_location location) {
  scratch_arena_->Reserve(capacity_bytes, location);
}

auto ExecutionLane::GetScratchCapacityBytes() const noexcept -> size_t { return scratch_arena_->GetCapacityBytes(); }

auto ExecutionLane::GetScratchHighWaterBytes() const noexcept -> size_t { return scratch_arena_->GetHighWaterBytes(); }

auto ExecutionLane::GetScratchStorage() const noexcept -> const std::shared_ptr<Storage> & {
  return scratch_arena_->GetStorage();
}

auto ExecutionLane::GetBlasWorkspaceStorage(std::source_location location) -> const std::shared_ptr<Storage> & {
  return GetBlas(location).GetWorkspaceStorage();
}

auto ExecutionLane::HasBlas() const noexcept -> bool { return blas_.has_value(); }

auto ExecutionLane::GetBlas(std::source_location location) -> BlasHandleLease & {
  if (!blas_.has_value()) {
    blas_.emplace(blas_handle_pool_->Acquire(stream_, location));
  }
  return *blas_;
}

}  // namespace ttl::internal
