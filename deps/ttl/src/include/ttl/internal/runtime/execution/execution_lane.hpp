#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <source_location>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "ttl/internal/runtime/library/blas_handle_pool.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/internal/tensor/storage.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {

/** One context-private submission queue together with its stream-bound mutable library resources. */
class ExecutionLane final {
 public:
  ExecutionLane(Stream stream, std::shared_ptr<BlasHandlePool> blas_handle_pool,
                std::shared_ptr<DeviceAllocator> allocator);

  ExecutionLane(const ExecutionLane &) = delete;
  auto operator=(const ExecutionLane &) -> ExecutionLane & = delete;
  ExecutionLane(ExecutionLane &&) noexcept = default;
  auto operator=(ExecutionLane &&) noexcept -> ExecutionLane & = default;

  [[nodiscard]] auto GetStream() const noexcept -> const Stream &;
  [[nodiscard]] auto GetCublasHandle(std::source_location location) -> cublasHandle_t;
  [[nodiscard]] auto GetCublasLtHandle(std::source_location location) -> cublasLtHandle_t;
  [[nodiscard]] auto GetBlasWorkspace(std::source_location location) -> Storage &;
  [[nodiscard]] auto MakeScratchScope(ScratchGrowthPolicy growth_policy = ScratchGrowthPolicy::GROWABLE,
                                      std::source_location location = std::source_location::current())
      -> ScratchArena::Scope;
  void ReserveScratch(size_t capacity_bytes, std::source_location location = std::source_location::current());
  [[nodiscard]] auto GetScratchCapacityBytes() const noexcept -> size_t;
  [[nodiscard]] auto GetScratchHighWaterBytes() const noexcept -> size_t;
  [[nodiscard]] auto GetScratchStorage() const noexcept -> const std::shared_ptr<Storage> &;
  [[nodiscard]] auto GetBlasWorkspaceStorage(std::source_location location) -> const std::shared_ptr<Storage> &;
  [[nodiscard]] auto HasBlas() const noexcept -> bool;

 private:
  [[nodiscard]] auto GetBlas(std::source_location location) -> BlasHandleLease &;

  Stream stream_;
  std::shared_ptr<BlasHandlePool> blas_handle_pool_;
  std::unique_ptr<ScratchArena> scratch_arena_;
  std::optional<BlasHandleLease> blas_;
};

}  // namespace ttl::internal
