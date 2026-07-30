#include "ttl/internal/execution_lane.hpp"

#include <memory>
#include <source_location>
#include <utility>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "ttl/internal/blas_handle_pool.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {

ExecutionLane::ExecutionLane(Stream stream, std::shared_ptr<BlasHandlePool> blas_handle_pool) noexcept
    : stream_(std::move(stream)), blas_handle_pool_(std::move(blas_handle_pool)) {}

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

auto ExecutionLane::GetBlas(std::source_location location) -> BlasHandleLease & {
  if (!blas_.has_value()) {
    blas_.emplace(blas_handle_pool_->Acquire(stream_, location));
  }
  return *blas_;
}

}  // namespace ttl::internal
