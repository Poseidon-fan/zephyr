#include "ttl/internal/cublas_api.hpp"

#include <atomic>
#include <exception>

#include <cublas_v2.h>

namespace ttl::internal {
namespace {

constinit const CublasApi CUBLAS_API{
    .create_ = cublasCreate,
    .destroy_ = cublasDestroy,
    .set_stream_ = cublasSetStream,
    .set_pointer_mode_ = cublasSetPointerMode,
    .set_workspace_ = cublasSetWorkspace,
};

constinit std::atomic<const CublasApi *> active_cublas_api{&CUBLAS_API};

}  // namespace

auto GetCublasApi() noexcept -> const CublasApi & { return *active_cublas_api.load(std::memory_order_acquire); }

ScopedCublasApiOverride::ScopedCublasApiOverride(const CublasApi &cublas_api) noexcept
    : cublas_api_(&cublas_api),
      previous_cublas_api_(active_cublas_api.exchange(cublas_api_, std::memory_order_acq_rel)) {}

ScopedCublasApiOverride::~ScopedCublasApiOverride() noexcept {
  if (active_cublas_api.exchange(previous_cublas_api_, std::memory_order_acq_rel) != cublas_api_) {
    std::terminate();
  }
}

}  // namespace ttl::internal
