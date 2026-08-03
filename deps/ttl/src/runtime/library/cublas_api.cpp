#include "ttl/internal/runtime/library/cublas_api.hpp"

#include <atomic>
#include <exception>

#include <cublasLt.h>
#include <cublas_v2.h>

namespace ttl::internal {
namespace {

constinit const CublasApi CUBLAS_API{
    .create_ = cublasCreate,
    .destroy_ = cublasDestroy,
    .lt_create_ = cublasLtCreate,
    .lt_destroy_ = cublasLtDestroy,
    .lt_matmul_desc_set_attribute_ = cublasLtMatmulDescSetAttribute,
    .lt_matrix_layout_set_attribute_ = cublasLtMatrixLayoutSetAttribute,
    .lt_matmul_preference_set_attribute_ = cublasLtMatmulPreferenceSetAttribute,
    .lt_matmul_algo_get_heuristic_ = cublasLtMatmulAlgoGetHeuristic,
    .lt_matmul_ = cublasLtMatmul,
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
