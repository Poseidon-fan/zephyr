#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>

namespace ttl::internal {

/** Injectable cuBLAS entry points used by the execution-lane resource layer. */
struct CublasApi final {
  decltype(&cublasCreate) create_;
  decltype(&cublasDestroy) destroy_;
  decltype(&cublasLtCreate) lt_create_;
  decltype(&cublasLtDestroy) lt_destroy_;
  decltype(&cublasLtMatmulDescInit) lt_matmul_desc_init_;
  decltype(&cublasLtMatrixLayoutInit) lt_matrix_layout_init_;
  decltype(&cublasLtMatmulPreferenceInit) lt_matmul_preference_init_;
  decltype(&cublasLtMatmulDescSetAttribute) lt_matmul_desc_set_attribute_;
  decltype(&cublasLtMatrixLayoutSetAttribute) lt_matrix_layout_set_attribute_;
  decltype(&cublasLtMatmulPreferenceSetAttribute) lt_matmul_preference_set_attribute_;
  decltype(&cublasLtMatmulAlgoGetHeuristic) lt_matmul_algo_get_heuristic_;
  decltype(&cublasLtMatmul) lt_matmul_;
  decltype(&cublasSetStream) set_stream_;
  decltype(&cublasSetPointerMode) set_pointer_mode_;
  decltype(&cublasSetWorkspace) set_workspace_;
};

/** Return the active process-wide table, backed by cuBLAS outside scoped tests. */
[[nodiscard]] auto GetCublasApi() noexcept -> const CublasApi &;

/**
 * @brief Process-wide cuBLAS API override for deterministic unit tests.
 *
 * Overrides may be nested only in strict LIFO order and must not overlap across host threads.
 */
class ScopedCublasApiOverride final {
 public:
  explicit ScopedCublasApiOverride(const CublasApi &cublas_api) noexcept;

  ScopedCublasApiOverride(const ScopedCublasApiOverride &) = delete;
  auto operator=(const ScopedCublasApiOverride &) -> ScopedCublasApiOverride & = delete;
  ScopedCublasApiOverride(ScopedCublasApiOverride &&) = delete;
  auto operator=(ScopedCublasApiOverride &&) -> ScopedCublasApiOverride & = delete;

  ~ScopedCublasApiOverride() noexcept;

 private:
  const CublasApi *cublas_api_;
  const CublasApi *previous_cublas_api_;
};

}  // namespace ttl::internal
