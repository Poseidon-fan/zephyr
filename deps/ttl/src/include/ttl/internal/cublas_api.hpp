#pragma once

#include <cublas_v2.h>

namespace ttl::internal {

/** Injectable cuBLAS entry points used by the execution-lane resource layer. */
struct CublasApi final {
  decltype(&cublasCreate) create_;
  decltype(&cublasDestroy) destroy_;
  decltype(&cublasSetStream) set_stream_;
  decltype(&cublasSetPointerMode) set_pointer_mode_;
  decltype(&cublasSetWorkspace) set_workspace_;
};

/** Return the active process-wide table, backed by cuBLAS outside scoped tests. */
[[nodiscard]] auto GetCublasApi() noexcept -> const CublasApi &;

/**
 * Process-wide cuBLAS API override for deterministic unit tests.
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
