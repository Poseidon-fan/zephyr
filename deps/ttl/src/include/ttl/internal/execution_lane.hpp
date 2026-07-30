#pragma once

#include <memory>
#include <optional>
#include <source_location>

#include <cublasLt.h>
#include <cublas_v2.h>

#include "ttl/internal/blas_handle_pool.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {

/** One context-private submission queue together with its stream-bound mutable library resources. */
class ExecutionLane final {
 public:
  ExecutionLane(Stream stream, std::shared_ptr<BlasHandlePool> blas_handle_pool) noexcept;

  ExecutionLane(const ExecutionLane &) = delete;
  auto operator=(const ExecutionLane &) -> ExecutionLane & = delete;
  ExecutionLane(ExecutionLane &&) noexcept = default;
  auto operator=(ExecutionLane &&) noexcept -> ExecutionLane & = default;

  [[nodiscard]] auto GetStream() const noexcept -> const Stream &;
  [[nodiscard]] auto GetCublasHandle(std::source_location location) -> cublasHandle_t;
  [[nodiscard]] auto GetCublasLtHandle(std::source_location location) -> cublasLtHandle_t;
  [[nodiscard]] auto GetBlasWorkspace(std::source_location location) -> Storage &;

 private:
  [[nodiscard]] auto GetBlas(std::source_location location) -> BlasHandleLease &;

  Stream stream_;
  std::shared_ptr<BlasHandlePool> blas_handle_pool_;
  std::optional<BlasHandleLease> blas_;
};

}  // namespace ttl::internal
