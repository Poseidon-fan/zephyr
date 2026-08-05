#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>

#include <cuda_runtime_api.h>

#include "ttl/common/device.hpp"
#include "ttl/internal/runtime/error_report.hpp"

namespace ttl {

class ErrorSink;

}  // namespace ttl

namespace ttl::internal {

struct CudaMemoryPoolStats final {
  uint64_t used_bytes_;
  uint64_t reserved_bytes_;
};

/** Private CUDA memory-pool owner used by one device allocator. */
class CudaMemoryPool final {
 public:
  [[nodiscard]] static auto Create(Device device, const std::shared_ptr<ErrorSink> &error_sink,
                                   uint64_t release_threshold_bytes, std::source_location location)
      -> std::unique_ptr<CudaMemoryPool>;

  CudaMemoryPool(const CudaMemoryPool &) = delete;
  auto operator=(const CudaMemoryPool &) -> CudaMemoryPool & = delete;
  CudaMemoryPool(CudaMemoryPool &&) = delete;
  auto operator=(CudaMemoryPool &&) -> CudaMemoryPool & = delete;
  ~CudaMemoryPool() noexcept;

  [[nodiscard]] auto AllocateAsync(void **pointer, size_t bytes, cudaStream_t stream) const noexcept -> cudaError_t;
  [[nodiscard]] auto FreeAsync(void *pointer, cudaStream_t stream) const noexcept -> cudaError_t;
  void SetPeerAccess(Device peer, bool enabled, std::source_location location);
  void TrimTo(size_t target_reserved_bytes, std::source_location location);
  [[nodiscard]] auto GetStats(std::source_location location) const -> CudaMemoryPoolStats;
  [[nodiscard]] auto TryGetStats(const ErrorReportContext &context) const noexcept -> CudaMemoryPoolStats;
  void Close(std::source_location location);
  [[nodiscard]] auto TryCloseNoexcept() noexcept -> bool;

 private:
  CudaMemoryPool(Device device, std::shared_ptr<ErrorSink> error_sink, cudaMemPool_t pool,
                 std::source_location location) noexcept;

  Device device_;
  std::shared_ptr<ErrorSink> error_sink_;
  cudaMemPool_t pool_;
  std::source_location location_;
};

}  // namespace ttl::internal
