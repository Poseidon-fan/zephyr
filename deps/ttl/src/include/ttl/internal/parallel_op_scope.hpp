#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <string_view>

#include <cuda_runtime_api.h>

#include "ttl/internal/op_guard.hpp"
#include "ttl/stream.hpp"
#include "ttl/tensor.hpp"

namespace ttl::internal {

class ExecutionContextImpl;

/**
 * Structured fork/join scope for one operator that submits work to context-private auxiliary streams.
 *
 * Tensor usage must be recorded before the first submission on each auxiliary stream. Finish is the throwing success
 * boundary. Destruction of an unfinished scope performs best-effort joining, marks the context failed, and never
 * throws.
 */
class ParallelOpScope final {
 public:
  ParallelOpScope(OpGuard &guard, size_t auxiliary_stream_count,
                  std::source_location location = std::source_location::current());

  ParallelOpScope(const ParallelOpScope &) = delete;
  auto operator=(const ParallelOpScope &) -> ParallelOpScope & = delete;
  ParallelOpScope(ParallelOpScope &&) = delete;
  auto operator=(ParallelOpScope &&) -> ParallelOpScope & = delete;

  ~ParallelOpScope() noexcept;

  [[nodiscard]] auto GetAuxiliaryStreamCount() const noexcept -> size_t;
  [[nodiscard]] auto GetAuxiliaryStream(size_t index) const -> const Stream &;
  [[nodiscard]] auto GetNativeAuxiliaryStream(size_t index) const -> cudaStream_t;

  void RecordTensor(const Tensor &tensor, size_t auxiliary_stream_index);
  void CheckLaunch() const;
  void Finish();

 private:
  struct JoinResult final {
    cudaError_t status_;
    std::string_view operation_;
  };

  enum class Status : uint8_t {
    ACTIVE,
    JOINED,
    FAILED,
  };

  void MarkFailed() noexcept;
  [[nodiscard]] auto EnqueueJoin() noexcept -> JoinResult;

  OpGuard &guard_;
  ExecutionContextImpl &impl_;
  size_t auxiliary_stream_count_;
  std::source_location location_;
  Status status_{Status::ACTIVE};
};

}  // namespace ttl::internal
