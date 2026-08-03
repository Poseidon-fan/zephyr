#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <string_view>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <driver_types.h>

#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::internal {

class ExecutionContextImpl;
class ExecutionLane;

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
  [[nodiscard]] auto GetAuxiliaryCublasHandle(size_t index) const -> cublasHandle_t;
  [[nodiscard]] auto GetAuxiliaryCublasLtHandle(size_t index) const -> cublasLtHandle_t;
  [[nodiscard]] auto GetAuxiliaryBlasWorkspace(size_t index) const -> void *;
  [[nodiscard]] auto GetAuxiliaryBlasWorkspaceBytes(size_t index) const -> size_t;
  [[nodiscard]] auto MakeAuxiliaryScratchScope(size_t index) const -> ScratchArena::Scope;
  void ReserveAuxiliaryScratch(size_t index, size_t capacity_bytes) const;
  [[nodiscard]] auto GetAuxiliaryScratchCapacityBytes(size_t index) const -> size_t;
  [[nodiscard]] auto GetAuxiliaryScratchHighWaterBytes(size_t index) const -> size_t;

  void RecordTensor(const Tensor &tensor, size_t auxiliary_stream_index);
  void CheckLaunch() const;
  void Finish();
  void FailExternalSubmissionNoexcept() noexcept;

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

  void FailNoexcept(bool report_unfinished_scope) noexcept;
  void MarkFailed() noexcept;
  [[nodiscard]] auto EnqueueJoin() noexcept -> JoinResult;
  [[nodiscard]] auto GetAuxiliaryLane(size_t index) const -> ExecutionLane &;

  OpGuard &guard_;
  ExecutionContextImpl &impl_;
  size_t auxiliary_stream_count_;
  std::source_location location_;
  Status status_{Status::ACTIVE};
};

}  // namespace ttl::internal
