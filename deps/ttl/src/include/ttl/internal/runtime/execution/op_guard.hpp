#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <string_view>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <driver_types.h>

#include "ttl/internal/runtime/device_guard.hpp"
#include "ttl/internal/runtime/execution/device_error.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::internal {

class CaptureSessionState;
class CommunicatorGroupState;
class MatmulAlgorithmCache;
class ParallelOpScope;
class Storage;

enum class CapturePolicy : uint8_t {
  FORBIDDEN,
  SAFE,
};

/** Common checked entry scope for CUDA operator wrappers. */
class OpGuard final {
 public:
  OpGuard(ExecutionContext &context, std::string_view operation,
          std::source_location location = std::source_location::current(),
          CapturePolicy capture_policy = CapturePolicy::FORBIDDEN);

  OpGuard(const OpGuard &) = delete;
  auto operator=(const OpGuard &) -> OpGuard & = delete;
  OpGuard(OpGuard &&) = delete;
  auto operator=(OpGuard &&) -> OpGuard & = delete;

  void ValidateTensor(const Tensor &tensor) const;
  void RecordTensor(const Tensor &tensor);
  void RetainStorage(const std::shared_ptr<Storage> &storage);
  void RetainCommunicator(const std::shared_ptr<CommunicatorGroupState> &communicator);
  void CheckLaunch() const;
  void FailExternalSubmissionNoexcept() noexcept;
  [[nodiscard]] auto RegisterDeviceError(DType source_dtype, DType target_dtype) -> DeviceErrorLaunchContext;

  [[nodiscard]] auto GetStream() const noexcept -> const Stream &;
  [[nodiscard]] auto GetNativeStream() const noexcept -> cudaStream_t;
  [[nodiscard]] auto GetCublasHandle() const -> cublasHandle_t;
  [[nodiscard]] auto GetCublasLtHandle() const -> cublasLtHandle_t;
  [[nodiscard]] auto GetBlasWorkspace() const -> void *;
  [[nodiscard]] auto GetBlasWorkspaceBytes() const -> size_t;
  [[nodiscard]] auto GetMatmulAlgorithmCache() const -> MatmulAlgorithmCache &;
  [[nodiscard]] auto MakeScratchScope() -> ScratchArena::Scope;
  void ReserveScratch(size_t capacity_bytes);
  [[nodiscard]] auto GetScratchCapacityBytes() const -> size_t;
  [[nodiscard]] auto GetScratchHighWaterBytes() const -> size_t;
  [[nodiscard]] auto IsCapturing() const noexcept -> bool;

 private:
  friend class ParallelOpScope;

  void RecordTensorOnStream(const Tensor &tensor, const Stream &stream);

  ExecutionContext &context_;
  std::string_view operation_;
  std::source_location location_;
  ContextUseGuard use_guard_;
  DeviceGuard device_guard_;
  bool parallel_scope_active_{false};
  std::shared_ptr<CaptureSessionState> capture_state_;
};

}  // namespace ttl::internal
