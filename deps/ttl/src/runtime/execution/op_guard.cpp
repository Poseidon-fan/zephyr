#include "ttl/internal/runtime/execution/op_guard.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <source_location>
#include <string>
#include <string_view>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <driver_types.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/ops/matmul_plan.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/execution/device_error.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/execution/execution_lane.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/graph/graph.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/internal/tensor/storage.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/device.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::internal {
namespace {

[[nodiscard]] auto FormatWrongTensorDevice(Device context_device, Device tensor_device) -> std::string {
  std::string message{"execution context for "};
  message.append(context_device.ToString());
  message.append(" cannot access a tensor on ");
  message.append(tensor_device.ToString());
  return message;
}

}  // namespace

OpGuard::OpGuard(ExecutionContext &context, std::string_view operation, std::source_location location,
                 CapturePolicy capture_policy)
    : context_(context),
      operation_(operation),
      location_(location),
      use_guard_(context, ContextUseMode::SUBMIT, location),
      device_guard_(context.GetDevice(), *ContextAccess::GetErrorSink(context, location), location) {
  if (operation_.empty()) {
    throw InvalidArgumentError("operator name must not be empty", location_);
  }
  capture_state_ = GetCaptureState(context_, location_);
  if (capture_state_ != nullptr) {
    if (capture_policy != CapturePolicy::SAFE) {
      throw CaptureError(std::string{operation_} + " is not allowed during CUDA graph capture", location_);
    }
    capture_state_->BeginOperation(operation_, location_);
  }
}

void OpGuard::RecordTensorOnStream(const Tensor &tensor, const Stream &stream) {
  ValidateTensor(tensor);
  if (stream.GetDevice() != context_.GetDevice()) {
    throw InvalidArgumentError("operator stream device does not match the execution context", location_);
  }
  const auto &storage = TensorAccess::GetStorage(tensor, location_);
  storage->RecordUsage(stream);
  RetainStorage(storage);
}

void OpGuard::ValidateTensor(const Tensor &tensor) const {
  const auto &storage = *TensorAccess::GetStorage(tensor, location_);
  if (storage.GetDevice() != context_.GetDevice()) {
    throw InvalidArgumentError(FormatWrongTensorDevice(context_.GetDevice(), storage.GetDevice()), location_);
  }
}

void OpGuard::RecordTensor(const Tensor &tensor) { RecordTensorOnStream(tensor, context_.GetStream()); }

void OpGuard::RetainStorage(const std::shared_ptr<Storage> &storage) {
  if (capture_state_ != nullptr) {
    capture_state_->RetainStorage(storage, location_);
  }
}

void OpGuard::RetainCommunicator(const std::shared_ptr<CommunicatorGroupState> &communicator) {
  if (capture_state_ != nullptr) {
    capture_state_->RetainCommunicator(communicator, location_);
  }
}

void OpGuard::CheckLaunch() const {
  const auto status = GetCudaApi().get_last_error_();
  if (status != cudaSuccess) {
    ContextAccess::GetImpl(context_, location_)
        .status_.store(ExecutionContextStatus::FAILED, std::memory_order_release);
    if (capture_state_ != nullptr) {
      capture_state_->Invalidate();
    }
  }
  CheckCuda(status, operation_, location_);
}

void OpGuard::FailExternalSubmissionNoexcept() noexcept {
  if (capture_state_ != nullptr) {
    capture_state_->Invalidate();
  }
  try {
    auto &impl = ContextAccess::GetImpl(context_, location_);
    impl.status_.store(ExecutionContextStatus::FAILED, std::memory_order_release);
    const auto status = GetCudaApi().get_last_error_();
    if (status == cudaSuccess) {
      return;
    }
    const ErrorReportContext report_context{
        .location_ = location_,
        .device_ = context_.GetDevice(),
        .stream_id_ = context_.GetStream().GetId(),
    };
    TryCuda(status, operation_, *ContextAccess::GetErrorSink(context_, location_), report_context);
  } catch (...) {
    return;
  }
}

auto OpGuard::RegisterDeviceError(DType source_dtype, DType target_dtype) -> DeviceErrorLaunchContext {
  auto &state = ContextAccess::GetDeviceErrorState(context_, location_);
  RetainStorage(state.GetStorage());
  return state.Register(context_.GetStream(), source_dtype, target_dtype, location_);
}

auto OpGuard::GetStream() const noexcept -> const Stream & { return context_.GetStream(); }

auto OpGuard::GetNativeStream() const noexcept -> cudaStream_t { return StreamAccess::GetNative(context_.GetStream()); }

auto OpGuard::GetCublasHandle() const -> cublasHandle_t {
  auto &lane = ContextAccess::GetPrimaryLane(context_, location_);
  if (capture_state_ != nullptr && !lane.HasBlas()) {
    throw CaptureError("cuBLAS resources must be warmed up before CUDA graph capture", location_);
  }
  return lane.GetCublasHandle(location_);
}

auto OpGuard::GetCublasLtHandle() const -> cublasLtHandle_t {
  auto &lane = ContextAccess::GetPrimaryLane(context_, location_);
  if (capture_state_ != nullptr && !lane.HasBlas()) {
    throw CaptureError("cuBLAS resources must be warmed up before CUDA graph capture", location_);
  }
  return lane.GetCublasLtHandle(location_);
}

auto OpGuard::GetBlasWorkspace() const -> void * {
  auto &lane = ContextAccess::GetPrimaryLane(context_, location_);
  if (capture_state_ != nullptr && !lane.HasBlas()) {
    throw CaptureError("cuBLAS resources must be warmed up before CUDA graph capture", location_);
  }
  const auto &storage = lane.GetBlasWorkspaceStorage(location_);
  if (capture_state_ != nullptr) {
    capture_state_->RetainStorage(storage, location_);
  }
  return storage->GetBasePointer();
}

auto OpGuard::GetBlasWorkspaceBytes() const -> size_t {
  auto &lane = ContextAccess::GetPrimaryLane(context_, location_);
  if (capture_state_ != nullptr && !lane.HasBlas()) {
    throw CaptureError("cuBLAS resources must be warmed up before CUDA graph capture", location_);
  }
  const auto &storage = lane.GetBlasWorkspaceStorage(location_);
  if (capture_state_ != nullptr) {
    capture_state_->RetainStorage(storage, location_);
  }
  return storage->GetCapacityBytes();
}

auto OpGuard::GetMatmulAlgorithmCache() const -> MatmulAlgorithmCache & {
  return ContextAccess::GetMatmulAlgorithmCache(context_, location_);
}

auto OpGuard::MakeScratchScope() -> ScratchArena::Scope {
  auto &lane = ContextAccess::GetPrimaryLane(context_, location_);
  if (capture_state_ != nullptr && lane.GetScratchStorage() != nullptr) {
    capture_state_->RetainStorage(lane.GetScratchStorage(), location_);
  }
  return lane.MakeScratchScope(
      capture_state_ == nullptr ? ScratchGrowthPolicy::GROWABLE : ScratchGrowthPolicy::FIXED_CAPACITY, location_);
}

void OpGuard::ReserveScratch(size_t capacity_bytes) {
  if (capture_state_ != nullptr) {
    throw CaptureError("scratch capacity cannot be reserved during CUDA graph capture", location_);
  }
  ContextAccess::GetPrimaryLane(context_, location_).ReserveScratch(capacity_bytes, location_);
}

auto OpGuard::GetScratchCapacityBytes() const -> size_t {
  return ContextAccess::GetPrimaryLane(context_, location_).GetScratchCapacityBytes();
}

auto OpGuard::GetScratchHighWaterBytes() const -> size_t {
  return ContextAccess::GetPrimaryLane(context_, location_).GetScratchHighWaterBytes();
}

auto OpGuard::IsCapturing() const noexcept -> bool { return capture_state_ != nullptr; }

}  // namespace ttl::internal
