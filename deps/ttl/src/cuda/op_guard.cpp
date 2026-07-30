#include "ttl/internal/op_guard.hpp"

#include <cstddef>
#include <source_location>
#include <string>
#include <string_view>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <driver_types.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/internal/execution_lane.hpp"
#include "ttl/internal/scratch_arena.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/stream.hpp"
#include "ttl/tensor.hpp"

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

OpGuard::OpGuard(ExecutionContext &context, std::string_view operation, std::source_location location)
    : context_(context),
      operation_(operation),
      location_(location),
      use_guard_(context, ContextUseMode::SUBMIT, location),
      device_guard_(context.GetDevice(), *ContextAccess::GetErrorSink(context, location), location) {
  if (operation_.empty()) {
    throw InvalidArgumentError("operator name must not be empty", location_);
  }
}

void OpGuard::RecordTensorOnStream(const Tensor &tensor, const Stream &stream) {
  ValidateTensor(tensor);
  if (stream.GetDevice() != context_.GetDevice()) {
    throw InvalidArgumentError("operator stream device does not match the execution context", location_);
  }
  TensorAccess::GetStorage(tensor, location_)->RecordUsage(stream);
}

void OpGuard::ValidateTensor(const Tensor &tensor) const {
  const auto &storage = *TensorAccess::GetStorage(tensor, location_);
  if (storage.GetDevice() != context_.GetDevice()) {
    throw InvalidArgumentError(FormatWrongTensorDevice(context_.GetDevice(), storage.GetDevice()), location_);
  }
}

void OpGuard::RecordTensor(const Tensor &tensor) { RecordTensorOnStream(tensor, context_.GetStream()); }

void OpGuard::CheckLaunch() const { CheckCuda(GetCudaApi().peek_at_last_error_(), operation_, location_); }

auto OpGuard::RegisterDeviceError(DType source_dtype, DType target_dtype) -> DeviceErrorLaunchContext {
  return ContextAccess::GetDeviceErrorState(context_, location_)
      .Register(context_.GetStream(), source_dtype, target_dtype, location_);
}

auto OpGuard::GetStream() const noexcept -> const Stream & { return context_.GetStream(); }

auto OpGuard::GetNativeStream() const noexcept -> cudaStream_t { return StreamAccess::GetNative(context_.GetStream()); }

auto OpGuard::GetCublasHandle() const -> cublasHandle_t {
  return ContextAccess::GetPrimaryLane(context_, location_).GetCublasHandle(location_);
}

auto OpGuard::GetCublasLtHandle() const -> cublasLtHandle_t {
  return ContextAccess::GetPrimaryLane(context_, location_).GetCublasLtHandle(location_);
}

auto OpGuard::GetBlasWorkspace() const -> void * {
  return ContextAccess::GetPrimaryLane(context_, location_).GetBlasWorkspace(location_).GetBasePointer();
}

auto OpGuard::GetBlasWorkspaceBytes() const -> size_t {
  return ContextAccess::GetPrimaryLane(context_, location_).GetBlasWorkspace(location_).GetCapacityBytes();
}

auto OpGuard::MakeScratchScope() -> ScratchArena::Scope {
  return ContextAccess::GetPrimaryLane(context_, location_).MakeScratchScope(ScratchGrowthPolicy::GROWABLE, location_);
}

void OpGuard::ReserveScratch(size_t capacity_bytes) {
  ContextAccess::GetPrimaryLane(context_, location_).ReserveScratch(capacity_bytes, location_);
}

auto OpGuard::GetScratchCapacityBytes() const -> size_t {
  return ContextAccess::GetPrimaryLane(context_, location_).GetScratchCapacityBytes();
}

auto OpGuard::GetScratchHighWaterBytes() const -> size_t {
  return ContextAccess::GetPrimaryLane(context_, location_).GetScratchHighWaterBytes();
}

}  // namespace ttl::internal
