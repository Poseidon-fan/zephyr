#include "ttl/internal/op_guard.hpp"

#include <source_location>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/execution_context.hpp"
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
  Storage &storage = *TensorAccess::GetStorage(tensor, location_);
  if (storage.GetDevice() != context_.GetDevice() || stream.GetDevice() != context_.GetDevice()) {
    throw InvalidArgumentError(FormatWrongTensorDevice(context_.GetDevice(), storage.GetDevice()), location_);
  }
  storage.RecordUsage(stream);
}

void OpGuard::RecordTensor(const Tensor &tensor) { RecordTensorOnStream(tensor, context_.GetStream()); }

void OpGuard::CheckLaunch() const { CheckCuda(GetCudaApi().peek_at_last_error_(), operation_, location_); }

auto OpGuard::GetStream() const noexcept -> const Stream & { return context_.GetStream(); }

auto OpGuard::GetNativeStream() const noexcept -> cudaStream_t { return StreamAccess::GetNative(context_.GetStream()); }

}  // namespace ttl::internal
