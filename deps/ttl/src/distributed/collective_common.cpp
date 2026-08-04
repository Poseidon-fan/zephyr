#include "ttl/internal/distributed/collective_common.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include <driver_types.h>
#include <nccl.h>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/distributed/communicator.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/distributed/communicator.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/execution/execution_lane.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/internal/tensor/storage.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::internal {

auto PreparedCollectiveCall::GetInput() const noexcept -> const Tensor & {
  return contiguous_input_.has_value() ? *contiguous_input_ : *input_;
}

auto PreparedCollectiveCall::GetOutput() noexcept -> Tensor & {
  return contiguous_output_.has_value() ? *contiguous_output_ : *output_;
}

auto PreparedCollectiveCall::GetOutput() const noexcept -> const Tensor & {
  return contiguous_output_.has_value() ? *contiguous_output_ : *output_;
}

auto ToNcclDType(DType dtype, std::source_location location) -> ncclDataType_t {
  switch (dtype) {
    case DType::BOOL:
    case DType::UINT8:
      return ncclUint8;
    case DType::INT32:
      return ncclInt32;
    case DType::INT64:
      return ncclInt64;
    case DType::FLOAT16:
      return ncclFloat16;
    case DType::BFLOAT16:
      return ncclBfloat16;
    case DType::FLOAT32:
      return ncclFloat32;
  }
  throw InvalidArgumentError("invalid collective dtype", location);
}

void ValidateCollectiveContext(ExecutionContext &context, const NcclCommunicator &communicator,
                               const std::shared_ptr<CommunicatorGroupState> &state, std::source_location location) {
  const auto rank = CommunicatorAccess::GetRank(communicator, location);
  if (context.GetDevice() != state->GetDevice(rank)) {
    throw InvalidArgumentError("execution context device does not match the communicator rank", location);
  }
  if (!state->BelongsTo(ContextAccess::GetRuntimeState(context, location))) {
    throw InvalidArgumentError("execution context and communicator belong to different runtimes", location);
  }
}

void ValidateCollectiveTensorDevice(const Tensor &tensor, Device device, std::string_view role,
                                    std::source_location location) {
  if (tensor.GetDevice() == device) {
    return;
  }
  std::string message{"collective "};
  message.append(role);
  message.append(" tensor is on the wrong CUDA device");
  throw InvalidArgumentError(std::move(message), location);
}

auto CollectiveByteOffset(const void *pointer, size_t offset) noexcept -> const void * {
  return static_cast<const std::byte *>(pointer) + offset;
}

auto MutableCollectiveByteOffset(void *pointer, size_t offset) noexcept -> void * {
  return static_cast<std::byte *>(pointer) + offset;
}

namespace {

[[nodiscard]] auto CreateScratchTensor(ExecutionContext &context, ScratchArena::Scope &scratch_scope,
                                       const Tensor &prototype, std::source_location location) -> Tensor {
  const auto element_size = GetDTypeSize(prototype.GetDType(), location);
  const auto bytes = CheckedBytes(prototype.GetNumElements(), element_size, location);
  const auto allocation = scratch_scope.AllocateBytes(bytes, 256, location);
  if (allocation.GetOffsetBytes() % element_size != 0) {
    throw InternalError("collective scratch offset is not aligned to the tensor element size", location);
  }
  const auto &storage = ContextAccess::GetPrimaryLane(context, location).GetScratchStorage();
  if (storage == nullptr) {
    throw InternalError("collective scratch allocation did not create backing storage", location);
  }
  return TensorFactory::Create(
      storage, prototype.GetDType(), prototype.GetShape(), GetContiguousStrides(prototype.GetShape(), location),
      CheckedNarrow<int64_t>(allocation.GetOffsetBytes() / element_size, "collective scratch element offset", location),
      location);
}

}  // namespace

auto PrepareCollectiveCall(ExecutionContext &context, Tensor &output, const Tensor &input, std::string_view operation,
                           bool pack_input, bool unpack_output, std::source_location location)
    -> PreparedCollectiveCall {
  PreparedCollectiveCall call{
      .context_ = &context,
      .output_ = &output,
      .input_ = &input,
      .scratch_scope_ = std::nullopt,
      .contiguous_input_ = std::nullopt,
      .contiguous_output_ = std::nullopt,
  };
  if (!pack_input && !unpack_output) {
    return call;
  }
  {
    OpGuard guard{context, operation, location, CapturePolicy::SAFE};
    call.scratch_scope_.emplace(guard.MakeScratchScope());
    if (pack_input) {
      call.contiguous_input_.emplace(CreateScratchTensor(context, *call.scratch_scope_, input, location));
    }
    if (unpack_output) {
      call.contiguous_output_.emplace(CreateScratchTensor(context, *call.scratch_scope_, output, location));
    }
  }
  if (pack_input) {
    CopyOut(context, *call.contiguous_input_, input, location);
  }
  return call;
}

void CompleteCollectiveCall(PreparedCollectiveCall &call, std::source_location location) {
  if (call.contiguous_output_.has_value()) {
    CopyOut(*call.context_, *call.output_, *call.contiguous_output_, location);
  }
}

void CompleteCollectiveCallOrFail(PreparedCollectiveCall &call, const std::shared_ptr<CommunicatorGroupState> &state,
                                  std::source_location location) {
  try {
    CompleteCollectiveCall(call, location);
  } catch (...) {
    state->MarkFailed();
    throw;
  }
}

void ReportCollectiveCleanupErrors(ExecutionContext &context, ncclResult_t group_end_status,
                                   std::optional<cudaError_t> restore_device_status,
                                   std::string_view group_end_operation, std::string_view restore_device_operation,
                                   std::source_location location) noexcept {
  try {
    const ErrorReportContext report_context{
        .location_ = location,
        .device_ = std::nullopt,
        .stream_id_ = std::nullopt,
    };
    auto &error_sink = *ContextAccess::GetErrorSink(context, location);
    static_cast<void>(TryNccl(group_end_status, group_end_operation, error_sink, report_context));
    if (restore_device_status.has_value()) {
      static_cast<void>(TryCuda(*restore_device_status, restore_device_operation, error_sink, report_context));
    }
  } catch (...) {
    return;
  }
}

}  // namespace ttl::internal
