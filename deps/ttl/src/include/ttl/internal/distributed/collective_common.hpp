#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <source_location>
#include <string_view>

#include <driver_types.h>
#include <nccl.h>

#include "ttl/common/device.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

class ExecutionContext;
class NcclCommunicator;

}  // namespace ttl

namespace ttl::internal {

class CommunicatorGroupState;

struct PreparedCollectiveCall final {
  ExecutionContext *context_;
  Tensor *output_;
  const Tensor *input_;
  std::optional<ScratchArena::Scope> scratch_scope_;
  std::optional<Tensor> contiguous_input_;
  std::optional<Tensor> contiguous_output_;

  [[nodiscard]] auto GetInput() const noexcept -> const Tensor &;
  [[nodiscard]] auto GetOutput() noexcept -> Tensor &;
  [[nodiscard]] auto GetOutput() const noexcept -> const Tensor &;
};

[[nodiscard]] auto ToNcclDType(DType dtype, std::source_location location) -> ncclDataType_t;
void ValidateCollectiveContext(ExecutionContext &context, const NcclCommunicator &communicator,
                               const std::shared_ptr<CommunicatorGroupState> &state, std::source_location location);
void ValidateCollectiveTensorDevice(const Tensor &tensor, Device device, std::string_view role,
                                    std::source_location location);
[[nodiscard]] auto CollectiveByteOffset(const void *pointer, size_t offset) noexcept -> const void *;
[[nodiscard]] auto MutableCollectiveByteOffset(void *pointer, size_t offset) noexcept -> void *;
[[nodiscard]] auto PrepareCollectiveCall(ExecutionContext &context, Tensor &output, const Tensor &input,
                                         std::string_view operation, bool pack_input, bool unpack_output,
                                         std::source_location location) -> PreparedCollectiveCall;
void CompleteCollectiveCall(PreparedCollectiveCall &call, std::source_location location);
void CompleteCollectiveCallOrFail(PreparedCollectiveCall &call, const std::shared_ptr<CommunicatorGroupState> &state,
                                  std::source_location location);
void ReportCollectiveCleanupErrors(ExecutionContext &context, ncclResult_t group_end_status,
                                   std::optional<cudaError_t> restore_device_status,
                                   std::string_view group_end_operation, std::string_view restore_device_operation,
                                   std::source_location location) noexcept;

}  // namespace ttl::internal
