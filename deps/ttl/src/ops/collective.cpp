#include "ttl/ops/collective.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <nccl.h>

#include "ttl/communicator.hpp"
#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/communicator.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/internal/nccl_api.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/runtime.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/stream.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

enum class CollectiveKind : uint8_t {
  ALL_REDUCE,
  REDUCE,
  ALL_GATHER,
  REDUCE_SCATTER,
  BROADCAST,
  ALL_TO_ALL,
  GATHER,
  SCATTER,
};

struct PreparedCall final {
  ExecutionContext *context_;
  Tensor *output_;
  const Tensor *input_;
  NcclCommunicator *communicator_;
  std::optional<Tensor> contiguous_input_;
  std::optional<Tensor> contiguous_output_;

  [[nodiscard]] auto GetInput() const noexcept -> const Tensor & {
    return contiguous_input_.has_value() ? *contiguous_input_ : *input_;
  }

  [[nodiscard]] auto GetOutput() noexcept -> Tensor & {
    return contiguous_output_.has_value() ? *contiguous_output_ : *output_;
  }

  [[nodiscard]] auto GetOutput() const noexcept -> const Tensor & {
    return contiguous_output_.has_value() ? *contiguous_output_ : *output_;
  }
};

[[nodiscard]] constexpr auto GetCollectiveName(CollectiveKind kind) noexcept -> std::string_view {
  switch (kind) {
    case CollectiveKind::ALL_REDUCE:
      return "AllReduceOut";
    case CollectiveKind::REDUCE:
      return "ReduceOut";
    case CollectiveKind::ALL_GATHER:
      return "AllGatherOut";
    case CollectiveKind::REDUCE_SCATTER:
      return "ReduceScatterOut";
    case CollectiveKind::BROADCAST:
      return "BroadcastOut";
    case CollectiveKind::ALL_TO_ALL:
      return "AllToAllOut";
    case CollectiveKind::GATHER:
      return "GatherOut";
    case CollectiveKind::SCATTER:
      return "ScatterOut";
  }
  return "Collective";
}

[[nodiscard]] constexpr auto UsesReduction(CollectiveKind kind) noexcept -> bool {
  return kind == CollectiveKind::ALL_REDUCE || kind == CollectiveKind::REDUCE || kind == CollectiveKind::REDUCE_SCATTER;
}

[[nodiscard]] constexpr auto UsesPointToPoint(CollectiveKind kind) noexcept -> bool {
  return kind == CollectiveKind::ALL_TO_ALL || kind == CollectiveKind::GATHER || kind == CollectiveKind::SCATTER;
}

[[nodiscard]] constexpr auto ReadsInput(CollectiveKind kind, size_t rank, size_t root) noexcept -> bool {
  return (kind != CollectiveKind::BROADCAST && kind != CollectiveKind::SCATTER) || rank == root;
}

[[nodiscard]] constexpr auto WritesOutput(CollectiveKind kind, size_t rank, size_t root) noexcept -> bool {
  return (kind != CollectiveKind::REDUCE && kind != CollectiveKind::GATHER) || rank == root;
}

[[nodiscard]] auto ToNcclDType(DType dtype, std::source_location location) -> ncclDataType_t {
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

[[nodiscard]] auto ToNcclReduceOp(ReduceOp operation, DType dtype, std::source_location location) -> ncclRedOp_t {
  switch (operation) {
    case ReduceOp::SUM:
      if (dtype == DType::BOOL) {
        throw InvalidArgumentError("SUM reduction does not support BOOL tensors", location);
      }
      return ncclSum;
    case ReduceOp::MINIMUM:
      return ncclMin;
    case ReduceOp::MAXIMUM:
      return ncclMax;
  }
  throw InvalidArgumentError("invalid collective reduction operation", location);
}

void ValidateReduceOperation(ReduceOp operation, DType dtype, std::source_location location) {
  switch (operation) {
    case ReduceOp::SUM:
      if (dtype == DType::BOOL) {
        throw InvalidArgumentError("SUM reduction does not support BOOL tensors", location);
      }
      return;
    case ReduceOp::MINIMUM:
    case ReduceOp::MAXIMUM:
      return;
  }
  throw InvalidArgumentError("invalid collective reduction operation", location);
}

void ValidateRoot(int32_t root, size_t world_size, std::source_location location) {
  if (root < 0 || std::cmp_greater_equal(root, world_size)) {
    throw InvalidArgumentError("collective root rank is outside the communicator", location);
  }
}

void ValidatePeer(int32_t peer, size_t rank, size_t world_size, std::source_location location) {
  if (peer < 0 || std::cmp_greater_equal(peer, world_size)) {
    throw InvalidArgumentError("point-to-point peer rank is outside the communicator", location);
  }
  if (std::cmp_equal(peer, rank)) {
    throw InvalidArgumentError("point-to-point peer rank must differ from the local rank", location);
  }
}

void ValidateContext(ExecutionContext &context, const NcclCommunicator &communicator,
                     const std::shared_ptr<internal::CommunicatorGroupState> &state, std::source_location location) {
  const auto rank = internal::CommunicatorAccess::GetRank(communicator, location);
  if (context.GetDevice() != state->GetDevice(rank)) {
    throw InvalidArgumentError("execution context device does not match the communicator rank", location);
  }
  if (!state->BelongsTo(internal::ContextAccess::GetRuntimeState(context, location))) {
    throw InvalidArgumentError("execution context and communicator belong to different runtimes", location);
  }
}

void ValidateTensorDevice(const Tensor &tensor, Device device, std::string_view role, std::source_location location) {
  if (tensor.GetDevice() == device) {
    return;
  }
  std::string message{"collective "};
  message.append(role);
  message.append(" tensor is on the wrong CUDA device");
  throw InvalidArgumentError(std::move(message), location);
}

[[nodiscard]] auto ExpectedScaledCount(int64_t count, size_t world_size, std::source_location location) -> int64_t {
  return internal::CheckedMultiply(count,
                                   internal::CheckedNarrow<int64_t>(world_size, "communicator world size", location),
                                   "collective element count", location);
}

void ValidateShapeAndDType(CollectiveKind kind, const Tensor &output, const Tensor &input, size_t world_size,
                           std::source_location location) {
  if (output.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("collective input and output dtypes must match", location);
  }
  const auto input_count = input.GetNumElements();
  const auto output_count = output.GetNumElements();
  switch (kind) {
    case CollectiveKind::ALL_REDUCE:
    case CollectiveKind::REDUCE:
    case CollectiveKind::BROADCAST:
      if (output.GetShape() != input.GetShape()) {
        throw InvalidArgumentError("collective input and output shapes must match", location);
      }
      return;
    case CollectiveKind::ALL_GATHER:
    case CollectiveKind::GATHER:
      if (output_count != ExpectedScaledCount(input_count, world_size, location)) {
        throw InvalidArgumentError("gather output element count must equal input count times world size", location);
      }
      return;
    case CollectiveKind::REDUCE_SCATTER:
    case CollectiveKind::SCATTER:
      if (input_count != ExpectedScaledCount(output_count, world_size, location)) {
        throw InvalidArgumentError("scatter input element count must equal output count times world size", location);
      }
      return;
    case CollectiveKind::ALL_TO_ALL:
      if (output_count != input_count) {
        throw InvalidArgumentError("AllToAllOut input and output element counts must match", location);
      }
      if (input_count % internal::CheckedNarrow<int64_t>(world_size, "communicator world size", location) != 0) {
        throw InvalidArgumentError("AllToAllOut element count must be divisible by world size", location);
      }
      return;
  }
}

[[nodiscard]] auto ByteOffset(const void *pointer, size_t offset) noexcept -> const void * {
  return static_cast<const std::byte *>(pointer) + offset;
}

[[nodiscard]] auto MutableByteOffset(void *pointer, size_t offset) noexcept -> void * {
  return static_cast<std::byte *>(pointer) + offset;
}

void ValidateAlias(CollectiveKind kind, const Tensor &output, const Tensor &input, size_t rank, size_t world_size,
                   size_t root, std::source_location location) {
  const auto alias = ClassifyAlias(output, input, location);
  if (alias == AliasKind::DISJOINT) {
    return;
  }
  if (!ReadsInput(kind, rank, root) || !WritesOutput(kind, rank, root)) {
    return;
  }
  if ((kind == CollectiveKind::ALL_REDUCE || kind == CollectiveKind::REDUCE || kind == CollectiveKind::BROADCAST) &&
      alias == AliasKind::EXACT) {
    return;
  }
  if (kind == CollectiveKind::ALL_TO_ALL && world_size == 1 && alias == AliasKind::EXACT) {
    return;
  }

  if (!input.IsContiguous() || !output.IsContiguous()) {
    throw InvalidArgumentError("collective in-place layout must be contiguous", location);
  }
  const auto element_size = GetDTypeSize(input.GetDType(), location);
  const auto input_data = internal::TensorAccess::GetData(input, location);
  const auto output_data = internal::TensorAccess::GetData(output, location);
  if (kind == CollectiveKind::ALL_GATHER || (kind == CollectiveKind::GATHER && rank == root)) {
    const auto offset = internal::CheckedBytes(input.GetNumElements(), element_size, location);
    if (input_data ==
        ByteOffset(output_data, internal::CheckedMultiply(rank, offset, "gather rank offset", location))) {
      return;
    }
  }
  if (kind == CollectiveKind::REDUCE_SCATTER || (kind == CollectiveKind::SCATTER && rank == root)) {
    const auto offset = internal::CheckedBytes(output.GetNumElements(), element_size, location);
    if (output_data ==
        ByteOffset(input_data, internal::CheckedMultiply(rank, offset, "scatter rank offset", location))) {
      return;
    }
  }
  throw InvalidArgumentError("collective does not support this input/output alias relationship", location);
}

void ValidateCall(CollectiveKind kind, ExecutionContext &context, Tensor &output, const Tensor &input,
                  NcclCommunicator &communicator, ReduceOp operation, int32_t root, std::source_location location) {
  const auto &state = internal::CommunicatorAccess::GetState(communicator, location);
  const auto rank = internal::CommunicatorAccess::GetRank(communicator, location);
  ValidateContext(context, communicator, state, location);
  ValidateTensorDevice(input, context.GetDevice(), "input", location);
  ValidateTensorDevice(output, context.GetDevice(), "output", location);
  internal::ValidateWritableOutput(output, GetCollectiveName(kind), location);
  ValidateShapeAndDType(kind, output, input, state->GetWorldSize(), location);
  if (UsesReduction(kind)) {
    ValidateReduceOperation(operation, input.GetDType(), location);
  }
  if (kind == CollectiveKind::REDUCE || kind == CollectiveKind::BROADCAST || kind == CollectiveKind::GATHER ||
      kind == CollectiveKind::SCATTER) {
    ValidateRoot(root, state->GetWorldSize(), location);
  }
  ValidateAlias(kind, output, input, rank, state->GetWorldSize(), static_cast<size_t>(root), location);
}

[[nodiscard]] auto PrepareCall(CollectiveKind kind, ExecutionContext &context, Tensor &output, const Tensor &input,
                               NcclCommunicator &communicator, int32_t root, std::source_location location)
    -> PreparedCall {
  const auto rank = internal::CommunicatorAccess::GetRank(communicator, location);
  PreparedCall call{
      .context_ = &context,
      .output_ = &output,
      .input_ = &input,
      .communicator_ = &communicator,
      .contiguous_input_ = std::nullopt,
      .contiguous_output_ = std::nullopt,
  };
  if (ReadsInput(kind, rank, static_cast<size_t>(root)) && !input.IsContiguous()) {
    call.contiguous_input_.emplace(Contiguous(context, input, location));
  }
  if (WritesOutput(kind, rank, static_cast<size_t>(root)) && !output.IsContiguous()) {
    call.contiguous_output_.emplace(Empty(context, output.GetShape(), output.GetDType(), location));
  }
  return call;
}

void CompleteCall(PreparedCall &call, std::source_location location) {
  if (call.contiguous_output_.has_value()) {
    CopyOut(*call.context_, *call.output_, *call.contiguous_output_, location);
  }
}

void RecordCall(internal::OpGuard &guard, CollectiveKind kind, PreparedCall &call, size_t rank, size_t root) {
  if (ReadsInput(kind, rank, root)) {
    guard.RecordTensor(call.GetInput());
  }
  if (WritesOutput(kind, rank, root)) {
    guard.RecordTensor(call.GetOutput());
  }
}

[[nodiscard]] auto GetChunkCount(CollectiveKind kind, const PreparedCall &call, size_t world_size) noexcept -> size_t {
  switch (kind) {
    case CollectiveKind::ALL_TO_ALL:
      return static_cast<size_t>(call.GetInput().GetNumElements()) / world_size;
    case CollectiveKind::GATHER:
      return static_cast<size_t>(call.GetInput().GetNumElements());
    case CollectiveKind::SCATTER:
      return static_cast<size_t>(call.GetOutput().GetNumElements());
    default:
      return 0;
  }
}

void SubmitSelfCopy(CollectiveKind kind, PreparedCall &call, size_t rank, size_t world_size, cudaStream_t stream,
                    std::source_location location) {
  if (!UsesPointToPoint(kind) || call.GetInput().GetNumElements() == 0) {
    return;
  }
  const auto chunk_count = GetChunkCount(kind, call, world_size);
  const auto chunk_bytes = internal::CheckedMultiply(chunk_count, GetDTypeSize(call.GetInput().GetDType(), location),
                                                     "collective self-copy byte count", location);
  const auto offset = internal::CheckedMultiply(rank, chunk_bytes, "collective self-copy offset", location);
  const void *source = nullptr;
  void *destination = nullptr;
  switch (kind) {
    case CollectiveKind::ALL_TO_ALL:
      source = ByteOffset(internal::TensorAccess::GetData(call.GetInput(), location), offset);
      destination = MutableByteOffset(internal::TensorAccess::GetMutableData(call.GetOutput(), location), offset);
      break;
    case CollectiveKind::GATHER:
      source = internal::TensorAccess::GetData(call.GetInput(), location);
      destination = MutableByteOffset(internal::TensorAccess::GetMutableData(call.GetOutput(), location), offset);
      break;
    case CollectiveKind::SCATTER:
      source = ByteOffset(internal::TensorAccess::GetData(call.GetInput(), location), offset);
      destination = internal::TensorAccess::GetMutableData(call.GetOutput(), location);
      break;
    default:
      return;
  }
  if (source == destination || chunk_bytes == 0) {
    return;
  }
  internal::CheckCuda(
      internal::GetCudaApi().memcpy_async_(destination, source, chunk_bytes, cudaMemcpyDeviceToDevice, stream),
      "cudaMemcpyAsync (collective self copy)", location);
}

void IssueCollective(CollectiveKind kind, PreparedCall &call, ncclComm_t communicator, size_t rank, size_t world_size,
                     size_t root, ReduceOp operation, cudaStream_t stream, std::vector<ncclResult_t> &statuses,
                     std::source_location location) {
  const auto &nccl_api = internal::GetNcclApi();
  const auto dtype = ToNcclDType(call.GetInput().GetDType(), location);
  const auto input = internal::TensorAccess::GetData(call.GetInput(), location);
  auto *output = internal::TensorAccess::GetMutableData(call.GetOutput(), location);
  const auto input_count = static_cast<size_t>(call.GetInput().GetNumElements());
  const auto output_count = static_cast<size_t>(call.GetOutput().GetNumElements());
  switch (kind) {
    case CollectiveKind::ALL_REDUCE:
      statuses.push_back(nccl_api.all_reduce_(input, output, input_count, dtype,
                                              ToNcclReduceOp(operation, call.GetInput().GetDType(), location),
                                              communicator, stream));
      return;
    case CollectiveKind::REDUCE:
      statuses.push_back(nccl_api.reduce_(input, output, input_count, dtype,
                                          ToNcclReduceOp(operation, call.GetInput().GetDType(), location),
                                          static_cast<int>(root), communicator, stream));
      return;
    case CollectiveKind::ALL_GATHER:
      statuses.push_back(nccl_api.all_gather_(input, output, input_count, dtype, communicator, stream));
      return;
    case CollectiveKind::REDUCE_SCATTER:
      statuses.push_back(nccl_api.reduce_scatter_(input, output, output_count, dtype,
                                                  ToNcclReduceOp(operation, call.GetInput().GetDType(), location),
                                                  communicator, stream));
      return;
    case CollectiveKind::BROADCAST:
      statuses.push_back(
          nccl_api.broadcast_(input, output, input_count, dtype, static_cast<int>(root), communicator, stream));
      return;
    case CollectiveKind::ALL_TO_ALL: {
      const auto chunk_count = input_count / world_size;
      const auto chunk_bytes = internal::CheckedMultiply(
          chunk_count, GetDTypeSize(call.GetInput().GetDType(), location), "AllToAllOut chunk bytes", location);
      for (size_t peer = 0; peer < world_size; peer++) {
        if (peer == rank) {
          continue;
        }
        statuses.push_back(nccl_api.send_(ByteOffset(input, peer * chunk_bytes), chunk_count, dtype,
                                          static_cast<int>(peer), communicator, stream));
        statuses.push_back(nccl_api.receive_(MutableByteOffset(output, peer * chunk_bytes), chunk_count, dtype,
                                             static_cast<int>(peer), communicator, stream));
      }
      return;
    }
    case CollectiveKind::GATHER:
      if (rank == root) {
        const auto chunk_bytes = internal::CheckedBytes(call.GetInput().GetNumElements(),
                                                        GetDTypeSize(call.GetInput().GetDType(), location), location);
        for (size_t peer = 0; peer < world_size; peer++) {
          if (peer != rank) {
            statuses.push_back(nccl_api.receive_(MutableByteOffset(output, peer * chunk_bytes), input_count, dtype,
                                                 static_cast<int>(peer), communicator, stream));
          }
        }
      } else {
        statuses.push_back(nccl_api.send_(input, input_count, dtype, static_cast<int>(root), communicator, stream));
      }
      return;
    case CollectiveKind::SCATTER:
      if (rank == root) {
        const auto chunk_bytes = internal::CheckedBytes(call.GetOutput().GetNumElements(),
                                                        GetDTypeSize(call.GetOutput().GetDType(), location), location);
        for (size_t peer = 0; peer < world_size; peer++) {
          if (peer != rank) {
            statuses.push_back(nccl_api.send_(ByteOffset(input, peer * chunk_bytes), output_count, dtype,
                                              static_cast<int>(peer), communicator, stream));
          }
        }
      } else {
        statuses.push_back(
            nccl_api.receive_(output, output_count, dtype, static_cast<int>(root), communicator, stream));
      }
      return;
  }
}

void SubmitRank(CollectiveKind kind, ExecutionContext &context, Tensor &output, const Tensor &input,
                NcclCommunicator &communicator, ReduceOp operation, int32_t root, std::source_location location) {
  ValidateCall(kind, context, output, input, communicator, operation, root, location);
  auto call = PrepareCall(kind, context, output, input, communicator, root, location);
  const auto &state = internal::CommunicatorAccess::GetState(communicator, location);
  const auto rank = internal::CommunicatorAccess::GetRank(communicator, location);
  if (input.GetNumElements() != 0) {
    {
      internal::OpGuard guard{context, GetCollectiveName(kind), location};
      auto lease = state->AcquireRank(rank, location);
      RecordCall(guard, kind, call, rank, static_cast<size_t>(root));
      SubmitSelfCopy(kind, call, rank, state->GetWorldSize(), guard.GetNativeStream(), location);

      std::vector<ncclResult_t> statuses;
      if (UsesPointToPoint(kind)) {
        statuses.reserve((state->GetWorldSize() - 1) * 2);
        internal::CheckNccl(internal::GetNcclApi().group_start_(), "ncclGroupStart", location);
      } else {
        statuses.reserve(1);
      }
      try {
        IssueCollective(kind, call, lease.GetHandle(rank), rank, state->GetWorldSize(), static_cast<size_t>(root),
                        operation, guard.GetNativeStream(), statuses, location);
      } catch (...) {
        if (UsesPointToPoint(kind)) {
          internal::GetNcclApi().group_end_();
          state->MarkFailed();
        }
        throw;
      }
      const auto end_status = UsesPointToPoint(kind) ? internal::GetNcclApi().group_end_() : ncclSuccess;
      state->CheckGroupedSubmission(lease.GetRanks(), statuses, end_status, GetCollectiveName(kind), location);
    }
  }
  CompleteCall(call, location);
}

[[nodiscard]] auto ValidateLocalCalls(CollectiveKind kind, std::span<const LocalCollectiveCall> calls,
                                      ReduceOp operation, int32_t root, std::source_location location)
    -> std::shared_ptr<internal::CommunicatorGroupState> {
  if (calls.empty()) {
    throw InvalidArgumentError("local collective call list must not be empty", location);
  }
  if (calls.front().context_ == nullptr || calls.front().output_ == nullptr || calls.front().input_ == nullptr ||
      calls.front().communicator_ == nullptr) {
    throw InvalidArgumentError("local collective call fields must not be null", location);
  }
  auto state = internal::CommunicatorAccess::GetState(*calls.front().communicator_, location);
  if (calls.size() != state->GetWorldSize()) {
    throw InvalidArgumentError("local collective requires exactly one call for every communicator rank", location);
  }
  const auto input_dtype = calls.front().input_->GetDType();
  const auto input_count = calls.front().input_->GetNumElements();
  const auto output_count = calls.front().output_->GetNumElements();
  for (size_t rank = 0; rank < calls.size(); rank++) {
    const auto &call = calls[rank];
    if (call.context_ == nullptr || call.output_ == nullptr || call.input_ == nullptr ||
        call.communicator_ == nullptr) {
      throw InvalidArgumentError("local collective call fields must not be null", location);
    }
    if (internal::CommunicatorAccess::GetState(*call.communicator_, location).get() != state.get() ||
        internal::CommunicatorAccess::GetRank(*call.communicator_, location) != rank) {
      throw InvalidArgumentError("local collective calls must be ordered by rank from one group", location);
    }
    ValidateCall(kind, *call.context_, *call.output_, *call.input_, *call.communicator_, operation, root, location);
    if (call.input_->GetDType() != input_dtype || call.input_->GetNumElements() != input_count ||
        call.output_->GetNumElements() != output_count) {
      throw InvalidArgumentError("all local collective ranks must use the same dtype and element counts", location);
    }
  }
  return state;
}

void SubmitLocal(CollectiveKind kind, std::span<const LocalCollectiveCall> calls, ReduceOp operation, int32_t root,
                 std::source_location location) {
  auto state = ValidateLocalCalls(kind, calls, operation, root, location);
  std::vector<PreparedCall> prepared;
  prepared.reserve(calls.size());
  for (const auto &call : calls) {
    prepared.push_back(
        PrepareCall(kind, *call.context_, *call.output_, *call.input_, *call.communicator_, root, location));
  }

  if (calls.front().input_->GetNumElements() != 0) {
    std::vector<std::unique_ptr<internal::OpGuard>> guards;
    guards.reserve(calls.size());
    for (const auto &call : calls) {
      guards.push_back(std::make_unique<internal::OpGuard>(*call.context_, GetCollectiveName(kind), location));
    }
    auto lease = state->AcquireAll(location);
    for (size_t rank = 0; rank < calls.size(); rank++) {
      RecordCall(*guards[rank], kind, prepared[rank], rank, static_cast<size_t>(root));
      SubmitSelfCopy(kind, prepared[rank], rank, calls.size(), guards[rank]->GetNativeStream(), location);
    }

    const auto &cuda_api = internal::GetCudaApi();
    int previous_device = -1;
    internal::CheckCuda(cuda_api.get_device_(&previous_device), "cudaGetDevice (local collective)", location);
    std::vector<ncclResult_t> statuses;
    const auto per_rank = UsesPointToPoint(kind) ? (calls.size() - 1) * 2 : size_t{1};
    statuses.reserve(internal::CheckedMultiply(calls.size(), per_rank, "grouped NCCL call count", location));
    internal::CheckNccl(internal::GetNcclApi().group_start_(), "ncclGroupStart", location);
    cudaError_t set_device_status = cudaSuccess;
    try {
      for (size_t rank = 0; rank < calls.size(); rank++) {
        set_device_status = cuda_api.set_device_(state->GetDevice(rank).GetOrdinal());
        if (set_device_status != cudaSuccess) {
          break;
        }
        IssueCollective(kind, prepared[rank], lease.GetHandle(rank), rank, calls.size(), static_cast<size_t>(root),
                        operation, guards[rank]->GetNativeStream(), statuses, location);
      }
    } catch (...) {
      internal::GetNcclApi().group_end_();
      cuda_api.set_device_(previous_device);
      state->MarkFailed();
      throw;
    }
    const auto end_status = internal::GetNcclApi().group_end_();
    const auto restore_status = cuda_api.set_device_(previous_device);
    if (set_device_status != cudaSuccess || restore_status != cudaSuccess) {
      state->MarkFailed();
    }
    internal::CheckCuda(set_device_status, "cudaSetDevice (local collective)", location);
    internal::CheckCuda(restore_status, "cudaSetDevice (restore after local collective)", location);
    state->CheckGroupedSubmission(lease.GetRanks(), statuses, end_status, GetCollectiveName(kind), location);
    guards.clear();
  }

  for (auto &call : prepared) {
    CompleteCall(call, location);
  }
}

void SubmitPointToPoint(ExecutionContext &context, const Tensor *send, int32_t send_peer, Tensor *receive,
                        int32_t receive_peer, NcclCommunicator &communicator, std::source_location location) {
  const auto &state = internal::CommunicatorAccess::GetState(communicator, location);
  const auto rank = internal::CommunicatorAccess::GetRank(communicator, location);
  ValidateContext(context, communicator, state, location);
  if (send == nullptr && receive == nullptr) {
    throw InvalidArgumentError("point-to-point submission must send or receive a tensor", location);
  }
  if (send != nullptr) {
    ValidatePeer(send_peer, rank, state->GetWorldSize(), location);
    ValidateTensorDevice(*send, context.GetDevice(), "send", location);
    if (!send->IsContiguous()) {
      throw InvalidArgumentError("Send requires a contiguous tensor", location);
    }
  }
  if (receive != nullptr) {
    ValidatePeer(receive_peer, rank, state->GetWorldSize(), location);
    ValidateTensorDevice(*receive, context.GetDevice(), "receive", location);
    internal::ValidateWritableOutput(*receive, "ReceiveOut", location);
    if (!receive->IsContiguous()) {
      throw InvalidArgumentError("ReceiveOut requires a contiguous tensor", location);
    }
  }
  if (send != nullptr && receive != nullptr && ClassifyAlias(*send, *receive, location) != AliasKind::DISJOINT) {
    throw InvalidArgumentError("simultaneous send and receive tensors must not overlap", location);
  }

  internal::OpGuard guard{context, send != nullptr && receive != nullptr ? "SendReceive" : "PointToPoint", location};
  auto lease = state->AcquireRank(rank, location);
  std::vector<ncclResult_t> statuses;
  statuses.reserve(2);
  const auto grouped = send != nullptr && receive != nullptr;
  if (send != nullptr) {
    guard.RecordTensor(*send);
  }
  if (receive != nullptr) {
    guard.RecordTensor(*receive);
  }
  if (grouped) {
    internal::CheckNccl(internal::GetNcclApi().group_start_(), "ncclGroupStart", location);
  }
  try {
    if (send != nullptr) {
      statuses.push_back(internal::GetNcclApi().send_(
          internal::TensorAccess::GetData(*send, location), static_cast<size_t>(send->GetNumElements()),
          ToNcclDType(send->GetDType(), location), send_peer, lease.GetHandle(rank), guard.GetNativeStream()));
    }
    if (receive != nullptr) {
      statuses.push_back(internal::GetNcclApi().receive_(
          internal::TensorAccess::GetMutableData(*receive, location), static_cast<size_t>(receive->GetNumElements()),
          ToNcclDType(receive->GetDType(), location), receive_peer, lease.GetHandle(rank), guard.GetNativeStream()));
    }
  } catch (...) {
    if (grouped) {
      internal::GetNcclApi().group_end_();
      state->MarkFailed();
    }
    throw;
  }
  const auto end_status = grouped ? internal::GetNcclApi().group_end_() : ncclSuccess;
  state->CheckGroupedSubmission(lease.GetRanks(), statuses, end_status, "NCCL point-to-point", location);
}

}  // namespace

void AllReduceOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                  ReduceOp operation, std::source_location location) {
  SubmitRank(CollectiveKind::ALL_REDUCE, context, output, input, communicator, operation, 0, location);
}

void ReduceOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
               ReduceOp operation, int32_t root, std::source_location location) {
  SubmitRank(CollectiveKind::REDUCE, context, output, input, communicator, operation, root, location);
}

void AllGatherOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                  std::source_location location) {
  SubmitRank(CollectiveKind::ALL_GATHER, context, output, input, communicator, ReduceOp::SUM, 0, location);
}

void ReduceScatterOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                      ReduceOp operation, std::source_location location) {
  SubmitRank(CollectiveKind::REDUCE_SCATTER, context, output, input, communicator, operation, 0, location);
}

void BroadcastOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                  int32_t root, std::source_location location) {
  SubmitRank(CollectiveKind::BROADCAST, context, output, input, communicator, ReduceOp::SUM, root, location);
}

void AllToAllOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                 std::source_location location) {
  SubmitRank(CollectiveKind::ALL_TO_ALL, context, output, input, communicator, ReduceOp::SUM, 0, location);
}

void GatherOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
               int32_t root, std::source_location location) {
  SubmitRank(CollectiveKind::GATHER, context, output, input, communicator, ReduceOp::SUM, root, location);
}

void ScatterOut(ExecutionContext &context, Tensor &output, const Tensor &input, NcclCommunicator &communicator,
                int32_t root, std::source_location location) {
  SubmitRank(CollectiveKind::SCATTER, context, output, input, communicator, ReduceOp::SUM, root, location);
}

void Send(ExecutionContext &context, const Tensor &input, NcclCommunicator &communicator, int32_t peer,
          std::source_location location) {
  SubmitPointToPoint(context, &input, peer, nullptr, 0, communicator, location);
}

void ReceiveOut(ExecutionContext &context, Tensor &output, NcclCommunicator &communicator, int32_t peer,
                std::source_location location) {
  SubmitPointToPoint(context, nullptr, 0, &output, peer, communicator, location);
}

void SendReceiveOut(ExecutionContext &context, const Tensor &send, int32_t send_peer, Tensor &receive,
                    int32_t receive_peer, NcclCommunicator &communicator, std::source_location location) {
  SubmitPointToPoint(context, &send, send_peer, &receive, receive_peer, communicator, location);
}

void Barrier(ExecutionContext &context, NcclCommunicator &communicator, std::source_location location) {
  const auto &state = internal::CommunicatorAccess::GetState(communicator, location);
  const auto rank = internal::CommunicatorAccess::GetRank(communicator, location);
  ValidateContext(context, communicator, state, location);
  internal::OpGuard guard{context, "Barrier", location};
  auto lease = state->AcquireRank(rank, location);
  state->BeginBarrier(rank, guard.GetStream(), location);
  const auto &storage = state->GetBarrierStorage(rank);
  const auto status =
      internal::GetNcclApi().all_reduce_(storage->GetBasePointer(), storage->GetBasePointer(), 1, ncclUint8, ncclMax,
                                         lease.GetHandle(rank), guard.GetNativeStream());
  state->EndBarrier(rank, guard.GetStream(), location);
  state->CheckSubmission(lease.GetRanks(), status, "NCCL barrier", location);
}

void AllReduceLocal(std::span<const LocalCollectiveCall> calls, ReduceOp operation, std::source_location location) {
  SubmitLocal(CollectiveKind::ALL_REDUCE, calls, operation, 0, location);
}

void ReduceLocal(std::span<const LocalCollectiveCall> calls, ReduceOp operation, int32_t root,
                 std::source_location location) {
  SubmitLocal(CollectiveKind::REDUCE, calls, operation, root, location);
}

void AllGatherLocal(std::span<const LocalCollectiveCall> calls, std::source_location location) {
  SubmitLocal(CollectiveKind::ALL_GATHER, calls, ReduceOp::SUM, 0, location);
}

void ReduceScatterLocal(std::span<const LocalCollectiveCall> calls, ReduceOp operation, std::source_location location) {
  SubmitLocal(CollectiveKind::REDUCE_SCATTER, calls, operation, 0, location);
}

void BroadcastLocal(std::span<const LocalCollectiveCall> calls, int32_t root, std::source_location location) {
  SubmitLocal(CollectiveKind::BROADCAST, calls, ReduceOp::SUM, root, location);
}

void AllToAllLocal(std::span<const LocalCollectiveCall> calls, std::source_location location) {
  SubmitLocal(CollectiveKind::ALL_TO_ALL, calls, ReduceOp::SUM, 0, location);
}

void GatherLocal(std::span<const LocalCollectiveCall> calls, int32_t root, std::source_location location) {
  SubmitLocal(CollectiveKind::GATHER, calls, ReduceOp::SUM, root, location);
}

void ScatterLocal(std::span<const LocalCollectiveCall> calls, int32_t root, std::source_location location) {
  SubmitLocal(CollectiveKind::SCATTER, calls, ReduceOp::SUM, root, location);
}

void SendReceiveLocal(std::span<const LocalPointToPointCall> calls, std::source_location location) {
  if (calls.empty()) {
    throw InvalidArgumentError("local point-to-point call list must not be empty", location);
  }
  if (calls.front().communicator_ == nullptr) {
    throw InvalidArgumentError("local point-to-point communicator must not be null", location);
  }
  auto state = internal::CommunicatorAccess::GetState(*calls.front().communicator_, location);
  if (calls.size() != state->GetWorldSize()) {
    throw InvalidArgumentError("local point-to-point requires one call for every communicator rank", location);
  }
  for (size_t rank = 0; rank < calls.size(); rank++) {
    const auto &call = calls[rank];
    if (call.context_ == nullptr || call.communicator_ == nullptr ||
        (call.send_ == nullptr && call.receive_ == nullptr)) {
      throw InvalidArgumentError("local point-to-point call is incomplete", location);
    }
    if (internal::CommunicatorAccess::GetState(*call.communicator_, location).get() != state.get() ||
        internal::CommunicatorAccess::GetRank(*call.communicator_, location) != rank) {
      throw InvalidArgumentError("local point-to-point calls must be ordered by rank from one group", location);
    }
    ValidateContext(*call.context_, *call.communicator_, state, location);
    if (call.send_ != nullptr) {
      ValidatePeer(call.send_peer_, rank, calls.size(), location);
      ValidateTensorDevice(*call.send_, call.context_->GetDevice(), "send", location);
      if (!call.send_->IsContiguous()) {
        throw InvalidArgumentError("local point-to-point send tensor must be contiguous", location);
      }
      const auto &peer_call = calls[static_cast<size_t>(call.send_peer_)];
      if (peer_call.receive_ == nullptr || !std::cmp_equal(peer_call.receive_peer_, rank) ||
          peer_call.receive_->GetDType() != call.send_->GetDType() ||
          peer_call.receive_->GetNumElements() != call.send_->GetNumElements()) {
        throw InvalidArgumentError("local point-to-point send has no matching receive", location);
      }
    }
    if (call.receive_ != nullptr) {
      ValidatePeer(call.receive_peer_, rank, calls.size(), location);
      ValidateTensorDevice(*call.receive_, call.context_->GetDevice(), "receive", location);
      internal::ValidateWritableOutput(*call.receive_, "SendReceiveLocal", location);
      if (!call.receive_->IsContiguous()) {
        throw InvalidArgumentError("local point-to-point receive tensor must be contiguous", location);
      }
    }
    if (call.send_ != nullptr && call.receive_ != nullptr &&
        ClassifyAlias(*call.send_, *call.receive_, location) != AliasKind::DISJOINT) {
      throw InvalidArgumentError("local send and receive tensors must not overlap", location);
    }
  }

  std::vector<std::unique_ptr<internal::OpGuard>> guards;
  guards.reserve(calls.size());
  for (const auto &call : calls) {
    guards.push_back(std::make_unique<internal::OpGuard>(*call.context_, "SendReceiveLocal", location));
  }
  auto lease = state->AcquireAll(location);
  for (size_t rank = 0; rank < calls.size(); rank++) {
    if (calls[rank].send_ != nullptr) {
      guards[rank]->RecordTensor(*calls[rank].send_);
    }
    if (calls[rank].receive_ != nullptr) {
      guards[rank]->RecordTensor(*calls[rank].receive_);
    }
  }
  const auto &cuda_api = internal::GetCudaApi();
  int previous_device = -1;
  internal::CheckCuda(cuda_api.get_device_(&previous_device), "cudaGetDevice (local point-to-point)", location);
  std::vector<ncclResult_t> statuses;
  statuses.reserve(calls.size() * 2);
  internal::CheckNccl(internal::GetNcclApi().group_start_(), "ncclGroupStart", location);
  cudaError_t set_device_status = cudaSuccess;
  try {
    for (size_t rank = 0; rank < calls.size(); rank++) {
      const auto &call = calls[rank];
      set_device_status = cuda_api.set_device_(state->GetDevice(rank).GetOrdinal());
      if (set_device_status != cudaSuccess) {
        break;
      }
      if (call.send_ != nullptr) {
        statuses.push_back(internal::GetNcclApi().send_(internal::TensorAccess::GetData(*call.send_, location),
                                                        static_cast<size_t>(call.send_->GetNumElements()),
                                                        ToNcclDType(call.send_->GetDType(), location), call.send_peer_,
                                                        lease.GetHandle(rank), guards[rank]->GetNativeStream()));
      }
      if (call.receive_ != nullptr) {
        statuses.push_back(internal::GetNcclApi().receive_(
            internal::TensorAccess::GetMutableData(*call.receive_, location),
            static_cast<size_t>(call.receive_->GetNumElements()), ToNcclDType(call.receive_->GetDType(), location),
            call.receive_peer_, lease.GetHandle(rank), guards[rank]->GetNativeStream()));
      }
    }
  } catch (...) {
    internal::GetNcclApi().group_end_();
    cuda_api.set_device_(previous_device);
    state->MarkFailed();
    throw;
  }
  const auto end_status = internal::GetNcclApi().group_end_();
  const auto restore_status = cuda_api.set_device_(previous_device);
  if (set_device_status != cudaSuccess || restore_status != cudaSuccess) {
    state->MarkFailed();
  }
  internal::CheckCuda(set_device_status, "cudaSetDevice (local point-to-point)", location);
  internal::CheckCuda(restore_status, "cudaSetDevice (restore after local point-to-point)", location);
  state->CheckGroupedSubmission(lease.GetRanks(), statuses, end_status, "NCCL point-to-point", location);
}

void BarrierLocal(std::span<const LocalBarrierCall> calls, std::source_location location) {
  if (calls.empty()) {
    throw InvalidArgumentError("local barrier call list must not be empty", location);
  }
  if (calls.front().communicator_ == nullptr) {
    throw InvalidArgumentError("local barrier communicator must not be null", location);
  }
  auto state = internal::CommunicatorAccess::GetState(*calls.front().communicator_, location);
  if (calls.size() != state->GetWorldSize()) {
    throw InvalidArgumentError("local barrier requires one call for every communicator rank", location);
  }

  std::vector<std::unique_ptr<internal::OpGuard>> guards;
  guards.reserve(calls.size());
  for (size_t rank = 0; rank < calls.size(); rank++) {
    const auto &call = calls[rank];
    if (call.context_ == nullptr || call.communicator_ == nullptr ||
        internal::CommunicatorAccess::GetState(*call.communicator_, location).get() != state.get() ||
        internal::CommunicatorAccess::GetRank(*call.communicator_, location) != rank) {
      throw InvalidArgumentError("local barrier calls must be ordered by rank from one group", location);
    }
    ValidateContext(*call.context_, *call.communicator_, state, location);
    guards.push_back(std::make_unique<internal::OpGuard>(*call.context_, "BarrierLocal", location));
  }

  auto lease = state->AcquireAll(location);
  for (size_t rank = 0; rank < calls.size(); rank++) {
    state->BeginBarrier(rank, guards[rank]->GetStream(), location);
  }
  const auto &cuda_api = internal::GetCudaApi();
  int previous_device = -1;
  internal::CheckCuda(cuda_api.get_device_(&previous_device), "cudaGetDevice (local barrier)", location);
  std::vector<ncclResult_t> statuses;
  statuses.reserve(calls.size());
  internal::CheckNccl(internal::GetNcclApi().group_start_(), "ncclGroupStart", location);
  cudaError_t set_device_status = cudaSuccess;
  for (size_t rank = 0; rank < calls.size(); rank++) {
    set_device_status = cuda_api.set_device_(state->GetDevice(rank).GetOrdinal());
    if (set_device_status != cudaSuccess) {
      break;
    }
    const auto &storage = state->GetBarrierStorage(rank);
    statuses.push_back(internal::GetNcclApi().all_reduce_(storage->GetBasePointer(), storage->GetBasePointer(), 1,
                                                          ncclUint8, ncclMax, lease.GetHandle(rank),
                                                          guards[rank]->GetNativeStream()));
  }
  const auto end_status = internal::GetNcclApi().group_end_();
  const auto restore_status = cuda_api.set_device_(previous_device);
  for (size_t rank = 0; rank < calls.size(); rank++) {
    state->EndBarrier(rank, guards[rank]->GetStream(), location);
  }
  if (set_device_status != cudaSuccess || restore_status != cudaSuccess) {
    state->MarkFailed();
  }
  internal::CheckCuda(set_device_status, "cudaSetDevice (local barrier)", location);
  internal::CheckCuda(restore_status, "cudaSetDevice (restore after local barrier)", location);
  state->CheckGroupedSubmission(lease.GetRanks(), statuses, end_status, "NCCL barrier", location);
}

}  // namespace ttl
