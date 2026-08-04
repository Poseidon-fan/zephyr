#include "ttl/distributed/collective.hpp"

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

#include <driver_types.h>
#include <nccl.h>

#include "ttl/common/error.hpp"
#include "ttl/distributed/communicator.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/distributed/collective_common.hpp"
#include "ttl/internal/distributed/communicator.hpp"
#include "ttl/internal/distributed/nccl_api.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/graph/graph.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

struct CountPlan final {
  std::vector<size_t> counts_;
  std::vector<size_t> offsets_bytes_;
  size_t total_count_;
};

struct PreparedVariableCall final {
  internal::PreparedCollectiveCall tensors_;
  CountPlan sends_;
  CountPlan receives_;
};

struct VariableCallPlan final {
  CountPlan sends_;
  CountPlan receives_;
  bool pack_input_;
  bool unpack_output_;
};

[[nodiscard]] auto BuildCountPlan(std::span<const int64_t> counts, size_t world_size, int64_t expected_total,
                                  size_t element_size, std::string_view role, std::source_location location)
    -> CountPlan {
  if (counts.size() != world_size) {
    throw InvalidArgumentError(
        std::string{"AllToAllVOut "} + std::string{role} + " count list must contain one value per communicator rank",
        location);
  }
  CountPlan plan;
  plan.counts_.reserve(world_size);
  plan.offsets_bytes_.reserve(world_size);
  auto total = size_t{0};
  for (const auto count : counts) {
    if (count < 0) {
      throw InvalidArgumentError(std::string{"AllToAllVOut "} + std::string{role} + " counts must be non-negative",
                                 location);
    }
    const auto narrowed = internal::CheckedNarrow<size_t>(count, "AllToAllVOut peer count", location);
    plan.offsets_bytes_.push_back(internal::CheckedMultiply(total, element_size, "AllToAllVOut byte offset", location));
    plan.counts_.push_back(narrowed);
    total = internal::CheckedAdd(total, narrowed, "AllToAllVOut total count", location);
  }
  if (!std::cmp_equal(total, expected_total)) {
    throw InvalidArgumentError(
        std::string{"AllToAllVOut "} + std::string{role} + " counts must sum to the corresponding tensor element count",
        location);
  }
  plan.total_count_ = total;
  return plan;
}

[[nodiscard]] auto ValidateAndPlan(ExecutionContext &context, Tensor &output, const Tensor &input,
                                   std::span<const int64_t> send_counts, std::span<const int64_t> receive_counts,
                                   NcclCommunicator &communicator, std::source_location location) -> VariableCallPlan {
  const auto &state = internal::CommunicatorAccess::GetState(communicator, location);
  const auto rank = internal::CommunicatorAccess::GetRank(communicator, location);
  internal::ValidateCollectiveContext(context, communicator, state, location);
  internal::ValidateCollectiveTensorDevice(input, context.GetDevice(), "input", location);
  internal::ValidateCollectiveTensorDevice(output, context.GetDevice(), "output", location);
  if (output.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("AllToAllVOut input and output dtypes must match", location);
  }
  internal::ValidateWritableOutput(output, "AllToAllVOut", location);
  const auto alias = ClassifyAlias(output, input, location);
  if (alias != AliasKind::DISJOINT && (state->GetWorldSize() != 1 || alias != AliasKind::EXACT)) {
    throw InvalidArgumentError("AllToAllVOut supports exact input/output aliasing only for world size one", location);
  }
  const auto element_size = GetDTypeInfo(input.GetDType(), location).size_bytes_;
  auto sends =
      BuildCountPlan(send_counts, state->GetWorldSize(), input.GetNumElements(), element_size, "send", location);
  auto receives =
      BuildCountPlan(receive_counts, state->GetWorldSize(), output.GetNumElements(), element_size, "receive", location);
  if (sends.counts_[rank] != receives.counts_[rank]) {
    throw InvalidArgumentError("AllToAllVOut self send and receive counts must match", location);
  }
  return VariableCallPlan{
      .sends_ = std::move(sends),
      .receives_ = std::move(receives),
      .pack_input_ = input.GetNumElements() != 0 && !input.IsContiguous(),
      .unpack_output_ = output.GetNumElements() != 0 && !output.IsContiguous(),
  };
}

[[nodiscard]] auto PrepareVariableCall(ExecutionContext &context, Tensor &output, const Tensor &input,
                                       VariableCallPlan plan, std::source_location location) -> PreparedVariableCall {
  auto tensors = internal::PrepareCollectiveCall(context, output, input, "AllToAllVOut", plan.pack_input_,
                                                 plan.unpack_output_, location);
  return PreparedVariableCall{
      .tensors_ = std::move(tensors),
      .sends_ = std::move(plan.sends_),
      .receives_ = std::move(plan.receives_),
  };
}

[[nodiscard]] auto HasRemoteTransfer(const PreparedVariableCall &call, size_t rank) noexcept -> bool {
  for (size_t peer = 0; peer < call.sends_.counts_.size(); ++peer) {
    if (peer != rank && (call.sends_.counts_[peer] != 0 || call.receives_.counts_[peer] != 0)) {
      return true;
    }
  }
  return false;
}

void SubmitSelfCopy(PreparedVariableCall &call, size_t rank, cudaStream_t stream, std::source_location location) {
  const auto count = call.sends_.counts_[rank];
  if (count == 0) {
    return;
  }
  const auto bytes =
      internal::CheckedBytes(count, GetDTypeInfo(call.tensors_.GetInput().GetDType(), location).size_bytes_, location);
  const auto source = internal::CollectiveByteOffset(
      internal::TensorAccess::GetData(call.tensors_.GetInput(), location), call.sends_.offsets_bytes_[rank]);
  auto *destination = internal::MutableCollectiveByteOffset(
      internal::TensorAccess::GetMutableData(call.tensors_.GetOutput(), location), call.receives_.offsets_bytes_[rank]);
  if (source == destination) {
    return;
  }
  internal::CheckCuda(
      internal::GetCudaApi().memcpy_async_(destination, source, bytes, cudaMemcpyDeviceToDevice, stream),
      "cudaMemcpyAsync (AllToAllVOut self copy)", location);
}

void IssueRemoteTransfers(PreparedVariableCall &call, size_t rank, ncclComm_t communicator, cudaStream_t stream,
                          std::vector<ncclResult_t> &statuses, std::source_location location) {
  const auto &nccl_api = internal::GetNcclApi();
  const auto dtype = internal::ToNcclDType(call.tensors_.GetInput().GetDType(), location);
  const auto input = internal::TensorAccess::GetData(call.tensors_.GetInput(), location);
  auto *output = internal::TensorAccess::GetMutableData(call.tensors_.GetOutput(), location);
  for (size_t peer = 0; peer < call.sends_.counts_.size(); ++peer) {
    if (peer == rank) {
      continue;
    }
    const auto send_count = call.sends_.counts_[peer];
    if (send_count != 0) {
      statuses.push_back(nccl_api.send_(internal::CollectiveByteOffset(input, call.sends_.offsets_bytes_[peer]),
                                        send_count, dtype, static_cast<int>(peer), communicator, stream));
    }
    const auto receive_count = call.receives_.counts_[peer];
    if (receive_count != 0) {
      statuses.push_back(
          nccl_api.receive_(internal::MutableCollectiveByteOffset(output, call.receives_.offsets_bytes_[peer]),
                            receive_count, dtype, static_cast<int>(peer), communicator, stream));
    }
  }
}

void RecordCall(internal::OpGuard &guard, PreparedVariableCall &call) {
  guard.RecordTensor(call.tensors_.GetInput());
  guard.RecordTensor(call.tensors_.GetOutput());
}

[[nodiscard]] auto ValidateLocalCalls(std::span<const LocalAllToAllVCall> calls, std::source_location location)
    -> std::shared_ptr<internal::CommunicatorGroupState> {
  if (calls.empty() || calls.front().context_ == nullptr || calls.front().output_ == nullptr ||
      calls.front().input_ == nullptr || calls.front().communicator_ == nullptr) {
    throw InvalidArgumentError("AllToAllVLocal requires a non-empty complete call list", location);
  }
  auto state = internal::CommunicatorAccess::GetState(*calls.front().communicator_, location);
  if (calls.size() != state->GetWorldSize()) {
    throw InvalidArgumentError("AllToAllVLocal requires one call for every communicator rank", location);
  }
  const auto dtype = calls.front().input_->GetDType();
  for (size_t rank = 0; rank < calls.size(); ++rank) {
    const auto &call = calls[rank];
    if (call.context_ == nullptr || call.output_ == nullptr || call.input_ == nullptr ||
        call.communicator_ == nullptr) {
      throw InvalidArgumentError("AllToAllVLocal call fields must not be null", location);
    }
    if (internal::CommunicatorAccess::GetState(*call.communicator_, location).get() != state.get() ||
        internal::CommunicatorAccess::GetRank(*call.communicator_, location) != rank) {
      throw InvalidArgumentError("AllToAllVLocal calls must be ordered by rank from one group", location);
    }
    if (call.send_counts_.size() != calls.size() || call.receive_counts_.size() != calls.size()) {
      throw InvalidArgumentError("AllToAllVLocal count lists must contain one value per rank", location);
    }
    if (call.input_->GetDType() != dtype || call.output_->GetDType() != dtype) {
      throw InvalidArgumentError("AllToAllVLocal requires one common dtype across all ranks", location);
    }
  }
  for (size_t rank = 0; rank < calls.size(); ++rank) {
    const auto &call = calls[rank];
    for (size_t peer = 0; peer < calls.size(); ++peer) {
      if (call.send_counts_[peer] < 0 || call.receive_counts_[peer] < 0) {
        throw InvalidArgumentError("AllToAllVLocal counts must be non-negative", location);
      }
      if (call.send_counts_[peer] != calls[peer].receive_counts_[rank]) {
        throw InvalidArgumentError("AllToAllVLocal send and receive count matrices do not match", location);
      }
    }
  }
  return state;
}

}  // namespace

void AllToAllVOut(ExecutionContext &context, Tensor &output, const Tensor &input, std::span<const int64_t> send_counts,
                  std::span<const int64_t> receive_counts, NcclCommunicator &communicator,
                  std::source_location location) {
  auto plan = ValidateAndPlan(context, output, input, send_counts, receive_counts, communicator, location);
  auto call = PrepareVariableCall(context, output, input, std::move(plan), location);
  const auto &state = internal::CommunicatorAccess::GetState(communicator, location);
  const auto rank = internal::CommunicatorAccess::GetRank(communicator, location);
  if (call.sends_.total_count_ != 0 || call.receives_.total_count_ != 0) {
    internal::OpGuard guard{context, "AllToAllVOut", location, internal::CapturePolicy::SAFE};
    guard.RetainCommunicator(state);
    RecordCall(guard, call);
    SubmitSelfCopy(call, rank, guard.GetNativeStream(), location);
    if (HasRemoteTransfer(call, rank)) {
      auto lease = state->AcquireRank(rank, location);
      std::vector<ncclResult_t> statuses;
      statuses.reserve((state->GetWorldSize() - 1) * 2);
      internal::CheckNccl(internal::GetNcclApi().group_start_(), "ncclGroupStart", location);
      try {
        IssueRemoteTransfers(call, rank, lease.GetHandle(rank), guard.GetNativeStream(), statuses, location);
      } catch (...) {
        const auto cleanup_status = internal::GetNcclApi().group_end_();
        internal::ReportCollectiveCleanupErrors(context, cleanup_status, std::nullopt,
                                                "ncclGroupEnd (AllToAllVOut exception cleanup)", {}, location);
        state->MarkFailed();
        throw;
      }
      const auto end_status = internal::GetNcclApi().group_end_();
      state->CheckGroupedSubmission(lease.GetRanks(), statuses, end_status, "AllToAllVOut", location);
    }
  }
  internal::CompleteCollectiveCallOrFail(call.tensors_, state, location);
}

void AllToAllVLocal(std::span<const LocalAllToAllVCall> calls, std::source_location location) {
  auto state = ValidateLocalCalls(calls, location);
  for (const auto &call : calls) {
    if (internal::GetCaptureState(*call.context_, location) != nullptr) {
      throw CaptureError("AllToAllVLocal is not allowed during CUDA graph capture; use rank-local graph callbacks",
                         location);
    }
  }

  std::vector<PreparedVariableCall> prepared;
  std::vector<VariableCallPlan> plans;
  plans.reserve(calls.size());
  for (const auto &call : calls) {
    plans.push_back(ValidateAndPlan(*call.context_, *call.output_, *call.input_, call.send_counts_,
                                    call.receive_counts_, *call.communicator_, location));
  }
  prepared.reserve(calls.size());
  for (size_t rank = 0; rank < calls.size(); ++rank) {
    prepared.push_back(PrepareVariableCall(*calls[rank].context_, *calls[rank].output_, *calls[rank].input_,
                                           std::move(plans[rank]), location));
  }

  std::vector<std::unique_ptr<internal::OpGuard>> guards;
  guards.reserve(calls.size());
  auto has_remote_transfer = false;
  for (size_t rank = 0; rank < calls.size(); ++rank) {
    guards.push_back(std::make_unique<internal::OpGuard>(*calls[rank].context_, "AllToAllVLocal", location));
    RecordCall(*guards.back(), prepared[rank]);
    SubmitSelfCopy(prepared[rank], rank, guards.back()->GetNativeStream(), location);
    has_remote_transfer = has_remote_transfer || HasRemoteTransfer(prepared[rank], rank);
  }

  if (has_remote_transfer) {
    auto lease = state->AcquireAll(location);
    const auto &cuda_api = internal::GetCudaApi();
    int previous_device = -1;
    internal::CheckCuda(cuda_api.get_device_(&previous_device), "cudaGetDevice (AllToAllVLocal)", location);
    std::vector<ncclResult_t> statuses;
    statuses.reserve(calls.size() * (calls.size() - 1) * 2);
    internal::CheckNccl(internal::GetNcclApi().group_start_(), "ncclGroupStart", location);
    auto set_device_status = cudaSuccess;
    try {
      for (size_t rank = 0; rank < calls.size(); ++rank) {
        set_device_status = cuda_api.set_device_(state->GetDevice(rank).GetOrdinal());
        if (set_device_status != cudaSuccess) {
          break;
        }
        IssueRemoteTransfers(prepared[rank], rank, lease.GetHandle(rank), guards[rank]->GetNativeStream(), statuses,
                             location);
      }
    } catch (...) {
      const auto cleanup_status = internal::GetNcclApi().group_end_();
      const auto restore_status = cuda_api.set_device_(previous_device);
      internal::ReportCollectiveCleanupErrors(*calls.front().context_, cleanup_status, restore_status,
                                              "ncclGroupEnd (AllToAllVLocal exception cleanup)",
                                              "cudaSetDevice (AllToAllVLocal exception cleanup)", location);
      state->MarkFailed();
      throw;
    }
    const auto end_status = internal::GetNcclApi().group_end_();
    const auto restore_status = cuda_api.set_device_(previous_device);
    if (set_device_status != cudaSuccess || restore_status != cudaSuccess) {
      state->MarkFailed();
    }
    internal::CheckCuda(set_device_status, "cudaSetDevice (AllToAllVLocal)", location);
    internal::CheckCuda(restore_status, "cudaSetDevice (restore after AllToAllVLocal)", location);
    state->CheckGroupedSubmission(lease.GetRanks(), statuses, end_status, "AllToAllVLocal", location);
  }
  guards.clear();
  for (auto &call : prepared) {
    internal::CompleteCollectiveCallOrFail(call.tensors_, state, location);
  }
}

}  // namespace ttl
