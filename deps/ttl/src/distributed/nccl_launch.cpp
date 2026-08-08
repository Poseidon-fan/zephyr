#include "ttl/distributed/nccl_launch.hpp"

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <ranges>
#include <source_location>
#include <span>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nccl.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/distributed/communicator.hpp"
#include "ttl/internal/distributed/nccl_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/graph/graph.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

void ValidateContext(ExecutionContext &context, const std::shared_ptr<internal::CommunicatorGroupState> &state,
                     size_t rank, std::source_location location) {
  if (context.GetDevice() != state->GetDevice(rank)) {
    throw InvalidArgumentError("NCCL kernel context device does not match its communicator rank", location);
  }
  if (!state->BelongsTo(internal::ContextAccess::GetRuntimeState(context, location))) {
    throw InvalidArgumentError("NCCL kernel context and communicator belong to different runtimes", location);
  }
}

}  // namespace

NcclKernelLaunch::NcclKernelLaunch(ExecutionContext &context, std::string_view operation,
                                   std::span<const Tensor> inputs, std::span<Tensor *const> outputs,
                                   const CudaKernelLaunchOptions &options,
                                   std::shared_ptr<internal::CommunicatorGroupState> state, size_t rank,
                                   ncclComm_t communicator, std::source_location location)
    : state_(std::move(state)),
      rank_(rank),
      communicator_(communicator),
      cuda_launch_(context, operation, inputs, outputs, options, location) {
  if (communicator_ == nullptr) {
    throw InternalError("NCCL operation lease returned a null communicator", location);
  }
  cuda_launch_.RetainCommunicator(state_);
}

NcclKernelLaunch::~NcclKernelLaunch() noexcept = default;

auto NcclKernelLaunch::GetCudaLaunch() noexcept -> CudaKernelLaunch & { return cuda_launch_; }

auto NcclKernelLaunch::GetCommunicator() const noexcept -> ncclComm_t { return communicator_; }

auto NcclKernelLaunch::GetRank() const noexcept -> size_t { return rank_; }

auto NcclKernelLaunch::GetWorldSize() const noexcept -> size_t { return state_->GetWorldSize(); }

void NcclKernelLaunch::Finish() { cuda_launch_.Finish(); }

void NcclKernelLaunch::FailAfterCallbackException() noexcept { cuda_launch_.FailAfterCallbackException(); }

void SubmitNcclKernel(ExecutionContext &context, NcclCommunicator &communicator, std::string_view operation,
                      std::span<const Tensor> inputs, std::span<Tensor *const> outputs,
                      const NcclKernelFunction &function, const CudaKernelLaunchOptions &options,
                      std::source_location location) {
  if (!function) {
    throw InvalidArgumentError("NCCL kernel submission requires a callback", location);
  }
  const auto &state = internal::CommunicatorAccess::GetState(communicator, location);
  const auto rank = internal::CommunicatorAccess::GetRank(communicator, location);
  ValidateContext(context, state, rank, location);
  auto lease = state->AcquireRank(rank, location);
  NcclKernelLaunch launch{context, operation, inputs, outputs, options, state, rank, lease.GetHandle(rank), location};

  const auto start_status = internal::GetNcclApi().group_start_();
  if (start_status != ncclSuccess) {
    state->MarkFailed();
    internal::CheckNccl(start_status, "ncclGroupStart (checked NCCL kernel)", location);
  }

  ncclResult_t submission_status = ncclSuccess;
  try {
    submission_status = std::invoke(function, launch);
  } catch (...) {
    static_cast<void>(internal::GetNcclApi().group_end_());
    launch.FailAfterCallbackException();
    state->MarkFailed();
    throw;
  }
  const auto end_status = internal::GetNcclApi().group_end_();
  std::exception_ptr launch_error;
  try {
    launch.Finish();
  } catch (...) {
    launch_error = std::current_exception();
    state->MarkFailed();
  }
  try {
    const auto statuses = std::span<const ncclResult_t>{&submission_status, 1};
    state->CheckGroupedSubmission(lease.GetRanks(), statuses, end_status, operation, location);
  } catch (...) {
    if (launch_error == nullptr) {
      throw;
    }
  }
  if (launch_error != nullptr) {
    std::rethrow_exception(launch_error);
  }
}

void SubmitNcclKernelsLocal(std::span<const LocalNcclKernelCall> calls, std::string_view operation,
                            const LocalNcclKernelFunction &function, std::source_location location) {
  if (calls.empty()) {
    throw InvalidArgumentError("local NCCL kernel submission requires at least one rank", location);
  }
  if (!function) {
    throw InvalidArgumentError("local NCCL kernel submission requires a callback", location);
  }
  if (calls.front().context_ == nullptr || calls.front().communicator_ == nullptr) {
    throw InvalidArgumentError("local NCCL kernel call contains a null context or communicator", location);
  }

  const auto state = internal::CommunicatorAccess::GetState(*calls.front().communicator_, location);
  if (calls.size() != state->GetWorldSize()) {
    throw InvalidArgumentError("local NCCL kernel submission requires exactly one call per rank", location);
  }
  std::unordered_set<size_t> ranks;
  ranks.reserve(calls.size());
  for (const auto &call : calls) {
    if (call.context_ == nullptr || call.communicator_ == nullptr) {
      throw InvalidArgumentError("local NCCL kernel call contains a null context or communicator", location);
    }
    if (internal::CommunicatorAccess::GetState(*call.communicator_, location).get() != state.get()) {
      throw InvalidArgumentError("local NCCL kernel calls must use one communicator group", location);
    }
    const auto rank = internal::CommunicatorAccess::GetRank(*call.communicator_, location);
    if (!ranks.insert(rank).second) {
      throw InvalidArgumentError("local NCCL kernel submission contains a duplicate rank", location);
    }
    ValidateContext(*call.context_, state, rank, location);
    if (internal::GetCaptureState(*call.context_, location) != nullptr) {
      throw CaptureError(
          "local NCCL kernel submission is not capture-safe; use rank-local submissions in CapturedGraphGroup",
          location);
    }
  }

  auto lease = state->AcquireAll(location);
  const auto start_status = internal::GetNcclApi().group_start_();
  if (start_status != ncclSuccess) {
    state->MarkFailed();
    internal::CheckNccl(start_status, "ncclGroupStart (local checked NCCL kernel)", location);
  }

  std::vector<std::unique_ptr<NcclKernelLaunch>> launches;
  std::vector<ncclResult_t> statuses;
  launches.reserve(calls.size());
  statuses.reserve(calls.size());
  try {
    for (size_t index = 0; index < calls.size(); ++index) {
      const auto &call = calls[index];
      const auto rank = internal::CommunicatorAccess::GetRank(*call.communicator_, location);
      auto launch = std::unique_ptr<NcclKernelLaunch>{new NcclKernelLaunch{*call.context_, operation, call.inputs_,
                                                                           call.outputs_, call.options_, state, rank,
                                                                           lease.GetHandle(rank), location}};
      launches.push_back(std::move(launch));
      statuses.push_back(std::invoke(function, index, *launches.back()));
    }
  } catch (...) {
    static_cast<void>(internal::GetNcclApi().group_end_());
    for (auto &launch : std::views::reverse(launches)) {
      launch->FailAfterCallbackException();
    }
    state->MarkFailed();
    throw;
  }

  const auto end_status = internal::GetNcclApi().group_end_();
  std::exception_ptr launch_error;
  auto finish_failed = false;
  for (auto &launch : std::views::reverse(launches)) {
    try {
      launch->Finish();
    } catch (...) {
      finish_failed = true;
      if (launch_error == nullptr) {
        launch_error = std::current_exception();
      }
    }
    launch.reset();
  }
  if (finish_failed) {
    state->MarkFailed();
  }
  try {
    state->CheckGroupedSubmission(lease.GetRanks(), statuses, end_status, operation, location);
  } catch (...) {
    if (launch_error == nullptr) {
      throw;
    }
  }
  if (launch_error != nullptr) {
    std::rethrow_exception(launch_error);
  }
}

}  // namespace ttl
