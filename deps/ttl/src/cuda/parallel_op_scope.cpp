#include "ttl/internal/parallel_op_scope.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <driver_types.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/cuda_graph.hpp"
#include "ttl/internal/event_pool.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/internal/execution_lane.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/runtime.hpp"
#include "ttl/internal/scratch_arena.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/stream.hpp"
#include "ttl/tensor.hpp"

namespace ttl::internal {
namespace {

struct FirstCudaFailure final {
  void Observe(cudaError_t call_status, std::string_view call_operation) noexcept {
    if (status_ == cudaSuccess && call_status != cudaSuccess) {
      status_ = call_status;
      operation_ = call_operation;
    }
  }

  cudaError_t status_{cudaSuccess};
  std::string_view operation_;
};

void ReportUnfinishedScope(ErrorSink &error_sink, Device device, uint64_t stream_id, std::string_view operation,
                           std::source_location location) noexcept {
  try {
    std::string message{"parallel operator scope for "};
    message.append(operation);
    message.append(" was destroyed without Finish");
    error_sink.Report(ErrorRecord{
        .code_ = ErrorCode::INTERNAL,
        .message_ = std::move(message),
        .device_ = device,
        .stream_id_ = stream_id,
        .location_ = location,
    });
  } catch (...) {
    return;
  }
}

}  // namespace

ParallelOpScope::ParallelOpScope(OpGuard &guard, size_t auxiliary_stream_count, std::source_location location)
    : guard_(guard),
      impl_(ContextAccess::GetImpl(guard.context_, location)),
      auxiliary_stream_count_(auxiliary_stream_count),
      location_(location) {
  if (auxiliary_stream_count_ == 0) {
    throw InvalidArgumentError("parallel operator scope requires at least one auxiliary stream", location_);
  }
  if (auxiliary_stream_count_ > impl_.auxiliary_lanes_.size()) {
    throw InvalidArgumentError("parallel operator scope requested more auxiliary streams than the context owns",
                               location_);
  }
  if (guard_.parallel_scope_active_) {
    throw InvalidArgumentError("nested parallel operator scopes are not supported", location_);
  }
  if (guard_.capture_state_ == nullptr &&
      (!impl_.fork_event_.has_value() || impl_.join_events_.size() < auxiliary_stream_count_)) {
    throw InternalError("execution context auxiliary dependency resources are incomplete", location_);
  }

  guard_.parallel_scope_active_ = true;
  const auto &cuda_api = GetCudaApi();
  const auto primary_stream = StreamAccess::GetNative(impl_.primary_lane_.GetStream());
  auto &fork_event = guard_.capture_state_ == nullptr ? *impl_.fork_event_ : guard_.capture_state_->GetForkEvent();
  FirstCudaFailure failure;
  failure.Observe(cuda_api.record_event_(fork_event.GetNative(), primary_stream), "cudaEventRecord (parallel fork)");
  if (failure.status_ == cudaSuccess) {
    for (size_t index = 0; index < auxiliary_stream_count_; index++) {
      if (guard_.capture_state_ != nullptr) {
        guard_.capture_state_->RetainAuxiliaryStream(StreamAccess::GetState(impl_.auxiliary_lanes_[index].GetStream()));
      }
      failure.Observe(cuda_api.stream_wait_event_(StreamAccess::GetNative(impl_.auxiliary_lanes_[index].GetStream()),
                                                  fork_event.GetNative(), cudaEventWaitDefault),
                      "cudaStreamWaitEvent (parallel fork)");
    }
  }

  if (failure.status_ != cudaSuccess) {
    MarkFailed();
    CheckCuda(failure.status_, failure.operation_, location_);
  }
}

ParallelOpScope::~ParallelOpScope() noexcept {
  if (status_ != Status::ACTIVE) {
    return;
  }

  RuntimeState &runtime_state = *impl_.runtime_state_;
  const auto result = EnqueueJoin();
  if (result.status_ != cudaSuccess) {
    const ErrorReportContext error_context{
        .location_ = location_,
        .device_ = impl_.primary_lane_.GetStream().GetDevice(),
        .stream_id_ = impl_.primary_lane_.GetStream().GetId(),
    };
    TryCuda(result.status_, result.operation_, *runtime_state.GetErrorSink(), error_context);
  }
  MarkFailed();

  ReportUnfinishedScope(*runtime_state.GetErrorSink(), impl_.primary_lane_.GetStream().GetDevice(),
                        impl_.primary_lane_.GetStream().GetId(), guard_.operation_, location_);
}

auto ParallelOpScope::GetAuxiliaryStreamCount() const noexcept -> size_t { return auxiliary_stream_count_; }

auto ParallelOpScope::GetAuxiliaryLane(size_t index) const -> ExecutionLane & {
  if (index >= auxiliary_stream_count_) {
    throw InvalidArgumentError("auxiliary stream index is out of range", location_);
  }
  return impl_.auxiliary_lanes_[index];
}

auto ParallelOpScope::GetAuxiliaryStream(size_t index) const -> const Stream & {
  return GetAuxiliaryLane(index).GetStream();
}

auto ParallelOpScope::GetNativeAuxiliaryStream(size_t index) const -> cudaStream_t {
  return StreamAccess::GetNative(GetAuxiliaryStream(index));
}

auto ParallelOpScope::GetAuxiliaryCublasHandle(size_t index) const -> cublasHandle_t {
  auto &lane = GetAuxiliaryLane(index);
  if (guard_.capture_state_ != nullptr && !lane.HasBlas()) {
    throw CaptureError("auxiliary cuBLAS resources must be warmed up before CUDA graph capture", location_);
  }
  return lane.GetCublasHandle(location_);
}

auto ParallelOpScope::GetAuxiliaryCublasLtHandle(size_t index) const -> cublasLtHandle_t {
  auto &lane = GetAuxiliaryLane(index);
  if (guard_.capture_state_ != nullptr && !lane.HasBlas()) {
    throw CaptureError("auxiliary cuBLAS resources must be warmed up before CUDA graph capture", location_);
  }
  return lane.GetCublasLtHandle(location_);
}

auto ParallelOpScope::GetAuxiliaryBlasWorkspace(size_t index) const -> void * {
  auto &lane = GetAuxiliaryLane(index);
  if (guard_.capture_state_ != nullptr && !lane.HasBlas()) {
    throw CaptureError("auxiliary cuBLAS resources must be warmed up before CUDA graph capture", location_);
  }
  const auto &storage = lane.GetBlasWorkspaceStorage(location_);
  guard_.RetainStorage(storage);
  return storage->GetBasePointer();
}

auto ParallelOpScope::GetAuxiliaryBlasWorkspaceBytes(size_t index) const -> size_t {
  auto &lane = GetAuxiliaryLane(index);
  if (guard_.capture_state_ != nullptr && !lane.HasBlas()) {
    throw CaptureError("auxiliary cuBLAS resources must be warmed up before CUDA graph capture", location_);
  }
  const auto &storage = lane.GetBlasWorkspaceStorage(location_);
  guard_.RetainStorage(storage);
  return storage->GetCapacityBytes();
}

auto ParallelOpScope::MakeAuxiliaryScratchScope(size_t index) const -> ScratchArena::Scope {
  auto &lane = GetAuxiliaryLane(index);
  if (lane.GetScratchStorage() != nullptr) {
    guard_.RetainStorage(lane.GetScratchStorage());
  }
  return lane.MakeScratchScope(
      guard_.capture_state_ == nullptr ? ScratchGrowthPolicy::GROWABLE : ScratchGrowthPolicy::FIXED_CAPACITY,
      location_);
}

void ParallelOpScope::ReserveAuxiliaryScratch(size_t index, size_t capacity_bytes) const {
  if (guard_.capture_state_ != nullptr) {
    throw CaptureError("auxiliary scratch capacity cannot be reserved during CUDA graph capture", location_);
  }
  GetAuxiliaryLane(index).ReserveScratch(capacity_bytes, location_);
}

auto ParallelOpScope::GetAuxiliaryScratchCapacityBytes(size_t index) const -> size_t {
  return GetAuxiliaryLane(index).GetScratchCapacityBytes();
}

auto ParallelOpScope::GetAuxiliaryScratchHighWaterBytes(size_t index) const -> size_t {
  return GetAuxiliaryLane(index).GetScratchHighWaterBytes();
}

void ParallelOpScope::RecordTensor(const Tensor &tensor, size_t auxiliary_stream_index) {
  guard_.RecordTensorOnStream(tensor, GetAuxiliaryStream(auxiliary_stream_index));
}

void ParallelOpScope::CheckLaunch() const { guard_.CheckLaunch(); }

void ParallelOpScope::Finish() {
  if (status_ != Status::ACTIVE) {
    throw InvalidArgumentError("parallel operator scope is not active", location_);
  }
  const auto result = EnqueueJoin();
  if (result.status_ != cudaSuccess) {
    MarkFailed();
    CheckCuda(result.status_, result.operation_, location_);
  }
  status_ = Status::JOINED;
  guard_.parallel_scope_active_ = false;
}

void ParallelOpScope::MarkFailed() noexcept {
  impl_.status_.store(ExecutionContextStatus::FAILED, std::memory_order_release);
  if (impl_.fork_event_.has_value()) {
    impl_.fork_event_->Discard();
  }
  for (auto &event : impl_.join_events_) {
    event.Discard();
  }
  guard_.parallel_scope_active_ = false;
  status_ = Status::FAILED;
}

auto ParallelOpScope::EnqueueJoin() noexcept -> JoinResult {
  const auto &cuda_api = GetCudaApi();
  const auto primary_stream = StreamAccess::GetNative(impl_.primary_lane_.GetStream());
  FirstCudaFailure failure;

  for (size_t index = 0; index < auxiliary_stream_count_; index++) {
    auto &join_event =
        guard_.capture_state_ == nullptr ? impl_.join_events_[index] : guard_.capture_state_->GetJoinEvent(index);
    const auto record_status = cuda_api.record_event_(
        join_event.GetNative(), StreamAccess::GetNative(impl_.auxiliary_lanes_[index].GetStream()));
    failure.Observe(record_status, "cudaEventRecord (parallel join)");
    if (record_status == cudaSuccess) {
      const auto wait_status =
          cuda_api.stream_wait_event_(primary_stream, join_event.GetNative(), cudaEventWaitDefault);
      failure.Observe(wait_status, "cudaStreamWaitEvent (parallel join)");
    }
  }

  return JoinResult{
      .status_ = failure.status_,
      .operation_ = failure.operation_,
  };
}

}  // namespace ttl::internal
