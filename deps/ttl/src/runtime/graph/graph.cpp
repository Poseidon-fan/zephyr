#include "ttl/runtime/graph.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/distributed/communicator.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/graph/graph.hpp"
#include "ttl/internal/runtime/memory/allocation.hpp"
#include "ttl/internal/runtime/runtime.hpp"
#include "ttl/internal/tensor/storage.hpp"
#include "ttl/runtime/device.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {
namespace {

[[nodiscard]] auto MakeErrorContext(const StreamState &stream, std::source_location location) noexcept
    -> ErrorReportContext {
  return ErrorReportContext{
      .location_ = location,
      .device_ = stream.GetDevice(),
      .stream_id_ = stream.GetId(),
  };
}

[[nodiscard]] auto WrapCaptureError(const CudaError &error, std::source_location location) -> CaptureError {
  return CaptureError(error.what(), location);
}

[[noreturn]] void ThrowCaptureCudaError(cudaError_t status, std::string_view operation, std::source_location location) {
  try {
    CheckCuda(status, operation, location);
  } catch (const CudaError &error) {
    throw WrapCaptureError(error, location);
  }
  throw InternalError("capture CUDA error conversion received cudaSuccess", location);
}

auto DestroyGraphNoexcept(cudaGraph_t graph, ErrorSink &error_sink, const ErrorReportContext &context) noexcept
    -> bool {
  return graph == nullptr || TryCuda(GetCudaApi().destroy_graph_(graph), "cudaGraphDestroy", error_sink, context);
}

auto DestroyGraphExecutableNoexcept(cudaGraphExec_t executable, ErrorSink &error_sink,
                                    const ErrorReportContext &context) noexcept -> bool {
  return executable == nullptr ||
         TryCuda(GetCudaApi().destroy_graph_executable_(executable), "cudaGraphExecDestroy", error_sink, context);
}

class CaptureContextUseGuard final {
 public:
  CaptureContextUseGuard(std::shared_ptr<ExecutionContextImpl> context, std::source_location location)
      : context_(std::move(context)) {
    if (context_->in_use_.test_and_set(std::memory_order_acquire)) {
      throw InvalidArgumentError("execution context is already in use by another host thread", location);
    }
  }

  CaptureContextUseGuard(const CaptureContextUseGuard &) = delete;
  auto operator=(const CaptureContextUseGuard &) -> CaptureContextUseGuard & = delete;
  ~CaptureContextUseGuard() noexcept { context_->in_use_.clear(std::memory_order_release); }

 private:
  std::shared_ptr<ExecutionContextImpl> context_;
};

void QuarantineGraphResources(
    cudaGraph_t graph, cudaGraphExec_t executable, std::shared_ptr<StreamState> primary_stream,
    std::vector<std::shared_ptr<StreamState>> auxiliary_streams, std::vector<PooledEvent> dependency_events,
    std::map<const Storage *, std::shared_ptr<Storage>> storage,
    std::map<const CommunicatorGroupState *, std::shared_ptr<CommunicatorGroupState>> communicators,
    const std::shared_ptr<RuntimeState> &runtime_state, std::source_location location,
    bool registrations_active) noexcept {
  auto *cleanup_state = new (std::nothrow) GraphCleanupState{graph,
                                                             executable,
                                                             std::move(primary_stream),
                                                             std::move(auxiliary_streams),
                                                             std::move(dependency_events),
                                                             std::move(storage),
                                                             std::move(communicators),
                                                             location,
                                                             registrations_active};
  if (cleanup_state == nullptr) {
    std::terminate();
  }
  runtime_state->EnqueueGraphCleanup(cleanup_state);
}

void CleanupGraphResourcesNoexcept(
    cudaGraph_t graph, cudaGraphExec_t executable, std::shared_ptr<StreamState> primary_stream,
    std::vector<std::shared_ptr<StreamState>> auxiliary_streams, std::vector<PooledEvent> dependency_events,
    std::map<const Storage *, std::shared_ptr<Storage>> storage,
    std::map<const CommunicatorGroupState *, std::shared_ptr<CommunicatorGroupState>> communicators,
    const std::shared_ptr<RuntimeState> &runtime_state, std::source_location location,
    bool registrations_active) noexcept {
  const auto error_context = MakeErrorContext(*primary_stream, location);
  CleanupDeviceGuard device_guard{primary_stream->GetDevice(), *runtime_state->GetErrorSink(), error_context,
                                  "destroy CUDA graph resources", "restore after CUDA graph resource destruction"};
  auto executable_destroyed = executable == nullptr;
  auto graph_destroyed = graph == nullptr;
  if (device_guard) {
    executable_destroyed = DestroyGraphExecutableNoexcept(executable, *runtime_state->GetErrorSink(), error_context);
    if (executable_destroyed) {
      graph_destroyed = DestroyGraphNoexcept(graph, *runtime_state->GetErrorSink(), error_context);
    }
  }
  if (!executable_destroyed || !graph_destroyed) {
    QuarantineGraphResources(graph_destroyed ? nullptr : graph, executable_destroyed ? nullptr : executable,
                             std::move(primary_stream), std::move(auxiliary_streams), std::move(dependency_events),
                             std::move(storage), std::move(communicators), runtime_state, location,
                             registrations_active);
    return;
  }
  if (registrations_active) {
    for (auto &entry : communicators) {
      entry.second->UnregisterGraph();
    }
    runtime_state->UnregisterGraph();
  }
}

void RegisterGraphResources(
    RuntimeState &runtime_state,
    const std::map<const CommunicatorGroupState *, std::shared_ptr<CommunicatorGroupState>> &communicators,
    std::source_location location) {
  runtime_state.RegisterGraph(location);
  auto registered_communicator_count = size_t{0};
  try {
    for (const auto &entry : communicators) {
      entry.second->RegisterGraph(location);
      registered_communicator_count++;
    }
  } catch (...) {
    auto iterator = communicators.begin();
    while (registered_communicator_count > 0) {
      iterator->second->UnregisterGraph();
      ++iterator;
      registered_communicator_count--;
    }
    runtime_state.UnregisterGraph();
    throw;
  }
}

}  // namespace

GraphCleanupState::GraphCleanupState(
    cudaGraph_t graph, cudaGraphExec_t executable, std::shared_ptr<StreamState> primary_stream,
    std::vector<std::shared_ptr<StreamState>> auxiliary_streams, std::vector<PooledEvent> dependency_events,
    std::map<const Storage *, std::shared_ptr<Storage>> storage,
    std::map<const CommunicatorGroupState *, std::shared_ptr<CommunicatorGroupState>> communicators,
    std::source_location location, bool registrations_active) noexcept
    : graph_(graph),
      executable_(executable),
      primary_stream_(std::move(primary_stream)),
      auxiliary_streams_(std::move(auxiliary_streams)),
      dependency_events_(std::move(dependency_events)),
      storage_(std::move(storage)),
      communicators_(std::move(communicators)),
      location_(location),
      registrations_active_(registrations_active) {}

auto GraphCleanupState::RetryNoexcept(RuntimeState &runtime_state) noexcept -> bool {
  const auto error_context = MakeErrorContext(*primary_stream_, location_);
  CleanupDeviceGuard device_guard{primary_stream_->GetDevice(), *runtime_state.GetErrorSink(), error_context,
                                  "retry CUDA graph resource cleanup", "restore after CUDA graph cleanup retry"};
  if (!device_guard) {
    return false;
  }
  if (executable_ != nullptr) {
    if (!DestroyGraphExecutableNoexcept(executable_, *runtime_state.GetErrorSink(), error_context)) {
      return false;
    }
    executable_ = nullptr;
  }
  if (graph_ != nullptr) {
    if (!DestroyGraphNoexcept(graph_, *runtime_state.GetErrorSink(), error_context)) {
      return false;
    }
    graph_ = nullptr;
  }
  if (registrations_active_) {
    for (auto &entry : communicators_) {
      entry.second->UnregisterGraph();
    }
    runtime_state.UnregisterGraph();
    registrations_active_ = false;
  }
  return true;
}

CaptureSessionState::CaptureSessionState(std::shared_ptr<ExecutionContextImpl> context,
                                         std::shared_ptr<RuntimeState> runtime_state,
                                         std::shared_ptr<StreamState> primary_stream,
                                         std::vector<PooledEvent> dependency_events, std::string name,
                                         std::source_location location) noexcept
    : context_(std::move(context)),
      runtime_state_(std::move(runtime_state)),
      primary_stream_(std::move(primary_stream)),
      dependency_events_(std::move(dependency_events)),
      name_(std::move(name)),
      location_(location),
      registration_(runtime_state_) {}

void CaptureRegistration::Complete() noexcept {
  if (!active_) {
    std::terminate();
  }
  active_ = false;
  runtime_state_->EndCapture();
}

auto CaptureSessionState::Begin(ExecutionContext &context, const GraphCaptureOptions &options,
                                std::source_location location) -> std::shared_ptr<CaptureSessionState> {
  ContextUseGuard use_guard{context, ContextUseMode::SUBMIT, location};
  auto &impl = ContextAccess::GetImpl(context, location);
  if (impl.status_.load(std::memory_order_acquire) != ExecutionContextStatus::READY || !impl.capture_state_.expired()) {
    throw CaptureError("execution context is not ready to begin CUDA graph capture", location);
  }
  auto primary_stream = StreamAccess::GetState(impl.primary_lane_.GetStream());
  if (primary_stream->IsExternal() && !primary_stream->HasExternalOwner()) {
    throw CaptureError("CUDA graph capture requires an owner for an external stream", location);
  }

  std::vector<PooledEvent> dependency_events;
  if (!impl.auxiliary_lanes_.empty()) {
    if (impl.auxiliary_lanes_.size() == std::numeric_limits<size_t>::max()) {
      throw OverflowError("CUDA graph dependency event count overflow", location);
    }
    dependency_events.reserve(impl.auxiliary_lanes_.size() + 1);
    const auto &event_pool = impl.device_context_->GetEventPool();
    for (size_t index = 0; index <= impl.auxiliary_lanes_.size(); index++) {
      dependency_events.push_back(event_pool->Acquire(location));
    }
  }

  auto state = std::shared_ptr<CaptureSessionState>{
      new CaptureSessionState{ContextAccess::GetImplState(context, location), impl.runtime_state_,
                              std::move(primary_stream), std::move(dependency_events), options.name_, location}};

  impl.runtime_state_->BeginCapture(location);
  try {
    impl.runtime_state_->TrackCaptureSession(state, location);
    DeviceGuard device_guard{context.GetDevice(), *impl.runtime_state_->GetErrorSink(), location};
    CheckCuda(GetCudaApi().begin_stream_capture_(state->primary_stream_->GetNative(), cudaStreamCaptureModeRelaxed),
              "cudaStreamBeginCapture", location);
  } catch (...) {
    state->CancelBeforeNativeCapture();
    throw;
  }

  impl.capture_state_ = state;
  impl.status_.store(ExecutionContextStatus::CAPTURING, std::memory_order_release);
  return state;
}

CaptureSessionState::~CaptureSessionState() noexcept {
  if (IsActive()) {
    ReportCleanupFailure("active CUDA graph capture state outlived its CaptureSession");
  }
}

auto CaptureSessionState::Finish(std::source_location location) -> std::unique_ptr<CapturedGraphState> {
  CaptureContextUseGuard use_guard{context_, location};
  if (!IsActive()) {
    throw CaptureError("CUDA graph capture session is not active", location);
  }
  DeviceGuard device_guard{primary_stream_->GetDevice(), *runtime_state_->GetErrorSink(), location};
  cudaGraph_t graph = nullptr;
  const auto end_status = GetCudaApi().end_stream_capture_(primary_stream_->GetNative(), &graph);
  CompleteCapture(end_status == cudaSuccess ? CaptureStatus::FINISHED : CaptureStatus::ABORTED);
  if (end_status != cudaSuccess) {
    CleanupGraphResourcesNoexcept(graph, nullptr, std::move(primary_stream_), std::move(auxiliary_streams_),
                                  std::move(dependency_events_), std::move(retained_storage_),
                                  std::move(retained_communicators_), runtime_state_, location_, false);
    ThrowCaptureCudaError(end_status, "cudaStreamEndCapture", location);
  }
  if (graph == nullptr) {
    throw CaptureError("cudaStreamEndCapture returned a null graph", location);
  }

  size_t node_count = 0;
  cudaGraphExec_t executable = nullptr;
  auto registrations_active = false;
  try {
    CheckCuda(GetCudaApi().get_graph_nodes_(graph, nullptr, &node_count), "cudaGraphGetNodes", location);
    CheckCuda(GetCudaApi().instantiate_graph_(&executable, graph, 0), "cudaGraphInstantiate", location);
    if (executable == nullptr) {
      throw CaptureError("cudaGraphInstantiate returned a null executable", location);
    }
    RegisterGraphResources(*runtime_state_, retained_communicators_, location);
    registrations_active = true;

    auto graph_state = std::make_unique<CapturedGraphState>(
        primary_stream_->GetDevice(), primary_stream_->GetId(), graph, executable, primary_stream_,
        std::move(auxiliary_streams_), std::move(dependency_events_), std::move(retained_storage_),
        std::move(retained_communicators_), runtime_state_, std::move(name_), node_count, location_);
    graph = nullptr;
    executable = nullptr;
    status_.store(CaptureStatus::FINISHED, std::memory_order_release);
    return graph_state;
  } catch (const CudaError &error) {
    CleanupGraphResourcesNoexcept(graph, executable, std::move(primary_stream_), std::move(auxiliary_streams_),
                                  std::move(dependency_events_), std::move(retained_storage_),
                                  std::move(retained_communicators_), runtime_state_, location_, registrations_active);
    throw WrapCaptureError(error, location);
  } catch (...) {
    CleanupGraphResourcesNoexcept(graph, executable, std::move(primary_stream_), std::move(auxiliary_streams_),
                                  std::move(dependency_events_), std::move(retained_storage_),
                                  std::move(retained_communicators_), runtime_state_, location_, registrations_active);
    throw;
  }
}

void CaptureSessionState::Abort() noexcept {
  std::optional<CaptureContextUseGuard> use_guard;
  try {
    use_guard.emplace(context_, location_);
  } catch (...) {
    context_->status_.store(ExecutionContextStatus::FAILED, std::memory_order_release);
    ReportCleanupFailure("CUDA graph capture was aborted concurrently with context use");
    std::terminate();
  }
  if (!IsActive()) {
    return;
  }
  cudaGraph_t graph = nullptr;
  const ErrorReportContext error_context = MakeErrorContext(*primary_stream_, location_);
  CleanupDeviceGuard device_guard{primary_stream_->GetDevice(), *runtime_state_->GetErrorSink(), error_context,
                                  "abort CUDA graph capture", "restore after CUDA graph capture abort"};
  auto end_status = cudaErrorUnknown;
  if (device_guard) {
    end_status = GetCudaApi().end_stream_capture_(primary_stream_->GetNative(), &graph);
    if (end_status != cudaSuccess && end_status != cudaErrorStreamCaptureInvalidated) {
      TryCuda(end_status, "cudaStreamEndCapture", "abort CUDA graph capture", *runtime_state_->GetErrorSink(),
              error_context);
    }
  }
  if (!device_guard) {
    context_->status_.store(ExecutionContextStatus::FAILED, std::memory_order_release);
    pending_cleanup_.store(true, std::memory_order_release);
    pending_owner_ = shared_from_this();
    return;
  }
  if (end_status != cudaSuccess && end_status != cudaErrorStreamCaptureInvalidated) {
    context_->status_.store(ExecutionContextStatus::FAILED, std::memory_order_release);
  }
  CompleteCapture(CaptureStatus::ABORTED);
  CleanupGraphResourcesNoexcept(graph, nullptr, std::move(primary_stream_), std::move(auxiliary_streams_),
                                std::move(dependency_events_), std::move(retained_storage_),
                                std::move(retained_communicators_), runtime_state_, location_, false);
}

void CaptureSessionState::Invalidate() noexcept {
  auto expected = CaptureStatus::ACTIVE;
  static_cast<void>(status_.compare_exchange_strong(expected, CaptureStatus::INVALIDATED, std::memory_order_acq_rel));
}

auto CaptureSessionState::IsActive() const noexcept -> bool {
  const auto status = status_.load(std::memory_order_acquire);
  return status == CaptureStatus::ACTIVE || status == CaptureStatus::INVALIDATED;
}

auto CaptureSessionState::GetStatus() const noexcept -> CaptureStatus {
  return status_.load(std::memory_order_acquire);
}

auto CaptureSessionState::GetOperationCount() const noexcept -> uint64_t {
  return operation_count_.load(std::memory_order_acquire);
}

void CaptureSessionState::BeginOperation(std::string_view operation, std::source_location location) {
  if (status_.load(std::memory_order_acquire) != CaptureStatus::ACTIVE) {
    throw CaptureError("captured operator submitted to an invalidated or inactive capture session", location);
  }
  if (operation.empty()) {
    throw InternalError("captured operator name must not be empty", location);
  }
  const auto operation_count = operation_count_.load(std::memory_order_relaxed);
  if (operation_count == std::numeric_limits<uint64_t>::max()) {
    throw OverflowError("captured operation sequence overflow", location);
  }
  operation_count_.store(operation_count + 1, std::memory_order_release);
}

void CaptureSessionState::RetainStorage(const std::shared_ptr<Storage> &storage, std::source_location location) {
  if (storage == nullptr) {
    throw InternalError("CUDA graph cannot retain null storage", location);
  }
  if (storage->GetAllocationKind() == AllocationKind::EXTERNAL_BORROWED) {
    throw CaptureError("CUDA graph capture requires owned external memory", location);
  }
  retained_storage_.try_emplace(storage.get(), storage);
}

void CaptureSessionState::RetainCommunicator(const std::shared_ptr<CommunicatorGroupState> &communicator,
                                             std::source_location location) {
  if (communicator == nullptr) {
    throw InternalError("CUDA graph cannot retain a null communicator", location);
  }
  retained_communicators_.try_emplace(communicator.get(), communicator);
}

void CaptureSessionState::RetainAuxiliaryStream(std::shared_ptr<StreamState> stream) {
  if (stream == nullptr) {
    std::terminate();
  }
  const auto iterator =
      std::ranges::find_if(auxiliary_streams_, [&](const auto &candidate) { return candidate.get() == stream.get(); });
  if (iterator == auxiliary_streams_.end()) {
    auxiliary_streams_.push_back(std::move(stream));
  }
}

auto CaptureSessionState::GetForkEvent() noexcept -> PooledEvent & {
  if (dependency_events_.empty()) {
    std::terminate();
  }
  return dependency_events_.front();
}

auto CaptureSessionState::GetJoinEvent(size_t index) noexcept -> PooledEvent & {
  if (index >= dependency_events_.size() - 1) {
    std::terminate();
  }
  return dependency_events_[index + 1];
}

void CaptureSessionState::CompleteCapture(CaptureStatus final_status) noexcept {
  const auto registered_state = context_->capture_state_.lock();
  if (registered_state.get() != this) {
    context_->status_.store(ExecutionContextStatus::FAILED, std::memory_order_release);
    ReportCleanupFailure("execution context lost its active CUDA graph capture registration");
  } else {
    context_->capture_state_.reset();
    if (context_->status_.load(std::memory_order_acquire) != ExecutionContextStatus::FAILED) {
      context_->status_.store(ExecutionContextStatus::READY, std::memory_order_release);
    }
  }
  registration_.Complete();
  status_.store(final_status, std::memory_order_release);
}

void CaptureSessionState::CancelBeforeNativeCapture() noexcept {
  registration_.Complete();
  status_.store(CaptureStatus::ABORTED, std::memory_order_release);
}

auto CaptureSessionState::HasPendingCleanup() const noexcept -> bool {
  return pending_cleanup_.load(std::memory_order_acquire);
}

void CaptureSessionState::RetryPendingCleanupNoexcept() noexcept {
  if (!HasPendingCleanup()) {
    return;
  }
  std::optional<CaptureContextUseGuard> use_guard;
  try {
    use_guard.emplace(context_, location_);
  } catch (...) {
    return;
  }
  const ErrorReportContext error_context = MakeErrorContext(*primary_stream_, location_);
  CleanupDeviceGuard device_guard{primary_stream_->GetDevice(), *runtime_state_->GetErrorSink(), error_context,
                                  "retry CUDA graph capture cleanup", "restore after CUDA graph cleanup retry"};
  if (!device_guard) {
    return;
  }
  cudaGraph_t graph = nullptr;
  const auto end_status = GetCudaApi().end_stream_capture_(primary_stream_->GetNative(), &graph);
  if (end_status != cudaSuccess && end_status != cudaErrorStreamCaptureInvalidated) {
    TryCuda(end_status, "cudaStreamEndCapture", "retry CUDA graph capture cleanup", *runtime_state_->GetErrorSink(),
            error_context);
  }
  CompleteCapture(CaptureStatus::ABORTED);
  pending_cleanup_.store(false, std::memory_order_release);
  pending_owner_.reset();
  CleanupGraphResourcesNoexcept(graph, nullptr, std::move(primary_stream_), std::move(auxiliary_streams_),
                                std::move(dependency_events_), std::move(retained_storage_),
                                std::move(retained_communicators_), runtime_state_, location_, false);
}

void CaptureSessionState::ReportCleanupFailure(std::string_view message) noexcept {
  try {
    runtime_state_->GetErrorSink()->Report(ErrorRecord{
        .code_ = ErrorCode::CAPTURE,
        .message_ = std::string{message},
        .device_ = primary_stream_->GetDevice(),
        .stream_id_ = primary_stream_->GetId(),
        .location_ = location_,
    });
  } catch (...) {
    return;
  }
}

CapturedGraphState::CapturedGraphState(
    Device device, uint64_t stream_id, cudaGraph_t graph, cudaGraphExec_t executable,
    std::shared_ptr<StreamState> primary_stream, std::vector<std::shared_ptr<StreamState>> auxiliary_streams,
    std::vector<PooledEvent> dependency_events, std::map<const Storage *, std::shared_ptr<Storage>> storage,
    std::map<const CommunicatorGroupState *, std::shared_ptr<CommunicatorGroupState>> communicators,
    std::shared_ptr<RuntimeState> runtime_state, std::string name, size_t node_count,
    std::source_location location) noexcept
    : device_(device),
      stream_id_(stream_id),
      graph_(graph),
      executable_(executable),
      primary_stream_(std::move(primary_stream)),
      auxiliary_streams_(std::move(auxiliary_streams)),
      dependency_events_(std::move(dependency_events)),
      storage_(std::move(storage)),
      communicators_(std::move(communicators)),
      runtime_state_(std::move(runtime_state)),
      name_(std::move(name)),
      node_count_(node_count),
      location_(location) {}

CapturedGraphState::~CapturedGraphState() noexcept {
  CleanupGraphResourcesNoexcept(graph_, executable_, std::move(primary_stream_), std::move(auxiliary_streams_),
                                std::move(dependency_events_), std::move(storage_), std::move(communicators_),
                                runtime_state_, location_, true);
}

void CapturedGraphState::FailLaunchNoexcept(ExecutionContextImpl &context) noexcept {
  failed_.store(true, std::memory_order_release);
  context.status_.store(ExecutionContextStatus::FAILED, std::memory_order_release);
  for (const auto &entry : communicators_) {
    entry.second->MarkFailed();
  }
}

void CapturedGraphState::Launch(ExecutionContext &context, std::source_location location) {
  ContextUseGuard use_guard{context, ContextUseMode::SUBMIT, location};
  auto &impl = ContextAccess::GetImpl(context, location);
  if (failed_.load(std::memory_order_acquire)) {
    throw CaptureError("captured CUDA graph is in a failed state and cannot be replayed", location);
  }
  if (impl.status_.load(std::memory_order_acquire) != ExecutionContextStatus::READY) {
    throw CaptureError("captured graph launch requires a ready execution context", location);
  }
  if (context.GetDevice() != device_ || context.GetStream().GetId() != stream_id_ ||
      ContextAccess::GetRuntimeState(context, location).get() != runtime_state_.get()) {
    throw InvalidArgumentError("captured graph must launch on the context stream that captured it", location);
  }
  if (launch_count_.load(std::memory_order_relaxed) == std::numeric_limits<uint64_t>::max()) {
    throw OverflowError("captured graph launch count overflow", location);
  }
  for (const auto &entry : communicators_) {
    entry.second->ValidateGraphLaunch(location);
  }

  DeviceGuard device_guard{device_, *runtime_state_->GetErrorSink(), location};
  for (const auto &entry : storage_) {
    entry.second->RecordUsage(context.GetStream());
  }
  try {
    CheckCuda(GetCudaApi().launch_graph_(executable_, primary_stream_->GetNative()), "cudaGraphLaunch", location);
    CheckCuda(GetCudaApi().get_last_error_(), "cudaGraphLaunch", location);
  } catch (...) {
    FailLaunchNoexcept(impl);
    throw;
  }
  launch_count_.fetch_add(1, std::memory_order_relaxed);
}

void CapturedGraphState::DebugDumpDot(std::string_view path, std::source_location location) const {
  if (path.empty()) {
    throw InvalidArgumentError("CUDA graph debug path must not be empty", location);
  }
  const std::string owning_path{path};
  DeviceGuard device_guard{device_, *runtime_state_->GetErrorSink(), location};
  CheckCuda(GetCudaApi().debug_graph_dot_print_(graph_, owning_path.c_str(), cudaGraphDebugDotFlagsVerbose),
            "cudaGraphDebugDotPrint", location);
}

auto CapturedGraphState::GetDevice() const noexcept -> Device { return device_; }

auto CapturedGraphState::GetStreamId() const noexcept -> uint64_t { return stream_id_; }

auto CapturedGraphState::GetNodeCount() const noexcept -> size_t { return node_count_; }

auto CapturedGraphState::GetLaunchCount() const noexcept -> uint64_t {
  return launch_count_.load(std::memory_order_relaxed);
}

auto CapturedGraphState::GetName() const noexcept -> std::string_view { return name_; }

auto GetCaptureState(ExecutionContext &context, std::source_location location) -> std::shared_ptr<CaptureSessionState> {
  return ContextAccess::GetImpl(context, location).capture_state_.lock();
}

}  // namespace ttl::internal

namespace ttl {
namespace {

class CancellableRendezvous final {
 public:
  explicit CancellableRendezvous(size_t participant_count) : participant_count_(participant_count) {}

  [[nodiscard]] auto ArriveAndWait() -> bool {
    std::unique_lock lock{latch_};
    if (cancelled_) {
      return false;
    }
    arrived_++;
    if (arrived_ == participant_count_) {
      released_ = true;
      condition_.notify_all();
      return true;
    }
    condition_.wait(lock, [&] { return released_ || cancelled_; });
    return !cancelled_;
  }

  void Cancel() noexcept {
    const std::scoped_lock lock{latch_};
    cancelled_ = true;
    condition_.notify_all();
  }

 private:
  size_t participant_count_;
  size_t arrived_{0};
  std::mutex latch_;
  std::condition_variable condition_;
  bool released_{false};
  bool cancelled_{false};
};

[[nodiscard]] auto FormatRankName(std::string_view group_name, size_t rank) -> std::string {
  std::string name{group_name};
  if (!name.empty()) {
    name.append("/rank_");
    name.append(std::to_string(rank));
  }
  return name;
}

void ValidateGroupContexts(std::vector<ExecutionContext> &contexts, std::source_location location) {
  if (contexts.empty()) {
    throw InvalidArgumentError("captured CUDA graph group requires at least one execution context", location);
  }
  const auto *runtime_state = internal::ContextAccess::GetRuntimeState(contexts.front(), location).get();
  std::unordered_set<int32_t> devices;
  devices.reserve(contexts.size());
  for (auto &context : contexts) {
    if (internal::ContextAccess::GetRuntimeState(context, location).get() != runtime_state) {
      throw InvalidArgumentError("captured CUDA graph group contexts must belong to one runtime", location);
    }
    if (!devices.insert(context.GetDevice().GetOrdinal()).second) {
      throw InvalidArgumentError("captured CUDA graph group requires one context per CUDA device", location);
    }
  }
}

}  // namespace

CaptureSession::CaptureSession(std::shared_ptr<internal::CaptureSessionState> state) noexcept
    : state_(std::move(state)) {}

CaptureSession::CaptureSession(CaptureSession &&other) noexcept : state_(std::move(other.state_)) {}

auto CaptureSession::operator=(CaptureSession &&other) noexcept -> CaptureSession & {
  if (this != &other) {
    Abort();
    state_ = std::move(other.state_);
  }
  return *this;
}

CaptureSession::~CaptureSession() noexcept { Abort(); }

auto CaptureSession::Finish(std::source_location location) -> CapturedGraph {
  if (state_ == nullptr) {
    throw CaptureError("cannot finish a moved-from CUDA graph capture session", location);
  }
  auto graph = CapturedGraph{state_->Finish(location)};
  state_.reset();
  return graph;
}

void CaptureSession::Abort() noexcept {
  if (state_ != nullptr) {
    state_->Abort();
    state_.reset();
  }
}

auto CaptureSession::IsActive() const noexcept -> bool { return state_ != nullptr && state_->IsActive(); }

CapturedGraph::CapturedGraph(std::unique_ptr<internal::CapturedGraphState> state) noexcept : state_(std::move(state)) {}

CapturedGraph::CapturedGraph(CapturedGraph &&) noexcept = default;

auto CapturedGraph::operator=(CapturedGraph &&) noexcept -> CapturedGraph & = default;

CapturedGraph::~CapturedGraph() noexcept = default;

void CapturedGraph::Launch(ExecutionContext &context, std::source_location location) {
  if (state_ == nullptr) {
    throw InvalidArgumentError("cannot launch a moved-from captured CUDA graph", location);
  }
  state_->Launch(context, location);
}

void CapturedGraph::DebugDumpDot(std::string_view path, std::source_location location) const {
  if (state_ == nullptr) {
    throw InvalidArgumentError("cannot dump a moved-from captured CUDA graph", location);
  }
  state_->DebugDumpDot(path, location);
}

auto CapturedGraph::GetDevice(std::source_location location) const -> Device {
  if (state_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph", location);
  }
  return state_->GetDevice();
}

auto CapturedGraph::GetStreamId(std::source_location location) const -> uint64_t {
  if (state_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph", location);
  }
  return state_->GetStreamId();
}

auto CapturedGraph::GetNodeCount(std::source_location location) const -> size_t {
  if (state_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph", location);
  }
  return state_->GetNodeCount();
}

auto CapturedGraph::GetLaunchCount(std::source_location location) const -> uint64_t {
  if (state_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph", location);
  }
  return state_->GetLaunchCount();
}

auto CapturedGraph::GetName(std::source_location location) const -> std::string_view {
  if (state_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph", location);
  }
  return state_->GetName();
}

class CapturedGraphGroup::Impl final {
 public:
  Impl(std::vector<ExecutionContext> contexts, std::string name)
      : contexts_(std::move(contexts)), graphs_(contexts_.size()), name_(std::move(name)) {
    workers_.reserve(contexts_.size());
    try {
      for (size_t rank = 0; rank < contexts_.size(); rank++) {
        workers_.emplace_back([this, rank] { WorkerLoop(rank); });
      }
    } catch (...) {
      StopWorkers();
      throw;
    }
  }

  Impl(const Impl &) = delete;
  auto operator=(const Impl &) -> Impl & = delete;

  ~Impl() noexcept { StopWorkers(); }

  void StopWorkers() noexcept {
    {
      const std::scoped_lock lock{latch_};
      stopping_ = true;
      command_generation_++;
    }
    command_.notify_all();
    for (auto &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void Capture(GraphCaptureFunction capture_function, std::source_location location) {
    capture_function_ = std::move(capture_function);
    begin_rendezvous_ = std::make_shared<CancellableRendezvous>(contexts_.size());
    finish_rendezvous_ = std::make_shared<CancellableRendezvous>(contexts_.size());
    try {
      Dispatch(Command::CAPTURE, location);
    } catch (...) {
      capture_function_ = {};
      begin_rendezvous_.reset();
      finish_rendezvous_.reset();
      throw;
    }
    capture_function_ = {};
    begin_rendezvous_.reset();
    finish_rendezvous_.reset();
    for (const auto &graph : graphs_) {
      if (!graph.has_value()) {
        failed_ = true;
        throw CaptureError("multi-GPU CUDA graph capture did not produce every rank graph", location);
      }
    }
  }

  void Launch(std::source_location location) {
    if (failed_) {
      throw CaptureError("captured CUDA graph group is in a failed state", location);
    }
    if (launch_count_ == std::numeric_limits<uint64_t>::max()) {
      throw OverflowError("captured CUDA graph group launch count overflow", location);
    }
    Dispatch(Command::LAUNCH, location);
    launch_count_++;
  }

  void Synchronize(std::source_location location) {
    const std::scoped_lock dispatch_lock{dispatch_latch_};
    for (auto &context : contexts_) {
      context.Synchronize(location);
    }
  }

  [[nodiscard]] auto GetContext(size_t rank, std::source_location location) -> ExecutionContext & {
    ValidateRank(rank, location);
    return contexts_[rank];
  }

  [[nodiscard]] auto GetDevice(size_t rank, std::source_location location) const -> Device {
    ValidateRank(rank, location);
    return contexts_[rank].GetDevice();
  }

  [[nodiscard]] auto GetNodeCount(size_t rank, std::source_location location) const -> size_t {
    ValidateRank(rank, location);
    if (!graphs_[rank].has_value()) {
      throw CaptureError("captured CUDA graph group rank has no executable graph", location);
    }
    return graphs_[rank]->GetNodeCount();
  }

  [[nodiscard]] auto GetWorldSize() const noexcept -> size_t { return contexts_.size(); }
  [[nodiscard]] auto GetLaunchCount() const noexcept -> uint64_t { return launch_count_; }
  [[nodiscard]] auto GetName() const noexcept -> std::string_view { return name_; }

 private:
  enum class Command : uint8_t {
    NONE,
    CAPTURE,
    LAUNCH,
  };

  void Dispatch(Command command, std::source_location location) {
    const std::scoped_lock dispatch_lock{dispatch_latch_};
    std::unique_lock lock{latch_};
    completed_worker_count_ = 0;
    worker_errors_.assign(contexts_.size(), nullptr);
    command_location_ = location;
    current_command_ = command;
    command_generation_++;
    command_.notify_all();
    completion_.wait(lock, [&] { return completed_worker_count_ == contexts_.size(); });
    current_command_ = Command::NONE;

    for (const auto &error : worker_errors_) {
      if (error != nullptr) {
        failed_ = true;
        std::rethrow_exception(error);
      }
    }
  }

  void ValidateRank(size_t rank, std::source_location location) const {
    if (rank >= contexts_.size()) {
      throw InvalidArgumentError("captured CUDA graph group rank is out of range", location);
    }
  }

  void WorkerLoop(size_t rank) noexcept {
    auto observed_generation = uint64_t{0};
    while (true) {
      Command command = Command::NONE;
      std::source_location location;
      {
        std::unique_lock lock{latch_};
        command_.wait(lock, [&] { return stopping_ || command_generation_ != observed_generation; });
        if (stopping_) {
          return;
        }
        observed_generation = command_generation_;
        command = current_command_;
        location = command_location_;
      }

      std::exception_ptr error;
      try {
        if (command == Command::CAPTURE) {
          CaptureRank(rank, location);
        } else if (command == Command::LAUNCH) {
          if (!graphs_[rank].has_value()) {
            throw CaptureError("captured CUDA graph group rank has no executable graph", location);
          }
          graphs_[rank]->Launch(contexts_[rank], location);
        } else {
          throw InternalError("captured CUDA graph worker received an invalid command", location);
        }
      } catch (...) {
        error = std::current_exception();
      }

      {
        const std::scoped_lock lock{latch_};
        worker_errors_[rank] = error;
        completed_worker_count_++;
        if (completed_worker_count_ == contexts_.size()) {
          completion_.notify_one();
        }
      }
    }
  }

  void CaptureRank(size_t rank, std::source_location location) {
    std::optional<CaptureSession> session;
    try {
      session.emplace(
          contexts_[rank].BeginCapture(GraphCaptureOptions{.name_ = FormatRankName(name_, rank)}, location));
      if (!begin_rendezvous_->ArriveAndWait()) {
        session->Abort();
        return;
      }
      capture_function_(rank, contexts_[rank]);
      if (!finish_rendezvous_->ArriveAndWait()) {
        session->Abort();
        return;
      }
      graphs_[rank].emplace(session->Finish(location));
    } catch (...) {
      begin_rendezvous_->Cancel();
      finish_rendezvous_->Cancel();
      if (session.has_value()) {
        session->Abort();
      }
      throw;
    }
  }

  std::vector<ExecutionContext> contexts_;
  std::vector<std::optional<CapturedGraph>> graphs_;
  std::string name_;
  std::vector<std::thread> workers_;
  std::vector<std::exception_ptr> worker_errors_;
  GraphCaptureFunction capture_function_;
  std::shared_ptr<CancellableRendezvous> begin_rendezvous_;
  std::shared_ptr<CancellableRendezvous> finish_rendezvous_;
  std::mutex dispatch_latch_;
  std::mutex latch_;
  std::condition_variable command_;
  std::condition_variable completion_;
  std::source_location command_location_;
  Command current_command_{Command::NONE};
  uint64_t command_generation_{0};
  uint64_t launch_count_{0};
  size_t completed_worker_count_{0};
  bool stopping_{false};
  bool failed_{false};
};

auto CapturedGraphGroup::Capture(std::vector<ExecutionContext> contexts, GraphCaptureFunction capture_function,
                                 const GraphGroupCaptureOptions &options, std::source_location location)
    -> CapturedGraphGroup {
  ValidateGroupContexts(contexts, location);
  if (!capture_function) {
    throw InvalidArgumentError("captured CUDA graph group requires a capture function", location);
  }

  auto impl = std::make_unique<Impl>(std::move(contexts), options.name_);
  impl->Capture(std::move(capture_function), location);
  return CapturedGraphGroup{std::move(impl)};
}

CapturedGraphGroup::CapturedGraphGroup(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

CapturedGraphGroup::CapturedGraphGroup(CapturedGraphGroup &&) noexcept = default;

auto CapturedGraphGroup::operator=(CapturedGraphGroup &&) noexcept -> CapturedGraphGroup & = default;

CapturedGraphGroup::~CapturedGraphGroup() noexcept = default;

void CapturedGraphGroup::Launch(std::source_location location) {
  if (impl_ == nullptr) {
    throw InvalidArgumentError("cannot launch a moved-from captured CUDA graph group", location);
  }
  impl_->Launch(location);
}

void CapturedGraphGroup::Synchronize(std::source_location location) {
  if (impl_ == nullptr) {
    throw InvalidArgumentError("cannot synchronize a moved-from captured CUDA graph group", location);
  }
  impl_->Synchronize(location);
}

auto CapturedGraphGroup::GetWorldSize(std::source_location location) const -> size_t {
  if (impl_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph group", location);
  }
  return impl_->GetWorldSize();
}

auto CapturedGraphGroup::GetContext(size_t rank, std::source_location location) -> ExecutionContext & {
  if (impl_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph group", location);
  }
  return impl_->GetContext(rank, location);
}

auto CapturedGraphGroup::GetDevice(size_t rank, std::source_location location) const -> Device {
  if (impl_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph group", location);
  }
  return impl_->GetDevice(rank, location);
}

auto CapturedGraphGroup::GetNodeCount(size_t rank, std::source_location location) const -> size_t {
  if (impl_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph group", location);
  }
  return impl_->GetNodeCount(rank, location);
}

auto CapturedGraphGroup::GetLaunchCount(std::source_location location) const -> uint64_t {
  if (impl_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph group", location);
  }
  return impl_->GetLaunchCount();
}

auto CapturedGraphGroup::GetName(std::source_location location) const -> std::string_view {
  if (impl_ == nullptr) {
    throw InvalidArgumentError("cannot query a moved-from captured CUDA graph group", location);
  }
  return impl_->GetName();
}

}  // namespace ttl
