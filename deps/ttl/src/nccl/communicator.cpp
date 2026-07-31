#include "ttl/communicator.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <nccl.h>

#include "ttl/device.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/communicator.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/device_allocator.hpp"
#include "ttl/internal/device_guard.hpp"
#include "ttl/internal/nccl_api.hpp"
#include "ttl/internal/runtime.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/runtime.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {
namespace {

constexpr int32_t MINIMUM_NCCL_VERSION = 21'403;
constexpr auto POLL_INTERVAL = std::chrono::milliseconds{1};

[[nodiscard]] auto FormatRankOutOfRange(size_t rank, size_t world_size) -> std::string {
  std::string message{"communicator rank "};
  message.append(std::to_string(rank));
  message.append(" is outside [0, ");
  message.append(std::to_string(world_size));
  message.push_back(')');
  return message;
}

[[nodiscard]] auto FormatBusyRank(size_t rank) -> std::string {
  std::string message{"communicator rank "};
  message.append(std::to_string(rank));
  message.append(" is already in use by another host submission");
  return message;
}

[[nodiscard]] auto FormatNcclVersion(int32_t version) -> std::string {
  std::string message{"TTL requires NCCL 2.14.3 or newer, but found version "};
  message.append(std::to_string(version));
  return message;
}

[[nodiscard]] auto IsPending(ncclResult_t status) noexcept -> bool {
  return status == ncclSuccess || status == ncclInProgress;
}

void ValidateOptions(const NcclOptions &options, std::source_location location) {
  if (options.initialization_timeout_.count() <= 0) {
    throw InvalidArgumentError("NCCL initialization timeout must be positive", location);
  }
  if (options.enqueue_timeout_.count() <= 0) {
    throw InvalidArgumentError("NCCL enqueue timeout must be positive", location);
  }
  if (options.finalize_timeout_.count() <= 0) {
    throw InvalidArgumentError("NCCL finalize timeout must be positive", location);
  }
}

void ReportCommunicatorLeak(ErrorSink &error_sink, std::source_location location) noexcept {
  try {
    error_sink.Report(ErrorRecord{
        .code_ = ErrorCode::NCCL,
        .message_ = "NCCL communicator resources could not be aborted during cleanup",
        .device_ = std::nullopt,
        .stream_id_ = std::nullopt,
        .location_ = location,
    });
  } catch (...) {
    return;
  }
}

}  // namespace

CommunicatorOperationLease::CommunicatorOperationLease(std::shared_ptr<CommunicatorGroupState> state,
                                                       std::vector<size_t> ranks) noexcept
    : state_(std::move(state)), ranks_(std::move(ranks)) {}

CommunicatorOperationLease::CommunicatorOperationLease(CommunicatorOperationLease &&other) noexcept
    : state_(std::move(other.state_)), ranks_(std::move(other.ranks_)) {}

CommunicatorOperationLease::~CommunicatorOperationLease() noexcept {
  if (state_ != nullptr) {
    state_->Release(ranks_);
  }
}

auto CommunicatorOperationLease::GetHandle(size_t rank) const noexcept -> ncclComm_t { return state_->handles_[rank]; }

auto CommunicatorOperationLease::GetRanks() const noexcept -> std::span<const size_t> { return ranks_; }

CommunicatorGroupState::CommunicatorGroupState(std::shared_ptr<RuntimeState> runtime_state,
                                               std::shared_ptr<ErrorSink> error_sink, std::vector<Device> rank_order,
                                               NcclOptions options, std::source_location location) noexcept
    : runtime_state_(std::move(runtime_state)),
      error_sink_(std::move(error_sink)),
      rank_order_(std::move(rank_order)),
      handles_(rank_order_.size(), nullptr),
      barrier_storage_(rank_order_.size()),
      options_(options),
      location_(location),
      rank_in_use_(rank_order_.size(), uint8_t{0}) {}

auto CommunicatorGroupState::Create(const std::shared_ptr<RuntimeState> &runtime_state,
                                    std::span<const Device> rank_order, const NcclOptions &options,
                                    std::source_location location) -> std::shared_ptr<CommunicatorGroupState> {
  if (runtime_state == nullptr) {
    throw InvalidArgumentError("communicator runtime state must not be null", location);
  }
  ValidateOptions(options, location);
  if (rank_order.empty()) {
    throw InvalidArgumentError("communicator group must contain at least one rank", location);
  }
  if (rank_order.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw OverflowError("communicator world size exceeds the NCCL rank range", location);
  }
  for (size_t rank = 0; rank < rank_order.size(); rank++) {
    if (runtime_state->GetDeviceContext(rank_order[rank], location) == nullptr) {
      throw InternalError("runtime returned a null device context", location);
    }
    if (std::ranges::find(rank_order.begin() + static_cast<ptrdiff_t>(rank + 1), rank_order.end(), rank_order[rank]) !=
        rank_order.end()) {
      throw InvalidArgumentError("communicator rank order must contain unique CUDA devices", location);
    }
  }

  int version = 0;
  CheckNccl(GetNcclApi().get_version_(&version), "ncclGetVersion", location);
  if (version < MINIMUM_NCCL_VERSION) {
    throw NotSupportedError(FormatNcclVersion(version), location);
  }

  auto state = std::shared_ptr<CommunicatorGroupState>{
      new CommunicatorGroupState{runtime_state, runtime_state->GetErrorSink(),
                                 std::vector<Device>{rank_order.begin(), rank_order.end()}, options, location}};
  try {
    state->Initialize(location);
    state->InitializeBarrierStorage(location);
  } catch (...) {
    state->MarkFailed();
    state->AbortHandlesNoexcept();
    throw;
  }
  return state;
}

CommunicatorGroupState::~CommunicatorGroupState() noexcept {
  Abort();
  if (HasNativeResources()) {
    ReportCommunicatorLeak(*error_sink_, location_);
  }
  for (auto &event : barrier_events_) {
    event.Discard();
  }
}

void CommunicatorGroupState::Initialize(std::source_location location) {
  ncclUniqueId unique_id{};
  const auto &nccl_api = GetNcclApi();
  CheckNccl(nccl_api.get_unique_id_(&unique_id), "ncclGetUniqueId", location);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  std::vector<ncclResult_t> statuses(rank_order_.size(), ncclSuccess);

  const auto start_status = nccl_api.group_start_();
  if (start_status != ncclSuccess) {
    CheckNccl(start_status, "ncclGroupStart (communicator initialization)", location);
  }

  const auto &cuda_api = GetCudaApi();
  int previous_device = -1;
  auto get_device_status = cuda_api.get_device_(&previous_device);
  cudaError_t set_device_status = cudaSuccess;
  if (get_device_status == cudaSuccess) {
    for (size_t rank = 0; rank < rank_order_.size(); rank++) {
      set_device_status = cuda_api.set_device_(rank_order_[rank].GetOrdinal());
      if (set_device_status != cudaSuccess) {
        break;
      }
      statuses[rank] = nccl_api.comm_init_rank_config_(&handles_[rank], static_cast<int>(rank_order_.size()), unique_id,
                                                       static_cast<int>(rank), &config);
    }
  }
  const auto end_status = nccl_api.group_end_();
  const auto restore_status = get_device_status == cudaSuccess ? cuda_api.set_device_(previous_device) : cudaSuccess;
  const auto native_resource_count =
      static_cast<size_t>(std::ranges::count_if(handles_, [](ncclComm_t handle) { return handle != nullptr; }));
  native_resource_count_.store(native_resource_count, std::memory_order_release);

  CheckCuda(get_device_status, "cudaGetDevice (NCCL communicator initialization)", location);
  CheckCuda(set_device_status, "cudaSetDevice (NCCL communicator initialization)", location);
  CheckCuda(restore_status, "cudaSetDevice (restore after NCCL communicator initialization)", location);
  for (const auto status : statuses) {
    if (!IsPending(status)) {
      CheckNccl(status, "ncclCommInitRankConfig", location);
    }
  }
  if (!IsPending(end_status)) {
    CheckNccl(end_status, "ncclGroupEnd (communicator initialization)", location);
  }
  if (native_resource_count != handles_.size()) {
    throw InternalError("NCCL initialization returned a null communicator", location);
  }

  WaitForProgress({}, options_.initialization_timeout_, "NCCL communicator initialization", location);
  status_.store(CommunicatorStatus::READY, std::memory_order_release);
}

void CommunicatorGroupState::InitializeBarrierStorage(std::source_location location) {
  barrier_events_.reserve(rank_order_.size());
  for (size_t rank = 0; rank < rank_order_.size(); rank++) {
    auto stream = StreamAccess::CreateOwned(rank_order_[rank], 0, error_sink_, location);
    barrier_events_.push_back(
        runtime_state_->GetDeviceContext(rank_order_[rank], location)->GetEventPool()->Acquire(location));
    barrier_storage_[rank] = runtime_state_->GetDeviceContext(rank_order_[rank], location)
                                 ->GetAllocator()
                                 ->Allocate(stream, sizeof(uint8_t), alignof(uint8_t),
                                            AllocationContext{
                                                .operation_ = "NCCL barrier storage",
                                                .output_shape_ = std::nullopt,
                                                .dtype_ = DType::UINT8,
                                                .location_ = location,
                                            });
    DeviceGuard device_guard{rank_order_[rank], *error_sink_, location};
    const auto native_stream = StreamAccess::GetNative(stream);
    CheckCuda(GetCudaApi().memset_async_(barrier_storage_[rank]->GetBasePointer(), 0, sizeof(uint8_t), native_stream),
              "cudaMemsetAsync (NCCL barrier storage)", location);
    CheckCuda(GetCudaApi().record_event_(barrier_events_[rank].GetNative(), native_stream),
              "cudaEventRecord (NCCL barrier initialization)", location);
  }
}

auto CommunicatorGroupState::GetWorldSize() const noexcept -> size_t { return rank_order_.size(); }

auto CommunicatorGroupState::GetDevice(size_t rank) const noexcept -> Device { return rank_order_[rank]; }

auto CommunicatorGroupState::GetRankOrder() const noexcept -> std::span<const Device> { return rank_order_; }

auto CommunicatorGroupState::GetStatus() const noexcept -> CommunicatorStatus {
  return status_.load(std::memory_order_acquire);
}

auto CommunicatorGroupState::BelongsTo(const std::shared_ptr<RuntimeState> &runtime_state) const noexcept -> bool {
  return runtime_state_.get() == runtime_state.get();
}

auto CommunicatorGroupState::HasNativeResources() const noexcept -> bool {
  return native_resource_count_.load(std::memory_order_acquire) != 0;
}

auto CommunicatorGroupState::HasGraphReferences() const noexcept -> bool {
  const std::scoped_lock lock{lifecycle_latch_};
  return graph_reference_count_ != 0;
}

void CommunicatorGroupState::ValidateRank(size_t rank, std::source_location location) const {
  if (rank >= rank_order_.size()) {
    throw InvalidArgumentError(FormatRankOutOfRange(rank, rank_order_.size()), location);
  }
}

auto CommunicatorGroupState::AcquireRank(size_t rank, std::source_location location) -> CommunicatorOperationLease {
  ValidateRank(rank, location);
  return Acquire(std::vector<size_t>{rank}, location);
}

auto CommunicatorGroupState::AcquireAll(std::source_location location) -> CommunicatorOperationLease {
  std::vector<size_t> ranks(rank_order_.size());
  for (size_t rank = 0; rank < ranks.size(); rank++) {
    ranks[rank] = rank;
  }
  return Acquire(std::move(ranks), location);
}

auto CommunicatorGroupState::Acquire(std::vector<size_t> ranks, std::source_location location)
    -> CommunicatorOperationLease {
  const std::scoped_lock lock{lifecycle_latch_};
  if (GetStatus() != CommunicatorStatus::READY) {
    throw InvalidArgumentError("communicator group is not ready for submission", location);
  }
  if (graph_reference_count_ != 0) {
    throw CaptureError("communicator group is retained by a captured CUDA graph", location);
  }
  for (const auto rank : ranks) {
    ValidateRank(rank, location);
    if (rank_in_use_[rank] != 0) {
      throw InvalidArgumentError(FormatBusyRank(rank), location);
    }
  }
  for (const auto rank : ranks) {
    rank_in_use_[rank] = uint8_t{1};
  }
  active_rank_count_ += ranks.size();
  return CommunicatorOperationLease{shared_from_this(), std::move(ranks)};
}

void CommunicatorGroupState::RegisterGraph(std::source_location location) {
  const std::scoped_lock lock{lifecycle_latch_};
  if (GetStatus() != CommunicatorStatus::READY) {
    throw InvalidArgumentError("only a ready communicator group can be retained by a CUDA graph", location);
  }
  if (graph_reference_count_ == std::numeric_limits<size_t>::max()) {
    throw OverflowError("communicator CUDA graph reference count overflow", location);
  }
  graph_reference_count_++;
}

void CommunicatorGroupState::UnregisterGraph() noexcept {
  bool should_abort = false;
  {
    const std::scoped_lock lock{lifecycle_latch_};
    if (graph_reference_count_ == 0) {
      std::terminate();
    }
    graph_reference_count_--;
    should_abort = graph_reference_count_ == 0 &&
                   (!public_owner_alive_ || abort_requested_ || GetStatus() != CommunicatorStatus::READY);
  }
  if (should_abort) {
    AbortHandlesNoexcept();
  }
}

void CommunicatorGroupState::ValidateGraphLaunch(std::source_location location) const {
  const std::scoped_lock lock{lifecycle_latch_};
  if (graph_reference_count_ == 0 || abort_requested_ || GetStatus() != CommunicatorStatus::READY) {
    throw CaptureError("captured CUDA graph communicator is not ready for replay", location);
  }
}

void CommunicatorGroupState::ReleasePublicOwner() noexcept {
  bool should_abort = false;
  {
    const std::scoped_lock lock{lifecycle_latch_};
    if (!public_owner_alive_) {
      return;
    }
    public_owner_alive_ = false;
    should_abort = graph_reference_count_ == 0;
  }
  if (should_abort) {
    AbortHandlesNoexcept();
  }
}

void CommunicatorGroupState::Release(std::span<const size_t> ranks) noexcept {
  bool should_abort = false;
  {
    const std::scoped_lock lock{lifecycle_latch_};
    for (const auto rank : ranks) {
      if (rank >= rank_in_use_.size() || rank_in_use_[rank] == 0 || active_rank_count_ == 0) {
        std::terminate();
      }
      rank_in_use_[rank] = uint8_t{0};
      active_rank_count_--;
    }
    should_abort = active_rank_count_ == 0 && GetStatus() == CommunicatorStatus::FAILED;
  }
  if (should_abort) {
    AbortHandlesNoexcept();
  }
}

void CommunicatorGroupState::WaitForProgress(std::span<const size_t> ranks, std::chrono::milliseconds timeout,
                                             std::string_view operation, std::source_location location) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    bool complete = true;
    if (ranks.empty()) {
      for (const auto handle : handles_) {
        ncclResult_t async_status = ncclSuccess;
        CheckNccl(GetNcclApi().comm_get_async_error_(handle, &async_status), "ncclCommGetAsyncError", location);
        if (async_status == ncclInProgress) {
          complete = false;
        } else if (async_status != ncclSuccess) {
          CheckNccl(async_status, operation, location);
        }
      }
    } else {
      for (const auto rank : ranks) {
        ncclResult_t async_status = ncclSuccess;
        CheckNccl(GetNcclApi().comm_get_async_error_(handles_[rank], &async_status), "ncclCommGetAsyncError", location);
        if (async_status == ncclInProgress) {
          complete = false;
        } else if (async_status != ncclSuccess) {
          CheckNccl(async_status, operation, location);
        }
      }
    }
    if (complete) {
      return;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      std::string message{operation};
      message.append(" timed out while waiting for NCCL progress");
      throw NcclError(std::move(message), location);
    }
    std::this_thread::sleep_for(POLL_INTERVAL);
  }
}

void CommunicatorGroupState::CheckSubmission(std::span<const size_t> ranks, ncclResult_t status,
                                             std::string_view operation, std::source_location location) {
  try {
    if (status == ncclInProgress) {
      WaitForProgress(ranks, options_.enqueue_timeout_, operation, location);
      return;
    }
    CheckNccl(status, operation, location);
  } catch (...) {
    MarkFailed();
    throw;
  }
}

void CommunicatorGroupState::CheckGroupedSubmission(std::span<const size_t> ranks,
                                                    std::span<const ncclResult_t> statuses, ncclResult_t end_status,
                                                    std::string_view operation, std::source_location location) {
  try {
    bool pending = end_status == ncclInProgress;
    if (!IsPending(end_status)) {
      CheckNccl(end_status, "ncclGroupEnd", location);
    }
    for (const auto status : statuses) {
      pending = pending || status == ncclInProgress;
      if (!IsPending(status)) {
        CheckNccl(status, operation, location);
      }
    }
    if (pending) {
      WaitForProgress(ranks, options_.enqueue_timeout_, operation, location);
    }
  } catch (...) {
    MarkFailed();
    throw;
  }
}

void CommunicatorGroupState::MarkFailed() noexcept {
  const auto current = GetStatus();
  if (current != CommunicatorStatus::CLOSED && current != CommunicatorStatus::ABORTED) {
    status_.store(CommunicatorStatus::FAILED, std::memory_order_release);
  }
}

auto CommunicatorGroupState::GetBarrierStorage(size_t rank) const noexcept -> const std::shared_ptr<Storage> & {
  return barrier_storage_[rank];
}

void CommunicatorGroupState::BeginBarrier(size_t rank, const Stream &stream, bool capture_external,
                                          std::source_location location) {
  DeviceGuard device_guard{stream.GetDevice(), *error_sink_, location};
  CheckCuda(GetCudaApi().stream_wait_event_(StreamAccess::GetNative(stream), barrier_events_[rank].GetNative(),
                                            capture_external ? cudaEventWaitExternal : cudaEventWaitDefault),
            "cudaStreamWaitEvent (NCCL barrier)", location);
  barrier_storage_[rank]->RecordUsage(stream);
}

void CommunicatorGroupState::EndBarrier(size_t rank, const Stream &stream, bool capture_external,
                                        std::source_location location) {
  try {
    DeviceGuard device_guard{stream.GetDevice(), *error_sink_, location};
    CheckCuda(
        GetCudaApi().record_event_with_flags_(barrier_events_[rank].GetNative(), StreamAccess::GetNative(stream),
                                              capture_external ? cudaEventRecordExternal : cudaEventRecordDefault),
        "cudaEventRecord (NCCL barrier)", location);
  } catch (...) {
    MarkFailed();
    throw;
  }
}

auto CommunicatorGroupState::TryAcquireAllForPoll() noexcept -> bool {
  const std::scoped_lock lock{lifecycle_latch_};
  if (active_rank_count_ != 0 || GetStatus() != CommunicatorStatus::READY) {
    return false;
  }
  std::ranges::fill(rank_in_use_, uint8_t{1});
  active_rank_count_ = rank_in_use_.size();
  return true;
}

void CommunicatorGroupState::ReleaseAllFromPoll() noexcept {
  const std::scoped_lock lock{lifecycle_latch_};
  std::ranges::fill(rank_in_use_, uint8_t{0});
  active_rank_count_ = 0;
}

void CommunicatorGroupState::Poll(std::source_location location) {
  if (!TryAcquireAllForPoll()) {
    return;
  }
  try {
    for (const auto handle : handles_) {
      ncclResult_t async_status = ncclSuccess;
      CheckNccl(GetNcclApi().comm_get_async_error_(handle, &async_status), "ncclCommGetAsyncError", location);
      if (async_status != ncclSuccess && async_status != ncclInProgress) {
        CheckNccl(async_status, "NCCL asynchronous operation", location);
      }
    }
  } catch (...) {
    MarkFailed();
    ReleaseAllFromPoll();
    AbortHandlesNoexcept();
    throw;
  }
  ReleaseAllFromPoll();
}

void CommunicatorGroupState::PollNoexcept() noexcept {
  if (!TryAcquireAllForPoll()) {
    return;
  }
  bool failed = false;
  for (size_t rank = 0; rank < handles_.size(); rank++) {
    ncclResult_t async_status = ncclSuccess;
    const auto query_status = GetNcclApi().comm_get_async_error_(handles_[rank], &async_status);
    const ErrorReportContext context{
        .location_ = location_,
        .device_ = rank_order_[rank],
        .stream_id_ = std::nullopt,
    };
    const auto query_succeeded = TryNccl(query_status, "ncclCommGetAsyncError", *error_sink_, context);
    failed = !query_succeeded || failed;
    if (query_succeeded && async_status != ncclSuccess && async_status != ncclInProgress) {
      failed = !TryNccl(async_status, "NCCL asynchronous operation", *error_sink_, context) || failed;
    }
  }
  if (failed) {
    MarkFailed();
  }
  ReleaseAllFromPoll();
  if (failed) {
    AbortHandlesNoexcept();
  }
}

void CommunicatorGroupState::Close(std::source_location location) {
  std::unique_lock native_lock{native_lifecycle_latch_};
  {
    const std::scoped_lock lock{lifecycle_latch_};
    if (GetStatus() == CommunicatorStatus::CLOSED) {
      return;
    }
    if (active_rank_count_ != 0) {
      throw InvalidArgumentError("cannot close communicator group while a host submission is active", location);
    }
    if (graph_reference_count_ != 0) {
      throw InvalidArgumentError("cannot close communicator group while captured CUDA graphs retain it", location);
    }
    if (GetStatus() != CommunicatorStatus::READY) {
      throw InvalidArgumentError("only a ready communicator group can be closed", location);
    }
    status_.store(CommunicatorStatus::FINALIZING, std::memory_order_release);
  }

  try {
    bool pending = false;
    for (size_t rank = 0; rank < handles_.size(); rank++) {
      DeviceGuard device_guard{rank_order_[rank], *error_sink_, location};
      const auto finalize_status = GetNcclApi().comm_finalize_(handles_[rank]);
      pending = pending || finalize_status == ncclInProgress;
      if (!IsPending(finalize_status)) {
        CheckNccl(finalize_status, "ncclCommFinalize", location);
      }
    }
    if (pending) {
      WaitForProgress({}, options_.finalize_timeout_, "NCCL communicator finalization", location);
    }
    for (size_t rank = 0; rank < handles_.size(); rank++) {
      DeviceGuard device_guard{rank_order_[rank], *error_sink_, location};
      CheckNccl(GetNcclApi().comm_destroy_(handles_[rank]), "ncclCommDestroy", location);
      handles_[rank] = nullptr;
      native_resource_count_.fetch_sub(1, std::memory_order_release);
    }
  } catch (...) {
    MarkFailed();
    native_lock.unlock();
    AbortHandlesNoexcept();
    throw;
  }
  for (auto &event : barrier_events_) {
    event.Discard();
  }
  barrier_events_.clear();
  barrier_storage_.clear();
  status_.store(CommunicatorStatus::CLOSED, std::memory_order_release);
}

void CommunicatorGroupState::AbortHandlesNoexcept() noexcept {
  const std::scoped_lock native_lock{native_lifecycle_latch_};
  {
    const std::scoped_lock lock{lifecycle_latch_};
    const auto current = GetStatus();
    if (current == CommunicatorStatus::CLOSED || current == CommunicatorStatus::ABORTED || abort_in_progress_) {
      return;
    }
    status_.store(CommunicatorStatus::FAILED, std::memory_order_release);
    abort_requested_ = true;
    if (active_rank_count_ != 0 || graph_reference_count_ != 0) {
      return;
    }
    abort_in_progress_ = true;
  }

  bool all_released = true;
  for (size_t rank = 0; rank < handles_.size(); rank++) {
    if (handles_[rank] == nullptr) {
      continue;
    }
    const ErrorReportContext context{
        .location_ = location_,
        .device_ = rank_order_[rank],
        .stream_id_ = std::nullopt,
    };
    CleanupDeviceGuard device_guard{rank_order_[rank], *error_sink_, context, "abort NCCL communicator",
                                    "restore after NCCL communicator abort"};
    if (!device_guard || !TryNccl(GetNcclApi().comm_abort_(handles_[rank]), "ncclCommAbort", *error_sink_, context)) {
      all_released = false;
      continue;
    }
    handles_[rank] = nullptr;
    native_resource_count_.fetch_sub(1, std::memory_order_release);
  }
  if (all_released) {
    for (auto &event : barrier_events_) {
      event.Discard();
    }
    barrier_events_.clear();
    barrier_storage_.clear();
  }
  {
    const std::scoped_lock lock{lifecycle_latch_};
    abort_in_progress_ = false;
    abort_requested_ = !all_released;
    status_.store(all_released ? CommunicatorStatus::ABORTED : CommunicatorStatus::FAILED, std::memory_order_release);
  }
}

void CommunicatorGroupState::Abort() noexcept { AbortHandlesNoexcept(); }

auto CommunicatorAccess::GetState(NcclCommunicator &communicator, std::source_location location)
    -> const std::shared_ptr<CommunicatorGroupState> & {
  if (communicator.state_ == nullptr) {
    throw InvalidArgumentError("communicator has been moved from", location);
  }
  return communicator.state_;
}

auto CommunicatorAccess::GetState(const NcclCommunicator &communicator, std::source_location location)
    -> const std::shared_ptr<CommunicatorGroupState> & {
  if (communicator.state_ == nullptr) {
    throw InvalidArgumentError("communicator has been moved from", location);
  }
  return communicator.state_;
}

auto CommunicatorAccess::GetRank(const NcclCommunicator &communicator, std::source_location location) -> size_t {
  const auto &state = GetState(communicator, location);
  if (state == nullptr) {
    throw InternalError("communicator state validation returned null", location);
  }
  return communicator.rank_;
}

}  // namespace ttl::internal

namespace ttl {

NcclCommunicator::NcclCommunicator(std::shared_ptr<internal::CommunicatorGroupState> state, size_t rank) noexcept
    : state_(std::move(state)), rank_(rank) {}

auto NcclCommunicator::GetRank() const noexcept -> int32_t { return static_cast<int32_t>(rank_); }

auto NcclCommunicator::GetWorldSize() const noexcept -> int32_t { return static_cast<int32_t>(state_->GetWorldSize()); }

auto NcclCommunicator::GetDevice() const noexcept -> Device { return state_->GetDevice(rank_); }

auto NcclCommunicator::GetStatus() const noexcept -> CommunicatorStatus { return state_->GetStatus(); }

void NcclCommunicator::PollAsyncError(std::source_location location) { state_->Poll(location); }

void NcclCommunicator::Close(std::source_location location) { state_->Close(location); }

void NcclCommunicator::Abort() noexcept {
  if (state_ != nullptr) {
    state_->Abort();
  }
}

LocalCommunicatorGroup::LocalCommunicatorGroup(std::shared_ptr<internal::CommunicatorGroupState> state,
                                               std::vector<NcclCommunicator> communicators) noexcept
    : state_(std::move(state)), communicators_(std::move(communicators)) {}

auto LocalCommunicatorGroup::Create(Runtime &runtime, std::span<const Device> rank_order, const NcclOptions &options,
                                    std::source_location location) -> LocalCommunicatorGroup {
  auto state =
      internal::RuntimeAccess::GetState(runtime, location)->CreateCommunicatorGroup(rank_order, options, location);
  std::vector<NcclCommunicator> communicators;
  communicators.reserve(rank_order.size());
  for (size_t rank = 0; rank < rank_order.size(); rank++) {
    communicators.push_back(NcclCommunicator{state, rank});
  }
  return LocalCommunicatorGroup{std::move(state), std::move(communicators)};
}

LocalCommunicatorGroup::~LocalCommunicatorGroup() noexcept {
  if (state_ != nullptr) {
    state_->ReleasePublicOwner();
  }
}

auto LocalCommunicatorGroup::GetWorldSize() const noexcept -> size_t { return communicators_.size(); }

auto LocalCommunicatorGroup::GetCommunicator(size_t rank, std::source_location location) -> NcclCommunicator & {
  if (rank >= communicators_.size()) {
    throw InvalidArgumentError("communicator rank is outside the local group", location);
  }
  return communicators_[rank];
}

auto LocalCommunicatorGroup::GetRankOrder() const noexcept -> std::span<const Device> { return state_->GetRankOrder(); }

auto LocalCommunicatorGroup::GetStatus() const noexcept -> CommunicatorStatus { return state_->GetStatus(); }

void LocalCommunicatorGroup::Poll(std::source_location location) { state_->Poll(location); }

void LocalCommunicatorGroup::Close(std::source_location location) { state_->Close(location); }

void LocalCommunicatorGroup::Abort() noexcept {
  if (state_ != nullptr) {
    state_->Abort();
  }
}

}  // namespace ttl
