#include "ttl/internal/device_allocator.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <source_location>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/allocation.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/device_guard.hpp"
#include "ttl/internal/event_pool.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/shape.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {
namespace {

constexpr size_t MAXIMUM_DEVICE_ALIGNMENT = 256;
constexpr size_t INITIAL_RETIREMENT_CAPACITY = 16;
constexpr auto MAINTENANCE_INTERVAL = std::chrono::milliseconds{1};

enum class AllocatorStatus : uint8_t {
  RUNNING,
  CLOSING,
  CLOSED,
  FAILED,
};

class OutstandingStorageGuard final {
 public:
  OutstandingStorageGuard(std::atomic<uint64_t> &count, bool active) noexcept : count_(count), active_(active) {}

  OutstandingStorageGuard(const OutstandingStorageGuard &) = delete;
  auto operator=(const OutstandingStorageGuard &) -> OutstandingStorageGuard & = delete;
  OutstandingStorageGuard(OutstandingStorageGuard &&) = delete;
  auto operator=(OutstandingStorageGuard &&) -> OutstandingStorageGuard & = delete;

  ~OutstandingStorageGuard() noexcept {
    if (active_ && count_.fetch_sub(1, std::memory_order_release) == 0) {
      std::terminate();
    }
  }

 private:
  std::atomic<uint64_t> &count_;
  bool active_;
};

void ValidateErrorSink(const std::shared_ptr<ErrorSink> &error_sink, std::source_location location) {
  if (error_sink == nullptr) {
    throw InvalidArgumentError("device allocator error sink must not be null", location);
  }
}

void ValidateEventPool(Device device, const std::shared_ptr<EventPool> &event_pool, std::source_location location) {
  if (event_pool == nullptr) {
    throw InvalidArgumentError("device allocator event pool must not be null", location);
  }
  if (event_pool->GetDevice() != device) {
    throw InvalidArgumentError("device allocator and event pool must belong to the same device", location);
  }
}

void ValidateAlignment(size_t alignment, std::source_location location) {
  if (!std::has_single_bit(alignment) || alignment > MAXIMUM_DEVICE_ALIGNMENT) {
    std::string message{"device allocation alignment must be a power of two no greater than "};
    message.append(std::to_string(MAXIMUM_DEVICE_ALIGNMENT));
    throw InvalidArgumentError(std::move(message), location);
  }
}

[[nodiscard]] auto FormatWrongStreamDevice(Device allocator_device, Device stream_device) -> std::string {
  std::string message{"device allocator for "};
  message.append(allocator_device.ToString());
  message.append(" cannot allocate on a stream belonging to ");
  message.append(stream_device.ToString());
  return message;
}

[[nodiscard]] auto FormatOutstandingStorage(size_t count) -> std::string {
  std::string message{"cannot shut down device allocator while "};
  message.append(std::to_string(count));
  message.append(" Storage object");
  if (count != 1) {
    message.push_back('s');
  }
  message.append(" remain alive");
  return message;
}

void ReportError(ErrorCode code, std::string_view message, ErrorSink &error_sink,
                 const ErrorReportContext &context) noexcept {
  try {
    error_sink.Report(ErrorRecord{
        .code_ = code,
        .message_ = std::string{message},
        .device_ = context.device_,
        .stream_id_ = context.stream_id_,
        .location_ = context.location_,
    });
  } catch (...) {
    return;
  }
}

[[nodiscard]] auto MakeContext(Device device, std::source_location location,
                               std::optional<uint64_t> stream_id = std::nullopt) noexcept -> ErrorReportContext {
  return ErrorReportContext{
      .location_ = location,
      .device_ = device,
      .stream_id_ = stream_id,
  };
}

}  // namespace

struct RetirementRecord final {
  RetirementRecord(Allocation allocation, std::shared_ptr<StreamState> allocation_stream,
                   std::vector<std::shared_ptr<StreamState>> side_streams) noexcept
      : allocation_(std::move(allocation)),
        allocation_stream_(std::move(allocation_stream)),
        side_streams_(std::move(side_streams)) {}

  RetirementRecord(const RetirementRecord &) = delete;
  auto operator=(const RetirementRecord &) -> RetirementRecord & = delete;
  RetirementRecord(RetirementRecord &&) noexcept = default;
  auto operator=(RetirementRecord &&) noexcept -> RetirementRecord & = default;

  Allocation allocation_;
  std::shared_ptr<StreamState> allocation_stream_;
  std::vector<std::shared_ptr<StreamState>> side_streams_;
  std::optional<PooledEvent> completion_event_;
  bool free_attempted_{false};
  bool free_submitted_{false};
  bool uses_reclaim_stream_{false};
  bool poisoned_{false};
};

class DeviceAllocatorImpl final {
 public:
  DeviceAllocatorImpl(Device device, std::shared_ptr<ErrorSink> error_sink, std::shared_ptr<EventPool> event_pool,
                      DeviceAllocatorOptions options, cudaMemPool_t pool, std::shared_ptr<StreamState> reclaim_stream,
                      std::source_location location) noexcept
      : device_(device),
        error_sink_(std::move(error_sink)),
        event_pool_(std::move(event_pool)),
        options_(options),
        pool_(pool),
        reclaim_stream_(std::move(reclaim_stream)),
        location_(location) {}

  DeviceAllocatorImpl(const DeviceAllocatorImpl &) = delete;
  auto operator=(const DeviceAllocatorImpl &) -> DeviceAllocatorImpl & = delete;
  DeviceAllocatorImpl(DeviceAllocatorImpl &&) = delete;
  auto operator=(DeviceAllocatorImpl &&) -> DeviceAllocatorImpl & = delete;

  ~DeviceAllocatorImpl() noexcept {
    StopWorkerNoexcept();
    if (pool_ != nullptr) {
      TryDestroyPool();
    }
  }

  void StartWorker() {
    if (!options_.enable_maintenance_thread_) {
      return;
    }
    maintenance_worker_ = std::jthread{[this](const std::stop_token &stop_token) { MaintenanceLoop(stop_token); }};
  }

  [[nodiscard]] auto Allocate(const std::shared_ptr<DeviceAllocator> &allocator, const Stream &stream, size_t bytes,
                              size_t alignment, const AllocationContext &context) -> std::shared_ptr<Storage> {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    ValidateRunning(context.location_);
    if (stream.GetDevice() != device_) {
      throw InvalidArgumentError(FormatWrongStreamDevice(device_, stream.GetDevice()), context.location_);
    }
    ValidateAlignment(alignment, context.location_);

    const auto allocation_stream = StreamAccess::GetState(stream);
    if (bytes == 0) {
      auto storage = MakeUniqueStorage(Allocation{nullptr, 0, AllocationKind::DEVICE_POOL, nullptr, context.location_},
                                       allocator, allocation_stream);
      outstanding_storage_count_.fetch_add(1, std::memory_order_relaxed);
      return std::shared_ptr<Storage>{std::move(storage)};
    }

    const auto requested_bytes = CheckedNarrow<uint64_t>(bytes, "device allocation byte count", context.location_);
    ReserveRetirementSlot(context.location_);
    try {
      ReserveBudget(requested_bytes, context);
    } catch (...) {
      CancelRetirementSlot();
      throw;
    }

    void *pointer = nullptr;
    cudaError_t status = cudaSuccess;
    try {
      DeviceGuard device_guard{device_, *error_sink_, context.location_};
      status = GetCudaApi().malloc_from_pool_async_(&pointer, bytes, pool_, allocation_stream->GetNative());
      if (status == cudaErrorMemoryAllocation) {
        const auto last_error = GetCudaApi().get_last_error_();
        if (last_error != cudaSuccess && last_error != cudaErrorMemoryAllocation) {
          CheckCuda(last_error, "cudaGetLastError", context.location_);
        }
        retry_count_.fetch_add(1, std::memory_order_relaxed);
        PollRetirements();
        const auto target = GetOomTrimTarget(requested_bytes);
        CheckCuda(
            GetCudaApi().trim_memory_pool_(pool_, CheckedNarrow<size_t>(target, "OOM trim target", context.location_)),
            "cudaMemPoolTrimTo", context.location_);
        trim_count_.fetch_add(1, std::memory_order_relaxed);
        status = GetCudaApi().malloc_from_pool_async_(&pointer, bytes, pool_, allocation_stream->GetNative());
      }
    } catch (...) {
      ReleaseBudget(requested_bytes);
      CancelRetirementSlot();
      throw;
    }

    if (status != cudaSuccess) {
      ReleaseBudget(requested_bytes);
      CancelRetirementSlot();
      if (status == cudaErrorMemoryAllocation) {
        oom_count_.fetch_add(1, std::memory_order_relaxed);
        throw OutOfMemoryError(FormatOutOfMemory(bytes, alignment, context), context.location_);
      }
      CheckCuda(status, "cudaMallocFromPoolAsync", context.location_);
    }
    if (pointer == nullptr) {
      ReleaseBudget(requested_bytes);
      CancelRetirementSlot();
      throw InternalError("cudaMallocFromPoolAsync returned a null pointer", context.location_);
    }

    allocation_count_.fetch_add(1, std::memory_order_relaxed);
    std::unique_ptr<Storage> storage;
    try {
      storage = MakeUniqueStorage(Allocation{pointer, bytes, AllocationKind::DEVICE_POOL, nullptr, context.location_},
                                  allocator, allocation_stream);
    } catch (...) {
      Retire(Allocation{pointer, bytes, AllocationKind::DEVICE_POOL, nullptr, context.location_}, allocation_stream, {},
             false);
      throw;
    }

    logical_live_bytes_.fetch_add(requested_bytes, std::memory_order_relaxed);
    outstanding_storage_count_.fetch_add(1, std::memory_order_relaxed);
    return std::shared_ptr<Storage>{std::move(storage)};
  }

  [[nodiscard]] auto WrapExternal(const std::shared_ptr<DeviceAllocator> &allocator, const Stream &allocation_stream,
                                  void *pointer, size_t capacity_bytes, ExternalOwnership ownership,
                                  std::shared_ptr<void> owner, std::source_location location)
      -> std::shared_ptr<Storage> {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    ValidateRunning(location);
    if (allocation_stream.GetDevice() != device_) {
      throw InvalidArgumentError(FormatWrongStreamDevice(device_, allocation_stream.GetDevice()), location);
    }
    ValidateExternal(pointer, capacity_bytes, ownership, owner, location);

    const auto kind =
        ownership == ExternalOwnership::BORROWED ? AllocationKind::EXTERNAL_BORROWED : AllocationKind::EXTERNAL_OWNED;
    const auto needs_retirement = kind == AllocationKind::EXTERNAL_OWNED && capacity_bytes != 0;
    if (needs_retirement) {
      ReserveRetirementSlot(location);
    }

    auto stream_state = StreamAccess::GetState(allocation_stream);
    std::unique_ptr<Storage> storage;
    try {
      storage = MakeUniqueStorage(Allocation{pointer, capacity_bytes, kind, std::move(owner), location}, allocator,
                                  stream_state);
    } catch (...) {
      if (needs_retirement) {
        CancelRetirementSlot();
      }
      throw;
    }

    outstanding_storage_count_.fetch_add(1, std::memory_order_relaxed);
    return std::shared_ptr<Storage>{std::move(storage)};
  }

  void Retire(Allocation allocation, std::shared_ptr<StreamState> allocation_stream,
              std::vector<std::shared_ptr<StreamState>> side_streams, bool counted_storage = true) noexcept {
    const OutstandingStorageGuard outstanding_storage_guard{outstanding_storage_count_, counted_storage};
    const auto capacity_bytes = allocation.capacity_bytes_;
    const auto kind = allocation.kind_;

    if (capacity_bytes == 0 || kind == AllocationKind::EXTERNAL_BORROWED) {
      return;
    }

    const auto bytes = static_cast<uint64_t>(capacity_bytes);
    if (kind == AllocationKind::DEVICE_POOL && counted_storage) {
      logical_live_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    }
    if (kind == AllocationKind::DEVICE_POOL) {
      retiring_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }

    RetirementRecord record{std::move(allocation), std::move(allocation_stream), std::move(side_streams)};
    const auto report_context = MakeContext(device_, record.allocation_.location_, record.allocation_stream_->GetId());
    try {
      record.completion_event_.emplace(event_pool_->Acquire(record.allocation_.location_));
    } catch (const Error &error) {
      ReportError(error.GetCode(), error.GetMessage(), *error_sink_, report_context);
      PoisonAndEnqueue(std::move(record));
      return;
    } catch (const std::exception &error) {
      ReportError(ErrorCode::INTERNAL, error.what(), *error_sink_, report_context);
      PoisonAndEnqueue(std::move(record));
      return;
    } catch (...) {
      ReportError(ErrorCode::INTERNAL, "unknown failure while acquiring a retirement event", *error_sink_,
                  report_context);
      PoisonAndEnqueue(std::move(record));
      return;
    }

    auto target_stream = record.allocation_stream_;
    if (!record.side_streams_.empty()) {
      record.uses_reclaim_stream_ = true;
      if (!SubmitDependency(record.allocation_stream_, record.allocation_.location_) ||
          !std::ranges::all_of(record.side_streams_, [this, &record](const auto &stream) {
            return SubmitDependency(stream, record.allocation_.location_);
          })) {
        record.completion_event_.reset();
        PoisonAndEnqueue(std::move(record));
        return;
      }
      target_stream = reclaim_stream_;
    }

    const ErrorReportContext target_context{
        .location_ = record.allocation_.location_,
        .device_ = target_stream->GetDevice(),
        .stream_id_ = target_stream->GetId(),
    };
    CleanupDeviceGuard device_guard{target_stream->GetDevice(), *error_sink_, target_context,
                                    "submit allocation retirement", "restore after allocation retirement"};
    if (!device_guard) {
      record.completion_event_.reset();
      PoisonAndEnqueue(std::move(record));
      return;
    }

    if (kind == AllocationKind::DEVICE_POOL) {
      record.free_attempted_ = true;
      const auto free_status = GetCudaApi().free_async_(record.allocation_.pointer_, target_stream->GetNative());
      if (!TryCuda(free_status, "cudaFreeAsync", *error_sink_, target_context)) {
        record.completion_event_.reset();
        PoisonAndEnqueue(std::move(record));
        return;
      }
      record.free_submitted_ = true;
    }

    if (!TryCuda(GetCudaApi().record_event_(record.completion_event_->GetNative(), target_stream->GetNative()),
                 "cudaEventRecord", "allocation retirement completion", *error_sink_, target_context)) {
      record.completion_event_->Discard();
      record.completion_event_.reset();
      PoisonAndEnqueue(std::move(record));
      return;
    }

    Enqueue(std::move(record));
  }

  void Poll() noexcept {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    PollRetirements();
  }

  void SetPeerAccess(Device peer, bool enabled, std::source_location location) {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    ValidateRunning(location);
    if (peer == device_) {
      throw InvalidArgumentError("peer memory-pool access requires two distinct devices", location);
    }

    if (enabled) {
      int can_access = 0;
      CheckCuda(GetCudaApi().can_access_peer_(&can_access, peer.GetOrdinal(), device_.GetOrdinal()),
                "cudaDeviceCanAccessPeer", location);
      if (can_access == 0) {
        std::string message{peer.ToString()};
        message.append(" cannot access allocations owned by ");
        message.append(device_.ToString());
        throw NotSupportedError(std::move(message), location);
      }
      if (can_access != 1) {
        throw InternalError("cudaDeviceCanAccessPeer returned a value other than zero or one", location);
      }
    }

    const cudaMemAccessDesc descriptor{
        .location =
            {
                .type = cudaMemLocationTypeDevice,
                .id = peer.GetOrdinal(),
            },
        .flags = enabled ? cudaMemAccessFlagsProtReadWrite : cudaMemAccessFlagsProtNone,
    };
    DeviceGuard device_guard{device_, *error_sink_, location};
    CheckCuda(GetCudaApi().set_memory_pool_access_(pool_, &descriptor, 1), "cudaMemPoolSetAccess", location);
  }

  void TrimTo(size_t target_reserved_bytes, std::source_location location) {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    ValidateUsable(location);
    PollRetirements();
    DeviceGuard device_guard{device_, *error_sink_, location};
    CheckCuda(GetCudaApi().trim_memory_pool_(pool_, target_reserved_bytes), "cudaMemPoolTrimTo", location);
    trim_count_.fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] auto GetStats(std::source_location location) const -> DeviceAllocatorStats {
    const std::shared_lock lifecycle_lock{lifecycle_latch_};
    uint64_t used_bytes = 0;
    uint64_t reserved_bytes = 0;
    if (pool_ != nullptr) {
      DeviceGuard device_guard{device_, *error_sink_, location};
      CheckCuda(GetCudaApi().get_memory_pool_attribute_(pool_, cudaMemPoolAttrUsedMemCurrent, &used_bytes),
                "cudaMemPoolGetAttribute (used current)", location);
      CheckCuda(GetCudaApi().get_memory_pool_attribute_(pool_, cudaMemPoolAttrReservedMemCurrent, &reserved_bytes),
                "cudaMemPoolGetAttribute (reserved current)", location);
    }

    return DeviceAllocatorStats{
        .logical_live_bytes_ = logical_live_bytes_.load(std::memory_order_relaxed),
        .retiring_bytes_ = retiring_bytes_.load(std::memory_order_relaxed),
        .peak_physical_in_use_bytes_ = peak_physical_in_use_bytes_.load(std::memory_order_relaxed),
        .allocation_count_ = allocation_count_.load(std::memory_order_relaxed),
        .retirement_count_ = retirement_count_.load(std::memory_order_relaxed),
        .retry_count_ = retry_count_.load(std::memory_order_relaxed),
        .oom_count_ = oom_count_.load(std::memory_order_relaxed),
        .trim_count_ = trim_count_.load(std::memory_order_relaxed),
        .pending_retirement_count_ = pending_retirement_count_.load(std::memory_order_relaxed),
        .pool_used_bytes_ = used_bytes,
        .pool_reserved_bytes_ = reserved_bytes,
        .outstanding_storage_count_ = outstanding_storage_count_.load(std::memory_order_relaxed),
    };
  }

  void Shutdown(std::source_location location) {
    const std::scoped_lock shutdown_lock{shutdown_latch_};
    const auto current_status = status_.load(std::memory_order_acquire);
    if (current_status == AllocatorStatus::CLOSED) {
      return;
    }
    status_.store(AllocatorStatus::CLOSING, std::memory_order_release);

    {
      const std::unique_lock lifecycle_lock{lifecycle_latch_};
      const auto outstanding_storage_count = outstanding_storage_count_.load(std::memory_order_acquire);
      if (outstanding_storage_count != 0) {
        throw InvalidArgumentError(FormatOutstandingStorage(outstanding_storage_count), location);
      }
    }

    StopWorker();
    const std::unique_lock lifecycle_lock{lifecycle_latch_};
    DrainRetirements(location);

    DeviceGuard device_guard{device_, *error_sink_, location};
    CheckCuda(GetCudaApi().synchronize_stream_(reclaim_stream_->GetNative()), "cudaStreamSynchronize (reclaim)",
              location);
    CheckCuda(GetCudaApi().trim_memory_pool_(pool_, 0), "cudaMemPoolTrimTo", location);
    CheckCuda(GetCudaApi().destroy_memory_pool_(pool_), "cudaMemPoolDestroy", location);
    pool_ = nullptr;
    reclaim_stream_.reset();
    status_.store(AllocatorStatus::CLOSED, std::memory_order_release);
  }

  [[nodiscard]] auto TryCloseWithoutSynchronization() noexcept -> bool {
    StopWorkerNoexcept();
    if (outstanding_storage_count_.load(std::memory_order_acquire) != 0 ||
        pending_retirement_count_.load(std::memory_order_acquire) != 0) {
      ReportError(ErrorCode::INTERNAL,
                  "device allocator was destroyed without Shutdown while Storage or retirement records remain",
                  *error_sink_, MakeContext(device_, location_));
      return false;
    }
    if (pool_ != nullptr && !TryDestroyPool()) {
      return false;
    }
    status_.store(AllocatorStatus::CLOSED, std::memory_order_release);
    return true;
  }

  [[nodiscard]] auto GetDevice() const noexcept -> Device { return device_; }

 private:
  void PollRetirements() noexcept {
    const auto poll_context = MakeContext(device_, location_);
    CleanupDeviceGuard device_guard{device_, *error_sink_, poll_context, "poll allocation retirements",
                                    "restore after polling allocation retirements"};
    if (!device_guard) {
      return;
    }

    while (true) {
      std::optional<RetirementRecord> completed;
      {
        std::scoped_lock lock{retirement_latch_};
        for (auto iterator = retirements_.begin(); iterator != retirements_.end(); ++iterator) {
          if (iterator->poisoned_ || !iterator->completion_event_.has_value()) {
            continue;
          }

          const auto status = GetCudaApi().query_event_(iterator->completion_event_->GetNative());
          if (status == cudaErrorNotReady) {
            continue;
          }
          if (status != cudaSuccess) {
            const auto context =
                MakeContext(device_, iterator->allocation_.location_, iterator->allocation_stream_->GetId());
            TryCuda(status, "cudaEventQuery", "allocation retirement completion", *error_sink_, context);
            iterator->completion_event_->Discard();
            iterator->completion_event_.reset();
            iterator->poisoned_ = true;
            status_.store(AllocatorStatus::FAILED, std::memory_order_release);
            continue;
          }

          completed.emplace(std::move(*iterator));
          if (iterator != retirements_.end() - 1) {
            *iterator = std::move(retirements_.back());
          }
          retirements_.pop_back();
          pending_retirement_count_.fetch_sub(1, std::memory_order_relaxed);
          break;
        }
      }

      if (!completed.has_value()) {
        return;
      }
      Finalize(std::move(*completed));
    }
  }
  [[nodiscard]] auto MakeUniqueStorage(Allocation allocation, const std::shared_ptr<DeviceAllocator> &allocator,
                                       const std::shared_ptr<StreamState> &allocation_stream)
      -> std::unique_ptr<Storage> {
    return std::unique_ptr<Storage>{new Storage(std::move(allocation), device_, allocator, allocation_stream)};
  }

  void ValidateRunning(std::source_location location) const {
    if (status_.load(std::memory_order_acquire) != AllocatorStatus::RUNNING) {
      throw InvalidArgumentError("device allocator is not accepting new allocations", location);
    }
  }

  void ValidateUsable(std::source_location location) const {
    if (status_.load(std::memory_order_acquire) == AllocatorStatus::CLOSED || pool_ == nullptr) {
      throw InvalidArgumentError("device allocator is closed", location);
    }
  }

  void ValidateExternal(void *pointer, size_t capacity_bytes, ExternalOwnership ownership,
                        const std::shared_ptr<void> &owner, std::source_location location) {
    if (capacity_bytes == 0) {
      if (pointer != nullptr) {
        throw InvalidArgumentError("zero-byte external memory must use a null pointer", location);
      }
    } else if (pointer == nullptr) {
      throw InvalidArgumentError("non-empty external memory requires a non-null pointer", location);
    }

    if (ownership == ExternalOwnership::BORROWED && owner != nullptr) {
      throw InvalidArgumentError("borrowed external memory must not provide an owner", location);
    }
    if (ownership == ExternalOwnership::SHARED_OWNER && owner == nullptr) {
      throw InvalidArgumentError("owned external memory requires a shared owner", location);
    }
    if (capacity_bytes == 0) {
      return;
    }

    cudaPointerAttributes attributes{};
    DeviceGuard device_guard{device_, *error_sink_, location};
    CheckCuda(GetCudaApi().get_pointer_attributes_(&attributes, pointer), "cudaPointerGetAttributes", location);
    if (attributes.type != cudaMemoryTypeDevice) {
      throw InvalidArgumentError("external memory must point to CUDA device memory", location);
    }
    if (attributes.device != device_.GetOrdinal()) {
      throw InvalidArgumentError("external memory belongs to a different CUDA device", location);
    }
  }

  void ReserveRetirementSlot(std::source_location location) {
    std::scoped_lock lock{retirement_latch_};
    const auto required =
        CheckedAdd(retirements_.size(),
                   CheckedAdd(reserved_retirement_slots_, size_t{1}, "retirement reservation count", location),
                   "retirement queue capacity", location);
    if (retirements_.capacity() < required) {
      auto new_capacity = std::max(required, INITIAL_RETIREMENT_CAPACITY);
      if (retirements_.capacity() != 0) {
        const auto doubled = CheckedMultiply(retirements_.capacity(), size_t{2}, "retirement queue growth", location);
        new_capacity = std::max(new_capacity, doubled);
      }
      retirements_.reserve(new_capacity);
    }
    reserved_retirement_slots_++;
  }

  void CancelRetirementSlot() noexcept {
    std::scoped_lock lock{retirement_latch_};
    if (reserved_retirement_slots_ == 0) {
      std::terminate();
    }
    reserved_retirement_slots_--;
  }

  void Enqueue(RetirementRecord record) noexcept {
    {
      std::scoped_lock lock{retirement_latch_};
      if (reserved_retirement_slots_ == 0 || retirements_.size() == retirements_.capacity()) {
        std::terminate();
      }
      reserved_retirement_slots_--;
      retirements_.push_back(std::move(record));
      pending_retirement_count_.fetch_add(1, std::memory_order_relaxed);
      retirement_count_.fetch_add(1, std::memory_order_relaxed);
    }
    maintenance_condition_.notify_one();
  }

  void PoisonAndEnqueue(RetirementRecord record) noexcept {
    record.poisoned_ = true;
    status_.store(AllocatorStatus::FAILED, std::memory_order_release);
    Enqueue(std::move(record));
  }

  void ReserveBudget(uint64_t bytes, const AllocationContext &context) {
    auto current = physical_in_use_bytes_.load(std::memory_order_relaxed);
    while (true) {
      const auto maximum = options_.max_live_bytes_;
      if (bytes > std::numeric_limits<uint64_t>::max() - current) {
        throw OverflowError("device allocator physical byte count overflow", context.location_);
      }
      if (maximum != 0 && (current > maximum || bytes > maximum - current)) {
        oom_count_.fetch_add(1, std::memory_order_relaxed);
        std::string message{"device allocation of "};
        message.append(std::to_string(bytes));
        message.append(" bytes for ");
        message.append(context.operation_);
        message.append(" exceeds the configured ");
        message.append(std::to_string(maximum));
        message.append("-byte live-memory budget");
        throw OutOfMemoryError(std::move(message), context.location_);
      }
      if (physical_in_use_bytes_.compare_exchange_weak(current, current + bytes, std::memory_order_acq_rel,
                                                       std::memory_order_relaxed)) {
        UpdatePeak(current + bytes);
        return;
      }
    }
  }

  void ReleaseBudget(uint64_t bytes) noexcept { physical_in_use_bytes_.fetch_sub(bytes, std::memory_order_relaxed); }

  void UpdatePeak(uint64_t value) noexcept {
    auto peak = peak_physical_in_use_bytes_.load(std::memory_order_relaxed);
    while (value > peak && !peak_physical_in_use_bytes_.compare_exchange_weak(peak, value, std::memory_order_relaxed,
                                                                              std::memory_order_relaxed)) {
    }
  }

  [[nodiscard]] auto GetOomTrimTarget(uint64_t requested_bytes) const noexcept -> uint64_t {
    const auto physical = physical_in_use_bytes_.load(std::memory_order_relaxed);
    const auto live_before_request = physical >= requested_bytes ? physical - requested_bytes : 0;
    if (options_.max_reserved_bytes_ == 0) {
      return live_before_request;
    }
    return std::max(live_before_request, options_.max_reserved_bytes_);
  }

  [[nodiscard]] auto FormatOutOfMemory(size_t bytes, size_t alignment, const AllocationContext &context)
      -> std::string {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    uint64_t used_bytes = 0;
    uint64_t reserved_bytes = 0;
    const auto report_context = MakeContext(device_, context.location_);
    TryCuda(GetCudaApi().get_memory_info_(&free_bytes, &total_bytes), "cudaMemGetInfo", *error_sink_, report_context);
    TryCuda(GetCudaApi().get_memory_pool_attribute_(pool_, cudaMemPoolAttrUsedMemCurrent, &used_bytes),
            "cudaMemPoolGetAttribute", "used current", *error_sink_, report_context);
    TryCuda(GetCudaApi().get_memory_pool_attribute_(pool_, cudaMemPoolAttrReservedMemCurrent, &reserved_bytes),
            "cudaMemPoolGetAttribute", "reserved current", *error_sink_, report_context);

    std::string message{"CUDA device allocation failed after one retry: requested="};
    message.append(std::to_string(bytes));
    message.append(", alignment=");
    message.append(std::to_string(alignment));
    message.append(", device=");
    message.append(device_.ToString());
    message.append(", operation=");
    message.append(context.operation_);
    if (context.output_shape_.has_value()) {
      message.append(", shape=");
      message.append(context.output_shape_->ToString());
    }
    if (context.dtype_.has_value()) {
      message.append(", dtype=");
      message.append(GetDTypeName(*context.dtype_, context.location_));
    }
    message.append(", logical_live=");
    message.append(std::to_string(logical_live_bytes_.load(std::memory_order_relaxed)));
    message.append(", retiring=");
    message.append(std::to_string(retiring_bytes_.load(std::memory_order_relaxed)));
    message.append(", physical_in_use=");
    message.append(std::to_string(physical_in_use_bytes_.load(std::memory_order_relaxed)));
    message.append(", peak_physical_in_use=");
    message.append(std::to_string(peak_physical_in_use_bytes_.load(std::memory_order_relaxed)));
    message.append(", allocations=");
    message.append(std::to_string(allocation_count_.load(std::memory_order_relaxed)));
    message.append(", retirements=");
    message.append(std::to_string(retirement_count_.load(std::memory_order_relaxed)));
    message.append(", retries=");
    message.append(std::to_string(retry_count_.load(std::memory_order_relaxed)));
    message.append(", ooms=");
    message.append(std::to_string(oom_count_.load(std::memory_order_relaxed)));
    message.append(", trims=");
    message.append(std::to_string(trim_count_.load(std::memory_order_relaxed)));
    message.append(", pending_retirements=");
    message.append(std::to_string(pending_retirement_count_.load(std::memory_order_relaxed)));
    message.append(", pool_used=");
    message.append(std::to_string(used_bytes));
    message.append(", pool_reserved=");
    message.append(std::to_string(reserved_bytes));
    message.append(", release_threshold=");
    message.append(std::to_string(options_.release_threshold_bytes_));
    message.append(", max_live=");
    message.append(std::to_string(options_.max_live_bytes_));
    message.append(", max_reserved=");
    message.append(std::to_string(options_.max_reserved_bytes_));
    message.append(", cuda_free=");
    message.append(std::to_string(free_bytes));
    message.append(", cuda_total=");
    message.append(std::to_string(total_bytes));
    return message;
  }

  [[nodiscard]] auto SubmitDependency(const std::shared_ptr<StreamState> &stream,
                                      std::source_location location) noexcept -> bool {
    cudaEvent_t event = nullptr;
    const auto source_context = MakeContext(stream->GetDevice(), location, stream->GetId());
    {
      CleanupDeviceGuard source_guard{stream->GetDevice(), *error_sink_, source_context,
                                      "create allocation retirement dependency",
                                      "restore after allocation retirement dependency creation"};
      if (!source_guard ||
          !TryCuda(GetCudaApi().create_event_with_flags_(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags",
                   "allocation retirement dependency", *error_sink_, source_context)) {
        return false;
      }
      if (event == nullptr) {
        ReportError(ErrorCode::INTERNAL, "cudaEventCreateWithFlags returned a null dependency event", *error_sink_,
                    source_context);
        return false;
      }
      if (!TryCuda(GetCudaApi().record_event_(event, stream->GetNative()), "cudaEventRecord",
                   "allocation retirement dependency", *error_sink_, source_context)) {
        TryCuda(GetCudaApi().destroy_event_(event), "cudaEventDestroy", "allocation retirement dependency",
                *error_sink_, source_context);
        return false;
      }
    }

    const auto reclaim_context = MakeContext(device_, location, reclaim_stream_->GetId());
    bool waited = false;
    {
      CleanupDeviceGuard reclaim_guard{device_, *error_sink_, reclaim_context, "wait for allocation usage",
                                       "restore after allocation usage wait"};
      if (reclaim_guard) {
        waited = TryCuda(GetCudaApi().stream_wait_event_(reclaim_stream_->GetNative(), event, cudaEventWaitDefault),
                         "cudaStreamWaitEvent", "allocation retirement dependency", *error_sink_, reclaim_context);
      }
    }

    {
      CleanupDeviceGuard source_guard{stream->GetDevice(), *error_sink_, source_context,
                                      "destroy allocation retirement dependency",
                                      "restore after allocation retirement dependency destruction"};
      if (!source_guard || !TryCuda(GetCudaApi().destroy_event_(event), "cudaEventDestroy",
                                    "allocation retirement dependency", *error_sink_, source_context)) {
        return false;
      }
    }
    return waited;
  }

  void Finalize(RetirementRecord record) noexcept {
    if (record.allocation_.kind_ == AllocationKind::DEVICE_POOL) {
      const auto bytes = static_cast<uint64_t>(record.allocation_.capacity_bytes_);
      retiring_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
      ReleaseBudget(bytes);
    }
  }

  void DrainRetirements(std::source_location location) {
    while (true) {
      std::optional<RetirementRecord> record;
      {
        std::unique_lock lock{retirement_latch_};
        if (retirements_.empty()) {
          return;
        }
        SynchronizeRetirement(retirements_.back(), location);
        record.emplace(std::move(retirements_.back()));
        retirements_.pop_back();
        pending_retirement_count_.fetch_sub(1, std::memory_order_relaxed);
      }

      Finalize(std::move(*record));
    }
  }

  void SynchronizeRetirement(RetirementRecord &record, std::source_location location) {
    if (!record.poisoned_ && record.completion_event_.has_value()) {
      DeviceGuard device_guard{device_, *error_sink_, location};
      CheckCuda(GetCudaApi().synchronize_event_(record.completion_event_->GetNative()), "cudaEventSynchronize",
                location);
      return;
    }

    SynchronizeUsageStream(record.allocation_stream_, location);
    for (const auto &stream : record.side_streams_) {
      SynchronizeUsageStream(stream, location);
    }

    if (record.allocation_.kind_ != AllocationKind::DEVICE_POOL) {
      return;
    }

    if (record.free_attempted_ && !record.free_submitted_) {
      throw InternalError("cannot retry cudaFreeAsync after a failed host call because submission status is ambiguous",
                          location);
    }

    DeviceGuard device_guard{device_, *error_sink_, location};
    if (!record.free_submitted_) {
      CheckCuda(GetCudaApi().free_async_(record.allocation_.pointer_, reclaim_stream_->GetNative()), "cudaFreeAsync",
                location);
      record.free_submitted_ = true;
      record.uses_reclaim_stream_ = true;
    }
    if (record.uses_reclaim_stream_) {
      CheckCuda(GetCudaApi().synchronize_stream_(reclaim_stream_->GetNative()), "cudaStreamSynchronize (reclaim)",
                location);
    }
  }

  void SynchronizeUsageStream(const std::shared_ptr<StreamState> &stream, std::source_location location) {
    DeviceGuard device_guard{stream->GetDevice(), *error_sink_, location};
    CheckCuda(GetCudaApi().synchronize_stream_(stream->GetNative()), "cudaStreamSynchronize (allocation usage)",
              location);
  }

  void MaintenanceLoop(const std::stop_token &stop_token) noexcept {
    std::unique_lock lock{maintenance_latch_};
    while (!stop_token.stop_requested()) {
      if (pending_retirement_count_.load(std::memory_order_relaxed) == 0) {
        maintenance_condition_.wait(lock, stop_token,
                                    [this] { return pending_retirement_count_.load(std::memory_order_relaxed) != 0; });
      } else {
        maintenance_condition_.wait_for(lock, stop_token, MAINTENANCE_INTERVAL, [] { return false; });
      }
      if (stop_token.stop_requested()) {
        return;
      }
      if (status_.load(std::memory_order_acquire) == AllocatorStatus::FAILED) {
        return;
      }
      lock.unlock();
      Poll();
      lock.lock();
    }
  }

  void StopWorker() {
    if (!maintenance_worker_.joinable()) {
      return;
    }
    maintenance_worker_.request_stop();
    maintenance_condition_.notify_all();
    maintenance_worker_.join();
  }

  void StopWorkerNoexcept() noexcept {
    try {
      StopWorker();
    } catch (const std::exception &error) {
      ReportError(ErrorCode::INTERNAL, error.what(), *error_sink_, MakeContext(device_, location_));
    } catch (...) {
      ReportError(ErrorCode::INTERNAL, "unknown failure while stopping allocator maintenance worker", *error_sink_,
                  MakeContext(device_, location_));
    }
  }

  auto TryDestroyPool() noexcept -> bool {
    const auto context = MakeContext(device_, location_);
    CleanupDeviceGuard device_guard{device_, *error_sink_, context, "destroy device memory pool",
                                    "restore after device memory pool destruction"};
    if (!device_guard) {
      return false;
    }
    if (!TryCuda(GetCudaApi().destroy_memory_pool_(pool_), "cudaMemPoolDestroy", *error_sink_, context)) {
      return false;
    }
    pool_ = nullptr;
    return true;
  }

  Device device_;
  std::shared_ptr<ErrorSink> error_sink_;
  std::shared_ptr<EventPool> event_pool_;
  DeviceAllocatorOptions options_;
  cudaMemPool_t pool_;
  std::shared_ptr<StreamState> reclaim_stream_;
  std::source_location location_;

  mutable std::shared_mutex lifecycle_latch_;
  std::mutex shutdown_latch_;
  std::atomic<AllocatorStatus> status_{AllocatorStatus::RUNNING};
  std::atomic<uint64_t> logical_live_bytes_{0};
  std::atomic<uint64_t> retiring_bytes_{0};
  std::atomic<uint64_t> physical_in_use_bytes_{0};
  std::atomic<uint64_t> peak_physical_in_use_bytes_{0};
  std::atomic<uint64_t> allocation_count_{0};
  std::atomic<uint64_t> retirement_count_{0};
  std::atomic<uint64_t> retry_count_{0};
  std::atomic<uint64_t> oom_count_{0};
  std::atomic<uint64_t> trim_count_{0};
  std::atomic<uint64_t> pending_retirement_count_{0};
  std::atomic<uint64_t> outstanding_storage_count_{0};

  mutable std::mutex retirement_latch_;
  std::vector<RetirementRecord> retirements_;
  size_t reserved_retirement_slots_{0};

  std::mutex maintenance_latch_;
  std::condition_variable_any maintenance_condition_;
  std::jthread maintenance_worker_;
};

auto DeviceAllocator::Create(Device device, const std::shared_ptr<ErrorSink> &error_sink,
                             const std::shared_ptr<EventPool> &event_pool, DeviceAllocatorOptions options,
                             std::source_location location) -> std::shared_ptr<DeviceAllocator> {
  ValidateErrorSink(error_sink, location);
  ValidateEventPool(device, event_pool, location);

  const auto reclaim_stream = StreamAccess::CreateOwned(device, 0, error_sink, location);
  const auto reclaim_state = StreamAccess::GetState(reclaim_stream);

  cudaMemPool_t pool = nullptr;
  DeviceGuard device_guard{device, *error_sink, location};
  cudaMemPoolProps properties{};
  properties.allocType = cudaMemAllocationTypePinned;
  properties.handleTypes = cudaMemHandleTypeNone;
  properties.location.type = cudaMemLocationTypeDevice;
  properties.location.id = device.GetOrdinal();
  CheckCuda(GetCudaApi().create_memory_pool_(&pool, &properties), "cudaMemPoolCreate", location);
  if (pool == nullptr) {
    throw InternalError("cudaMemPoolCreate returned a null pool", location);
  }

  try {
    auto threshold = options.release_threshold_bytes_;
    int enabled = 1;
    CheckCuda(GetCudaApi().set_memory_pool_attribute_(pool, cudaMemPoolAttrReleaseThreshold, &threshold),
              "cudaMemPoolSetAttribute (release threshold)", location);
    CheckCuda(GetCudaApi().set_memory_pool_attribute_(pool, cudaMemPoolReuseFollowEventDependencies, &enabled),
              "cudaMemPoolSetAttribute (follow event dependencies)", location);
    CheckCuda(GetCudaApi().set_memory_pool_attribute_(pool, cudaMemPoolReuseAllowOpportunistic, &enabled),
              "cudaMemPoolSetAttribute (allow opportunistic reuse)", location);
    CheckCuda(GetCudaApi().set_memory_pool_attribute_(pool, cudaMemPoolReuseAllowInternalDependencies, &enabled),
              "cudaMemPoolSetAttribute (allow internal dependencies)", location);

    auto impl =
        std::make_unique<DeviceAllocatorImpl>(device, error_sink, event_pool, options, pool, reclaim_state, location);
    pool = nullptr;
    auto allocator = std::shared_ptr<DeviceAllocator>{new DeviceAllocator(std::move(impl))};
    allocator->impl_->StartWorker();
    return allocator;
  } catch (...) {
    const auto context = MakeContext(device, location);
    if (pool != nullptr) {
      TryCuda(GetCudaApi().destroy_memory_pool_(pool), "cudaMemPoolDestroy", "allocator construction rollback",
              *error_sink, context);
    }
    throw;
  }
}

DeviceAllocator::DeviceAllocator(std::unique_ptr<DeviceAllocatorImpl> impl) noexcept : impl_(std::move(impl)) {}

DeviceAllocator::~DeviceAllocator() noexcept {
  if (impl_ != nullptr && !impl_->TryCloseWithoutSynchronization()) {
    [[maybe_unused]] auto *quarantined_impl = impl_.release();
  }
}

auto DeviceAllocator::Allocate(const Stream &stream, size_t bytes, size_t alignment, const AllocationContext &context)
    -> std::shared_ptr<Storage> {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CheckCuda(GetCudaApi().is_stream_capturing_(StreamAccess::GetNative(stream), &capture_status),
            "cudaStreamIsCapturing (device allocation)", context.location_);
  if (capture_status != cudaStreamCaptureStatusNone) {
    throw CaptureError("device allocation is forbidden during CUDA graph capture", context.location_);
  }
  return impl_->Allocate(shared_from_this(), stream, bytes, alignment, context);
}

auto DeviceAllocator::WrapExternal(const Stream &allocation_stream, void *pointer, size_t capacity_bytes,
                                   ExternalOwnership ownership, std::shared_ptr<void> owner,
                                   std::source_location location) -> std::shared_ptr<Storage> {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  CheckCuda(GetCudaApi().is_stream_capturing_(StreamAccess::GetNative(allocation_stream), &capture_status),
            "cudaStreamIsCapturing (external memory wrapping)", location);
  if (capture_status != cudaStreamCaptureStatusNone) {
    throw CaptureError("external memory wrapping is forbidden during CUDA graph capture", location);
  }
  return impl_->WrapExternal(shared_from_this(), allocation_stream, pointer, capacity_bytes, ownership,
                             std::move(owner), location);
}

void DeviceAllocator::SetPeerAccess(Device peer, bool enabled, std::source_location location) {
  impl_->SetPeerAccess(peer, enabled, location);
}

void DeviceAllocator::Poll() noexcept { impl_->Poll(); }

void DeviceAllocator::TrimTo(size_t target_reserved_bytes, std::source_location location) {
  impl_->TrimTo(target_reserved_bytes, location);
}

auto DeviceAllocator::GetStats(std::source_location location) const -> DeviceAllocatorStats {
  return impl_->GetStats(location);
}

void DeviceAllocator::Shutdown(std::source_location location) { impl_->Shutdown(location); }

auto DeviceAllocator::GetDevice() const noexcept -> Device { return impl_->GetDevice(); }

void DeviceAllocator::Retire(Allocation allocation, std::shared_ptr<StreamState> allocation_stream,
                             std::vector<std::shared_ptr<StreamState>> side_streams) noexcept {
  impl_->Retire(std::move(allocation), std::move(allocation_stream), std::move(side_streams));
}

}  // namespace ttl::internal
