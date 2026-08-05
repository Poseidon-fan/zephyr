#include "ttl/internal/runtime/execution/device_error.hpp"

#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <utility>

#include <driver_types.h>

#include "ttl/common/device.hpp"
#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/device_guard.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/memory/device_allocator.hpp"
#include "ttl/internal/tensor/storage.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl::internal {
namespace {

[[nodiscard]] auto DecodeOffendingValue(DType dtype, uint64_t bits) -> std::string {
  switch (dtype) {
    case DType::BOOL:
      return bits == 0 ? "false" : "true";
    case DType::UINT8:
      return std::to_string(static_cast<uint8_t>(bits));
    case DType::INT32:
      return std::to_string(std::bit_cast<int32_t>(static_cast<uint32_t>(bits)));
    case DType::INT64:
      return std::to_string(std::bit_cast<int64_t>(bits));
    case DType::FLOAT16:
      return std::to_string(Float16ToFloat(Float16{.bits_ = static_cast<uint16_t>(bits)}));
    case DType::BFLOAT16:
      return std::to_string(BFloat16ToFloat(BFloat16{.bits_ = static_cast<uint16_t>(bits)}));
    case DType::FLOAT32:
      return std::to_string(std::bit_cast<float>(static_cast<uint32_t>(bits)));
  }
  return "<invalid dtype>";
}

[[nodiscard]] auto FormatCastError(const DeviceErrorRecord &record, DType source_dtype, DType target_dtype)
    -> std::string {
  std::string message{"CastOut operation "};
  message.append(std::to_string(record.operation_sequence_));
  message.append(" cannot convert ");
  message.append(DecodeOffendingValue(source_dtype, record.offending_value_bits_));
  message.append(" from ");
  message.append(GetDTypeInfo(source_dtype).name_);
  message.append(" to ");
  message.append(GetDTypeInfo(target_dtype).name_);
  message.append(" at iterator index ");
  message.append(std::to_string(record.linear_index_));
  return message;
}

[[nodiscard]] auto FormatIntegerDivisionError(const DeviceErrorRecord &record, DType dtype) -> std::string {
  std::string message{"DivideOut operation "};
  message.append(std::to_string(record.operation_sequence_));
  message.append(" encountered integer division by zero for ");
  message.append(GetDTypeInfo(dtype).name_);
  message.append(" at iterator index ");
  message.append(std::to_string(record.linear_index_));
  return message;
}

[[nodiscard]] auto FormatIndexError(const DeviceErrorRecord &record, DType dtype) -> std::string {
  std::string message{"Indexing operation "};
  message.append(std::to_string(record.operation_sequence_));
  message.append(" encountered index ");
  message.append(DecodeOffendingValue(dtype, record.offending_value_bits_));
  message.append(" outside [0, ");
  message.append(std::to_string(record.bound_));
  message.append(") at output index ");
  message.append(std::to_string(record.linear_index_));
  return message;
}

[[nodiscard]] auto FormatRandomCounterError(const DeviceErrorRecord &record) -> std::string {
  std::string message{"random operation "};
  message.append(std::to_string(record.operation_sequence_));
  message.append(" exhausted its Philox counter space at counter ");
  message.append(std::to_string(record.offending_value_bits_));
  return message;
}

[[nodiscard]] auto FormatExternalError(const DeviceErrorRecord &record) -> std::string {
  std::string message{"external CUDA operation "};
  message.append(std::to_string(record.operation_sequence_));
  message.append(" reported device error code ");
  message.append(std::to_string(record.code_));
  message.append(" at linear index ");
  message.append(std::to_string(record.linear_index_));
  message.append(" with value bits ");
  message.append(std::to_string(record.offending_value_bits_));
  message.append(" and bound ");
  message.append(std::to_string(record.bound_));
  return message;
}

}  // namespace

// The host mirror is page-locked so an ordered D2H copy can snapshot the device record without synchronizing the
// device.
class PinnedDeviceErrorRecord final {
 public:
  PinnedDeviceErrorRecord(Device device, std::shared_ptr<ErrorSink> error_sink, std::source_location location)
      : device_(device), error_sink_(std::move(error_sink)), location_(location) {
    void *pointer = nullptr;
    CheckCuda(GetCudaApi().host_alloc_(&pointer, sizeof(DeviceErrorRecord), cudaHostAllocPortable),
              "cudaHostAlloc (device error mirror)", location_);
    if (pointer == nullptr) {
      throw InternalError("cudaHostAlloc returned a null device error mirror", location_);
    }
    record_ = static_cast<DeviceErrorRecord *>(pointer);
  }

  PinnedDeviceErrorRecord(const PinnedDeviceErrorRecord &) = delete;
  auto operator=(const PinnedDeviceErrorRecord &) -> PinnedDeviceErrorRecord & = delete;

  ~PinnedDeviceErrorRecord() noexcept {
    if (record_ == nullptr) {
      return;
    }
    TryCuda(GetCudaApi().free_host_(record_), "cudaFreeHost (device error mirror)", *error_sink_,
            ErrorReportContext{
                .location_ = location_,
                .device_ = device_,
                .stream_id_ = std::nullopt,
            });
  }

  [[nodiscard]] auto Get() noexcept -> DeviceErrorRecord * { return record_; }
  [[nodiscard]] auto Get() const noexcept -> const DeviceErrorRecord * { return record_; }

 private:
  Device device_;
  std::shared_ptr<ErrorSink> error_sink_;
  std::source_location location_;
  DeviceErrorRecord *record_{nullptr};
};

auto DeviceErrorState::Create(const std::shared_ptr<DeviceAllocator> &allocator, const Stream &stream,
                              std::shared_ptr<ErrorSink> error_sink, std::source_location location)
    -> std::unique_ptr<DeviceErrorState> {
  if (allocator == nullptr || error_sink == nullptr) {
    throw InvalidArgumentError("device error state requires an allocator and error sink", location);
  }
  if (allocator->GetDevice() != stream.GetDevice()) {
    throw InvalidArgumentError("device error state stream and allocator devices must match", location);
  }

  DeviceGuard device_guard{allocator->GetDevice(), *error_sink, location};
  auto host_record = std::make_unique<PinnedDeviceErrorRecord>(allocator->GetDevice(), std::move(error_sink), location);
  auto storage = allocator->Allocate(stream, sizeof(DeviceErrorRecord), alignof(DeviceErrorRecord),
                                     AllocationContext{
                                         .operation_ = "ExecutionContext device error record",
                                         .output_shape_ = std::nullopt,
                                         .dtype_ = std::nullopt,
                                         .location_ = location,
                                     });
  CheckCuda(GetCudaApi().memset_async_(storage->GetBasePointer(), 0, sizeof(DeviceErrorRecord),
                                       StreamAccess::GetNative(stream)),
            "cudaMemsetAsync (initialize device error record)", location);
  return std::unique_ptr<DeviceErrorState>{new DeviceErrorState{std::move(storage), std::move(host_record)}};
}

DeviceErrorState::DeviceErrorState(std::shared_ptr<Storage> storage,
                                   std::unique_ptr<PinnedDeviceErrorRecord> host_record) noexcept
    : storage_(std::move(storage)), host_record_(std::move(host_record)) {}

DeviceErrorState::~DeviceErrorState() noexcept = default;

auto DeviceErrorState::Register(const Stream &stream, DType source_dtype, DType target_dtype,
                                std::source_location location) -> DeviceErrorLaunchContext {
  if (stream.GetDevice() != storage_->GetDevice()) {
    throw InternalError("device error registration used a stream on the wrong device", location);
  }
  if (next_operation_sequence_ == std::numeric_limits<uint64_t>::max()) {
    throw OverflowError("device error operation sequence exhausted", location);
  }

  // Every registered kernel receives the same sticky record and a monotonically increasing sequence. Device-side CAS
  // preserves the first failure until the context reaches its explicit check boundary.
  storage_->RecordUsage(stream);
  const auto sequence = next_operation_sequence_;
  next_operation_sequence_++;
  return DeviceErrorLaunchContext{
      .record_ = static_cast<DeviceErrorRecord *>(storage_->GetBasePointer()),
      .operation_sequence_ = sequence,
      .source_dtype_ = source_dtype,
      .target_dtype_ = target_dtype,
  };
}

void DeviceErrorState::EnqueueRead(cudaStream_t stream, std::source_location location) {
  if (stream == nullptr) {
    throw InternalError("device error read requires a non-null stream", location);
  }
  // The copy is enqueued on the execution stream after submitted kernels, so stream ordering makes every record field
  // visible in the pinned mirror when the surrounding check synchronizes its completion event.
  CheckCuda(GetCudaApi().memcpy_async_(host_record_->Get(), storage_->GetBasePointer(), sizeof(DeviceErrorRecord),
                                       cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync (read device error record)", location);
}

void DeviceErrorState::ConsumeAndReset(cudaStream_t stream, std::source_location location) {
  const auto record = *host_record_->Get();
  // Reset before translating the snapshot. Even when translation throws, later submissions start with an empty record.
  CheckCuda(GetCudaApi().memset_async_(storage_->GetBasePointer(), 0, sizeof(DeviceErrorRecord), stream),
            "cudaMemsetAsync (reset device error record)", location);
  if (record.code_ == static_cast<uint32_t>(DeviceErrorCode::NONE)) {
    return;
  }

  const auto source_dtype = ParseDType(record.source_dtype_);
  const auto target_dtype = ParseDType(record.target_dtype_);
  if (!source_dtype.has_value() || !target_dtype.has_value()) {
    throw InternalError("device error record contains an invalid dtype", location);
  }

  switch (static_cast<DeviceErrorCode>(record.code_)) {
    case DeviceErrorCode::CAST_OUT_OF_RANGE:
      throw DeviceError(FormatCastError(record, *source_dtype, *target_dtype), location);
    case DeviceErrorCode::INTEGER_DIVIDE_BY_ZERO:
      throw DeviceError(FormatIntegerDivisionError(record, *source_dtype), location);
    case DeviceErrorCode::INDEX_OUT_OF_BOUNDS:
      throw DeviceError(FormatIndexError(record, *source_dtype), location);
    case DeviceErrorCode::RNG_COUNTER_OVERFLOW:
      throw DeviceError(FormatRandomCounterError(record), location);
    case DeviceErrorCode::INVALID_VALUE:
    case DeviceErrorCode::USER_DEFINED:
      throw DeviceError(FormatExternalError(record), location);
    case DeviceErrorCode::NONE:
      break;
  }
  if (record.code_ > static_cast<uint32_t>(DeviceErrorCode::USER_DEFINED)) {
    throw DeviceError(FormatExternalError(record), location);
  }
  throw InternalError("device error record contains an invalid error code", location);
}

auto DeviceErrorState::GetStorage() const noexcept -> const std::shared_ptr<Storage> & { return storage_; }

}  // namespace ttl::internal
