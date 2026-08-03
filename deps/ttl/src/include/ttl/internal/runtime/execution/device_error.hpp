#pragma once

#include <cstdint>
#include <memory>
#include <source_location>
#include <type_traits>

#include <driver_types.h>

#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl {

class ErrorSink;

}  // namespace ttl

namespace ttl::internal {

class DeviceAllocator;
class PinnedDeviceErrorRecord;
class Storage;

enum class DeviceErrorCode : uint8_t {
  NONE = 0,
  INDEX_OUT_OF_BOUNDS,
  INTEGER_DIVIDE_BY_ZERO,
  CAST_OUT_OF_RANGE,
  RNG_COUNTER_OVERFLOW,
};

/** Sticky first-error record shared by kernels submitted through one ExecutionContext. */
struct DeviceErrorRecord final {
  uint32_t code_{0};
  uint8_t source_dtype_{0};
  uint8_t target_dtype_{0};
  uint16_t reserved_{0};
  uint64_t operation_sequence_{0};
  int64_t linear_index_{0};
  uint64_t offending_value_bits_{0};
  int64_t bound_{0};
};

/** Per-launch device error destination and monotonically increasing operation identity. */
struct DeviceErrorLaunchContext final {
  DeviceErrorRecord *record_{nullptr};
  uint64_t operation_sequence_{0};
  DType source_dtype_{DType::BOOL};
  DType target_dtype_{DType::BOOL};
};

static_assert(std::is_trivially_copyable_v<DeviceErrorRecord>);
static_assert(std::is_standard_layout_v<DeviceErrorRecord>);
static_assert(std::is_trivially_copyable_v<DeviceErrorLaunchContext>);
static_assert(std::is_standard_layout_v<DeviceErrorLaunchContext>);

/** ExecutionContext-owned device record and pinned host mirror. */
class DeviceErrorState final {
 public:
  [[nodiscard]] static auto Create(const std::shared_ptr<DeviceAllocator> &allocator, const Stream &stream,
                                   std::shared_ptr<ErrorSink> error_sink,
                                   std::source_location location = std::source_location::current())
      -> std::unique_ptr<DeviceErrorState>;

  DeviceErrorState(const DeviceErrorState &) = delete;
  auto operator=(const DeviceErrorState &) -> DeviceErrorState & = delete;
  DeviceErrorState(DeviceErrorState &&) = delete;
  auto operator=(DeviceErrorState &&) -> DeviceErrorState & = delete;

  ~DeviceErrorState() noexcept;

  [[nodiscard]] auto Register(const Stream &stream, DType source_dtype, DType target_dtype,
                              std::source_location location) -> DeviceErrorLaunchContext;
  void EnqueueRead(cudaStream_t stream, std::source_location location);
  void ConsumeAndReset(cudaStream_t stream, std::source_location location);
  [[nodiscard]] auto GetStorage() const noexcept -> const std::shared_ptr<Storage> &;

 private:
  DeviceErrorState(std::shared_ptr<Storage> storage, std::unique_ptr<PinnedDeviceErrorRecord> host_record) noexcept;

  std::shared_ptr<Storage> storage_;
  std::unique_ptr<PinnedDeviceErrorRecord> host_record_;
  uint64_t next_operation_sequence_{1};
};

}  // namespace ttl::internal
