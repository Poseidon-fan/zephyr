#pragma once

#include <driver_types.h>
#include <cstdint>
#include <memory>
#include <source_location>

#include "ttl/runtime/device_error.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl {

class ErrorSink;

}  // namespace ttl

namespace ttl::internal {

class DeviceAllocator;
class PinnedAllocator;
class Storage;

using DeviceErrorCode = CudaDeviceErrorCode;
using DeviceErrorRecord = CudaDeviceErrorRecord;
using DeviceErrorLaunchContext = CudaDeviceErrorContext;

/**
 * @brief ExecutionContext-owned sticky device error record and pinned host mirror.
 *
 * Register assigns stream-ordered operation sequence numbers. EnqueueRead snapshots the device record after prior work;
 * ConsumeAndReset clears it before translating the first reported semantic failure into a host exception.
 */
class DeviceErrorState final {
 public:
  [[nodiscard]] static auto Create(const std::shared_ptr<DeviceAllocator> &allocator, const Stream &stream,
                                   const std::shared_ptr<PinnedAllocator> &pinned_allocator,
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
  void EnqueueRead(const Stream &stream, std::source_location location);
  void ConsumeAndReset(cudaStream_t stream, std::source_location location);
  [[nodiscard]] auto GetStorage() const noexcept -> const std::shared_ptr<Storage> &;

 private:
  DeviceErrorState(std::shared_ptr<Storage> storage, PinnedBuffer host_record) noexcept;

  std::shared_ptr<Storage> storage_;
  PinnedBuffer host_record_;
  uint64_t next_operation_sequence_{1};
};

}  // namespace ttl::internal
