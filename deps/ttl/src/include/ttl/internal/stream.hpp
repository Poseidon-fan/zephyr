#pragma once

#include <cstdint>
#include <memory>
#include <source_location>

#include <cuda_runtime_api.h>

#include "ttl/device.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {

/** Shared state and native-resource owner behind a public Stream handle. */
class StreamState final {
 public:
  StreamState(Device device, int32_t priority, std::shared_ptr<ErrorSink> error_sink, const CudaApi &cuda_api,
              std::source_location location);
  StreamState(Device device, cudaStream_t stream, std::shared_ptr<void> external_owner,
              std::shared_ptr<ErrorSink> error_sink, const CudaApi &cuda_api, std::source_location location);

  StreamState(const StreamState &) = delete;
  auto operator=(const StreamState &) -> StreamState & = delete;
  StreamState(StreamState &&) = delete;
  auto operator=(StreamState &&) -> StreamState & = delete;

  ~StreamState() noexcept;

  [[nodiscard]] auto GetId() const noexcept -> uint64_t;
  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetNative() const noexcept -> cudaStream_t;
  [[nodiscard]] auto IsExternal() const noexcept -> bool;

 private:
  uint64_t id_;
  Device device_;
  cudaStream_t stream_;
  std::shared_ptr<void> external_owner_;
  std::shared_ptr<ErrorSink> error_sink_;
  decltype(&cudaGetDevice) get_device_;
  decltype(&cudaSetDevice) set_device_;
  decltype(&cudaStreamDestroy) destroy_stream_;
  std::source_location location_;
  bool is_external_;
};

/**
 * Private construction and native-handle gateway for Runtime, ExecutionContext, Event, and tests.
 *
 * StreamState copies the cleanup function pointers it needs; the injected CudaApi table object need not outlive the
 * returned Stream.
 */
class StreamAccess final {
 public:
  [[nodiscard]] static auto CreateOwned(Device device, int32_t priority, std::shared_ptr<ErrorSink> error_sink,
                                        std::source_location location = std::source_location::current()) -> Stream;
  [[nodiscard]] static auto CreateOwned(Device device, int32_t priority, std::shared_ptr<ErrorSink> error_sink,
                                        const CudaApi &cuda_api,
                                        std::source_location location = std::source_location::current()) -> Stream;

  [[nodiscard]] static auto WrapExternal(Device device, cudaStream_t stream, std::shared_ptr<void> external_owner,
                                         std::shared_ptr<ErrorSink> error_sink,
                                         std::source_location location = std::source_location::current()) -> Stream;

  [[nodiscard]] static auto GetState(const Stream &stream) noexcept -> std::shared_ptr<StreamState>;
  [[nodiscard]] static auto GetNative(const Stream &stream) noexcept -> cudaStream_t;
};

}  // namespace ttl::internal
