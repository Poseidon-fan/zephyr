#pragma once

#include <concepts>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "ttl/device.hpp"

namespace ttl::internal {

class StreamAccess;
class StreamState;

}  // namespace ttl::internal

namespace ttl {

/**
 * A copyable handle to one owned or externally managed non-default CUDA stream.
 *
 * Copies share the same stream identity and lifetime. Stream deliberately exposes no arbitrary submission,
 * synchronization, or native-handle API; execution contexts and the CUDA interop layer provide those operations.
 */
class Stream final {
 public:
  Stream() = delete;
  Stream(const Stream &) noexcept = default;
  auto operator=(const Stream &) noexcept -> Stream & = default;

  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetId() const noexcept -> uint64_t;
  [[nodiscard]] auto IsExternal() const noexcept -> bool;

 private:
  friend class internal::StreamAccess;

  explicit Stream(std::shared_ptr<internal::StreamState> state) noexcept;

  std::shared_ptr<internal::StreamState> state_;
};

static_assert(!std::default_initializable<Stream>);
static_assert(std::copy_constructible<Stream>);
static_assert(std::is_nothrow_copy_constructible_v<Stream>);
static_assert(std::is_nothrow_copy_assignable_v<Stream>);

}  // namespace ttl
