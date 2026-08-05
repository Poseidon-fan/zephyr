#pragma once

#include <concepts>
#include <memory>
#include <source_location>
#include <type_traits>

#include "ttl/common/device.hpp"

namespace ttl::internal {

class EventAccess;
class EventState;

}  // namespace ttl::internal

namespace ttl {

/**
 * @brief Immutable, copyable completion token recorded once on a CUDA stream.
 *
 * Copies share the same native event and lifetime. Query is non-blocking; Synchronize is an explicit host-blocking
 * boundary. Cross-stream waits are submitted through ExecutionContext rather than the Event itself.
 */
class Event final {
 public:
  Event() = delete;
  Event(const Event &) noexcept = default;
  auto operator=(const Event &) noexcept -> Event & = default;
  Event(Event &&) noexcept = default;
  auto operator=(Event &&) noexcept -> Event & = default;

  [[nodiscard]] auto GetDevice(std::source_location location = std::source_location::current()) const -> Device;
  [[nodiscard]] auto Query(std::source_location location = std::source_location::current()) const -> bool;
  void Synchronize(std::source_location location = std::source_location::current()) const;

 private:
  friend class internal::EventAccess;

  explicit Event(std::shared_ptr<internal::EventState> state) noexcept;

  std::shared_ptr<internal::EventState> state_;
};

static_assert(!std::default_initializable<Event>);
static_assert(std::copy_constructible<Event>);
static_assert(std::is_nothrow_copy_constructible_v<Event>);
static_assert(std::is_nothrow_copy_assignable_v<Event>);
static_assert(std::is_nothrow_move_constructible_v<Event>);
static_assert(std::is_nothrow_move_assignable_v<Event>);

}  // namespace ttl
