#include "ttl/internal/runtime/memory/allocator_lifecycle.hpp"

#include <source_location>
#include <string>
#include <string_view>

#include "ttl/common/error.hpp"

namespace ttl::internal {
namespace {

[[nodiscard]] auto FormatStateError(std::string_view resource, std::string_view state) -> std::string {
  std::string message{resource};
  message.append(" is ");
  message.append(state);
  return message;
}

}  // namespace

auto AllocatorLifecycle::GetStatus() const noexcept -> AllocatorStatus {
  return status_.load(std::memory_order_acquire);
}

void AllocatorLifecycle::RequireRunning(std::string_view resource, std::source_location location) const {
  if (GetStatus() != AllocatorStatus::RUNNING) {
    throw InvalidArgumentError(FormatStateError(resource, "not accepting new work"), location);
  }
}

void AllocatorLifecycle::MarkFailed() noexcept {
  auto expected = AllocatorStatus::RUNNING;
  static_cast<void>(status_.compare_exchange_strong(expected, AllocatorStatus::FAILED, std::memory_order_acq_rel,
                                                    std::memory_order_acquire));
}

auto AllocatorLifecycle::BeginClosing() noexcept -> bool {
  auto current = status_.load(std::memory_order_acquire);
  while (current != AllocatorStatus::CLOSED && current != AllocatorStatus::CLOSING) {
    if (status_.compare_exchange_weak(current, AllocatorStatus::CLOSING, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      return true;
    }
  }
  return current == AllocatorStatus::CLOSING;
}

void AllocatorLifecycle::MarkClosed() noexcept { status_.store(AllocatorStatus::CLOSED, std::memory_order_release); }

}  // namespace ttl::internal
