#pragma once

#include <atomic>
#include <cstdint>
#include <source_location>
#include <string_view>

namespace ttl::internal {

enum class AllocatorStatus : uint8_t {
  RUNNING,
  FAILED,
  CLOSING,
  CLOSED,
};

class AllocatorLifecycle final {
 public:
  AllocatorLifecycle() noexcept = default;

  AllocatorLifecycle(const AllocatorLifecycle &) = delete;
  auto operator=(const AllocatorLifecycle &) -> AllocatorLifecycle & = delete;
  AllocatorLifecycle(AllocatorLifecycle &&) = delete;
  auto operator=(AllocatorLifecycle &&) -> AllocatorLifecycle & = delete;

  [[nodiscard]] auto GetStatus() const noexcept -> AllocatorStatus;
  void RequireRunning(std::string_view resource, std::source_location location) const;
  void MarkFailed() noexcept;

  /**
   * Enter or resume closing.
   *
   * Returns true in CLOSING so a shutdown that previously failed can resume, and false only after CLOSED is terminal.
   */
  [[nodiscard]] auto BeginClosing() noexcept -> bool;
  void MarkClosed() noexcept;

 private:
  std::atomic<AllocatorStatus> status_{AllocatorStatus::RUNNING};
};

}  // namespace ttl::internal
