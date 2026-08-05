#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <vector>

#include "ttl/runtime/stream.hpp"

namespace ttl::internal {

class DeviceAllocator;
class Storage;

enum class ScratchGrowthPolicy : uint8_t {
  GROWABLE,
  FIXED_CAPACITY,
};

/** Non-owning byte range that may be used to enqueue work only while its ScratchArena scope is alive. */
class ScratchAllocation final {
 public:
  [[nodiscard]] auto GetData() const noexcept -> void *;
  [[nodiscard]] auto GetSizeBytes() const noexcept -> size_t;
  [[nodiscard]] auto GetOffsetBytes() const noexcept -> size_t;

 private:
  friend class ScratchArena;

  ScratchAllocation(void *data, size_t size_bytes, size_t offset_bytes) noexcept;

  void *data_;
  size_t size_bytes_;
  size_t offset_bytes_;
};

/**
 * @brief Stream-local bump allocator backed by persistent device Storage.
 *
 * ScratchArena is not thread-safe. Its owning ExecutionLane is already protected by ContextUseGuard.
 */
class ScratchArena final {
 public:
  class Scope final {
   public:
    Scope(const Scope &) = delete;
    auto operator=(const Scope &) -> Scope & = delete;
    Scope(Scope &&other) noexcept;
    auto operator=(Scope &&) -> Scope & = delete;

    ~Scope() noexcept;

    [[nodiscard]] auto AllocateBytes(size_t bytes, size_t alignment = 256,
                                     std::source_location location = std::source_location::current())
        -> ScratchAllocation;

   private:
    friend class ScratchArena;

    Scope(ScratchArena &arena, uint64_t generation, size_t marker) noexcept;
    void Retain(std::shared_ptr<Storage> storage);

    ScratchArena *arena_;
    uint64_t generation_;
    size_t marker_;
    std::vector<std::shared_ptr<Storage>> retained_blocks_;
  };

  ScratchArena(Stream stream, std::shared_ptr<DeviceAllocator> allocator,
               std::source_location location = std::source_location::current());

  ScratchArena(const ScratchArena &) = delete;
  auto operator=(const ScratchArena &) -> ScratchArena & = delete;
  ScratchArena(ScratchArena &&) = delete;
  auto operator=(ScratchArena &&) -> ScratchArena & = delete;

  [[nodiscard]] auto MakeScope(ScratchGrowthPolicy growth_policy = ScratchGrowthPolicy::GROWABLE,
                               std::source_location location = std::source_location::current()) -> Scope;

  /** Ensure capacity outside an active scope. Existing submitted work remains protected by Storage retirement. */
  void Reserve(size_t capacity_bytes, std::source_location location = std::source_location::current());

  [[nodiscard]] auto GetCapacityBytes() const noexcept -> size_t;
  [[nodiscard]] auto GetHighWaterBytes() const noexcept -> size_t;
  [[nodiscard]] auto GetStorage() const noexcept -> const std::shared_ptr<Storage> &;

 private:
  [[nodiscard]] auto Allocate(Scope &scope, size_t bytes, size_t alignment, std::source_location location)
      -> ScratchAllocation;
  void Grow(Scope *scope, size_t required_bytes, std::source_location location);
  void Release(uint64_t generation, size_t marker) noexcept;

  Stream stream_;
  std::shared_ptr<DeviceAllocator> allocator_;
  std::shared_ptr<Storage> storage_;
  size_t offset_{0};
  size_t high_water_bytes_{0};
  uint64_t generation_{0};
  uint64_t active_generation_{0};
  ScratchGrowthPolicy active_growth_policy_{ScratchGrowthPolicy::GROWABLE};
  bool scope_active_{false};
};

}  // namespace ttl::internal
