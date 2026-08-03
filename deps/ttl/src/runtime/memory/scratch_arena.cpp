#include "ttl/internal/runtime/memory/scratch_arena.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <utility>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/runtime/memory/device_allocator.hpp"
#include "ttl/internal/tensor/storage.hpp"
#include "ttl/runtime/stream.hpp"

namespace ttl::internal {
namespace {

constexpr size_t SCRATCH_STORAGE_ALIGNMENT = 256;
constexpr size_t MAXIMUM_SCRATCH_ALIGNMENT = SCRATCH_STORAGE_ALIGNMENT;

[[nodiscard]] auto GetGrowthCapacity(size_t required_bytes) noexcept -> size_t {
  constexpr auto maximum_power_of_two = size_t{1} << (std::numeric_limits<size_t>::digits - 1);
  return required_bytes <= maximum_power_of_two ? std::bit_ceil(required_bytes) : required_bytes;
}

[[nodiscard]] auto FormatFixedCapacityError(size_t required_bytes, size_t capacity_bytes, size_t high_water_bytes)
    -> std::string {
  std::string message{"scratch arena cannot grow while using fixed capacity: required="};
  message.append(std::to_string(required_bytes));
  message.append(", capacity=");
  message.append(std::to_string(capacity_bytes));
  message.append(", high_water=");
  message.append(std::to_string(high_water_bytes));
  return message;
}

}  // namespace

ScratchAllocation::ScratchAllocation(void *data, size_t size_bytes, size_t offset_bytes) noexcept
    : data_(data), size_bytes_(size_bytes), offset_bytes_(offset_bytes) {}

auto ScratchAllocation::GetData() const noexcept -> void * { return data_; }

auto ScratchAllocation::GetSizeBytes() const noexcept -> size_t { return size_bytes_; }

auto ScratchAllocation::GetOffsetBytes() const noexcept -> size_t { return offset_bytes_; }

ScratchArena::Scope::Scope(ScratchArena &arena, uint64_t generation, size_t marker) noexcept
    : arena_(&arena), generation_(generation), marker_(marker) {}

ScratchArena::Scope::Scope(Scope &&other) noexcept
    : arena_(std::exchange(other.arena_, nullptr)),
      generation_(other.generation_),
      marker_(other.marker_),
      retained_blocks_(std::move(other.retained_blocks_)) {}

ScratchArena::Scope::~Scope() noexcept {
  if (arena_ != nullptr) {
    arena_->Release(generation_, marker_);
  }
}

auto ScratchArena::Scope::AllocateBytes(size_t bytes, size_t alignment, std::source_location location)
    -> ScratchAllocation {
  if (arena_ == nullptr) {
    throw InvalidArgumentError("cannot allocate from a moved-from scratch scope", location);
  }
  return arena_->Allocate(*this, bytes, alignment, location);
}

void ScratchArena::Scope::Retain(std::shared_ptr<Storage> storage) { retained_blocks_.push_back(std::move(storage)); }

ScratchArena::ScratchArena(Stream stream, std::shared_ptr<DeviceAllocator> allocator, std::source_location location)
    : stream_(std::move(stream)), allocator_(std::move(allocator)) {
  if (allocator_ == nullptr) {
    throw InvalidArgumentError("scratch arena requires a device allocator", location);
  }
  if (stream_.GetDevice() != allocator_->GetDevice()) {
    throw InvalidArgumentError("scratch arena stream and allocator must belong to the same device", location);
  }
}

auto ScratchArena::MakeScope(ScratchGrowthPolicy growth_policy, std::source_location location) -> Scope {
  if (scope_active_) {
    throw InvalidArgumentError("nested scratch arena scopes are not supported", location);
  }
  if (generation_ == std::numeric_limits<uint64_t>::max()) {
    throw OverflowError("scratch arena scope generation space exhausted", location);
  }

  generation_++;
  active_generation_ = generation_;
  active_growth_policy_ = growth_policy;
  scope_active_ = true;
  return Scope{*this, active_generation_, offset_};
}

void ScratchArena::Reserve(size_t capacity_bytes, std::source_location location) {
  if (scope_active_) {
    throw InvalidArgumentError("cannot reserve scratch capacity while a scope is active", location);
  }
  if (capacity_bytes <= GetCapacityBytes()) {
    return;
  }
  Grow(nullptr, capacity_bytes, location);
}

auto ScratchArena::GetCapacityBytes() const noexcept -> size_t {
  return storage_ == nullptr ? 0 : storage_->GetCapacityBytes();
}

auto ScratchArena::GetHighWaterBytes() const noexcept -> size_t { return high_water_bytes_; }

auto ScratchArena::GetStorage() const noexcept -> const std::shared_ptr<Storage> & { return storage_; }

auto ScratchArena::Allocate(Scope &scope, size_t bytes, size_t alignment, std::source_location location)
    -> ScratchAllocation {
  if (!scope_active_ || scope.generation_ != active_generation_ || scope.arena_ != this) {
    throw InvalidArgumentError("scratch scope is not active", location);
  }
  if (!std::has_single_bit(alignment) || alignment > MAXIMUM_SCRATCH_ALIGNMENT) {
    throw InvalidArgumentError("scratch alignment must be a power of two no greater than 256", location);
  }
  if (bytes == 0) {
    return ScratchAllocation{nullptr, 0, offset_};
  }

  const auto aligned_offset = AlignUp(offset_, alignment, "scratch allocation alignment", location);
  const auto end = CheckedAdd(aligned_offset, bytes, "scratch allocation byte range", location);
  if (end > GetCapacityBytes()) {
    if (active_growth_policy_ == ScratchGrowthPolicy::FIXED_CAPACITY) {
      throw CaptureError(FormatFixedCapacityError(end, GetCapacityBytes(), high_water_bytes_), location);
    }
    Grow(&scope, end, location);
  }

  auto *base = static_cast<std::byte *>(storage_->GetBasePointer());
  offset_ = end;
  high_water_bytes_ = std::max(high_water_bytes_, end);
  return ScratchAllocation{base + aligned_offset, bytes, aligned_offset};
}

void ScratchArena::Grow(Scope *scope, size_t required_bytes, std::source_location location) {
  const auto new_capacity = GetGrowthCapacity(required_bytes);
  auto new_storage = allocator_->Allocate(stream_, new_capacity, SCRATCH_STORAGE_ALIGNMENT,
                                          AllocationContext{
                                              .operation_ = "scratch arena growth",
                                              .output_shape_ = std::nullopt,
                                              .dtype_ = std::nullopt,
                                              .location_ = location,
                                          });

  if (scope != nullptr && storage_ != nullptr) {
    scope->Retain(storage_);
  }
  storage_ = std::move(new_storage);
}

void ScratchArena::Release(uint64_t generation, size_t marker) noexcept {
  if (!scope_active_ || generation != active_generation_ || marker > offset_) {
    std::terminate();
  }
  offset_ = marker;
  scope_active_ = false;
}

}  // namespace ttl::internal
