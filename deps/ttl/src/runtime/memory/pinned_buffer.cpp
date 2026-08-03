#include "ttl/runtime/pinned_buffer.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

#include "ttl/internal/runtime/memory/pinned_allocator.hpp"

namespace ttl {

PinnedBuffer::PinnedBuffer(std::shared_ptr<internal::PinnedBlock> block) noexcept : block_(std::move(block)) {}

auto PinnedBuffer::GetData() noexcept -> void * { return block_ == nullptr ? nullptr : block_->GetData(); }

auto PinnedBuffer::GetData() const noexcept -> const void * { return block_ == nullptr ? nullptr : block_->GetData(); }

auto PinnedBuffer::GetSizeBytes() const noexcept -> size_t {
  return block_ == nullptr ? size_t{0} : block_->GetSizeBytes();
}

auto PinnedBuffer::AsBytes() noexcept -> std::span<std::byte> {
  return {static_cast<std::byte *>(GetData()), GetSizeBytes()};
}

auto PinnedBuffer::AsBytes() const noexcept -> std::span<const std::byte> {
  return {static_cast<const std::byte *>(GetData()), GetSizeBytes()};
}

}  // namespace ttl
