#pragma once

#include <cstddef>
#include <memory>
#include <span>

namespace ttl::internal {

class PinnedAllocator;
class PinnedAllocatorImpl;
class PinnedBlock;
class PinnedBufferAccess;

}  // namespace ttl::internal

namespace ttl {

/**
 * Copyable owner of a fixed-size CUDA page-locked host allocation.
 *
 * Copies share the same mutable bytes. Releasing the last owner is asynchronous with respect to previously submitted
 * transfers: the underlying allocation is not reused until every recorded CUDA stream has completed. A moved-from
 * buffer is a valid empty buffer whose observers return null, zero, or an empty span.
 */
class PinnedBuffer final {
 public:
  PinnedBuffer(const PinnedBuffer &) noexcept = default;
  auto operator=(const PinnedBuffer &) noexcept -> PinnedBuffer & = default;
  PinnedBuffer(PinnedBuffer &&) noexcept = default;
  auto operator=(PinnedBuffer &&) noexcept -> PinnedBuffer & = default;

  ~PinnedBuffer() noexcept = default;

  [[nodiscard]] auto GetData() noexcept -> void *;
  [[nodiscard]] auto GetData() const noexcept -> const void *;
  [[nodiscard]] auto GetSizeBytes() const noexcept -> size_t;
  [[nodiscard]] auto AsBytes() noexcept -> std::span<std::byte>;
  [[nodiscard]] auto AsBytes() const noexcept -> std::span<const std::byte>;

 private:
  friend class internal::PinnedAllocator;
  friend class internal::PinnedAllocatorImpl;
  friend class internal::PinnedBufferAccess;

  explicit PinnedBuffer(std::shared_ptr<internal::PinnedBlock> block) noexcept;

  std::shared_ptr<internal::PinnedBlock> block_;
};

}  // namespace ttl
