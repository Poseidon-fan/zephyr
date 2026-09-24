#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/types.hpp"
#include "kv_cache/block_pool.hpp"

namespace zephyr::kv_cache {

/** Physical blocks reused from a contiguous prefix of a sequence. */
struct ComputedBlocks final {
  std::vector<block_id_t> block_ids_;
  size_t num_computed_tokens_{0};
};

/**
 * Owns sequence block tables and coordinates them with the physical block pool.
 * Mutating operations are expected to be serialized by the caller.
 */
class KVCacheManager final {
 public:
  KVCacheManager(size_t num_gpu_blocks, size_t block_size, std::vector<cache_group_id_t> cache_group_ids = {0});

  KVCacheManager(const KVCacheManager &) = delete;
  auto operator=(const KVCacheManager &) -> KVCacheManager & = delete;
  KVCacheManager(KVCacheManager &&) noexcept = default;
  auto operator=(KVCacheManager &&) noexcept -> KVCacheManager & = default;
  ~KVCacheManager() noexcept = default;

  /** Find the longest cached full-block prefix, leaving the final block for recomputation. */
  [[nodiscard]] auto GetComputedBlocks(std::span<const block_hash_t> block_hashes, size_t num_tokens) const
      -> ComputedBlocks;

  /** Pin a cached prefix and reserve the remaining logical prompt capacity. */
  [[nodiscard]] auto ReservePrompt(sequence_id_t sequence_id, size_t num_tokens,
                                   std::span<const block_id_t> computed_blocks) -> bool;

  /** Allocate physical blocks needed to reach the requested sequence length. */
  [[nodiscard]] auto AllocateSlots(sequence_id_t sequence_id, size_t num_tokens,
                                   std::span<const block_id_t> computed_blocks = {})
      -> std::optional<std::vector<block_id_t>>;

  /** Register newly completed blocks in every configured prefix-cache group. */
  void CacheBlocks(sequence_id_t sequence_id, std::span<const block_hash_t> block_hashes, size_t num_computed_tokens);

  /** Release all physical blocks and reservations owned by a sequence. */
  void Free(sequence_id_t sequence_id);

  /** Return the sequence's logical block table while the sequence remains unchanged. */
  [[nodiscard]] auto GetBlockIds(sequence_id_t sequence_id) const -> std::optional<std::span<const block_id_t>>;

  [[nodiscard]] auto GetBlockSize() const noexcept -> size_t { return block_size_; }
  [[nodiscard]] auto GetNumGpuBlocks() const noexcept -> size_t { return block_pool_.GetNumGpuBlocks(); }
  [[nodiscard]] auto GetNumFreeBlocks() const noexcept -> size_t { return block_pool_.GetNumFreeBlocks(); }
  [[nodiscard]] auto GetNumUnreservedBlocks() const -> size_t;

 private:
  struct SequenceBlocks final {
    std::vector<block_id_t> block_ids_;
    size_t num_cached_blocks_{0};
    size_t reserved_blocks_{0};
  };

  [[nodiscard]] auto GetRequiredBlockCount(size_t num_tokens) const noexcept -> size_t;
  void ValidateBlockIds(std::span<const block_id_t> block_ids) const;
  [[nodiscard]] auto GetNumEvictableBlocks(std::span<const block_id_t> block_ids) const -> size_t;
  [[nodiscard]] auto GetNonNullBlockIds(std::span<const block_id_t> block_ids) const -> std::vector<block_id_t>;
  void DebugAssertReservationInvariant() const;

  BlockPool block_pool_;
  size_t block_size_;
  std::vector<cache_group_id_t> cache_group_ids_;
  std::unordered_map<sequence_id_t, SequenceBlocks> sequence_blocks_;
  size_t total_reserved_blocks_{0};
};

}  // namespace zephyr::kv_cache
