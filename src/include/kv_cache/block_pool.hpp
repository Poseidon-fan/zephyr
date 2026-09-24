#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "kv_cache/types.hpp"

namespace zephyr::kv_cache {

/**
 * Owns the host-side physical block state used by paged KV caching.
 *
 * Block zero is a permanent null block. Free blocks are maintained in LRU
 * order, and releasing a block keeps its prefix hash until that block is reused.
 */
class BlockPool final {
 public:
  explicit BlockPool(size_t num_gpu_blocks);

  BlockPool(const BlockPool &) = delete;
  auto operator=(const BlockPool &) -> BlockPool & = delete;
  BlockPool(BlockPool &&) noexcept = default;
  auto operator=(BlockPool &&) noexcept -> BlockPool & = default;
  ~BlockPool() noexcept = default;

  /** Allocate blocks from the least-recently-used end of the free queue. */
  [[nodiscard]] auto GetNewBlocks(size_t count) -> std::optional<std::vector<block_id_t>>;

  /** Reacquire cached blocks for a new sequence. */
  void Touch(std::span<const block_id_t> block_ids);

  /** Release references in eviction order; freed blocks remain prefix-cacheable. */
  void FreeBlocks(std::span<const block_id_t> ordered_block_ids);

  /** Associate complete logical block hashes with physical blocks. */
  void CacheFullBlocks(std::span<const block_id_t> block_ids, std::span<const block_hash_t> block_hashes,
                       size_t num_cached_blocks, size_t num_full_blocks, cache_group_id_t group_id = 0);

  /** Find one physical block carrying the hash for every requested group. */
  [[nodiscard]] auto GetCommonCachedBlock(block_hash_t block_hash, std::span<const cache_group_id_t> group_ids) const
      -> std::optional<block_id_t>;

  /** Return the permanent null block used for placeholder positions. */
  [[nodiscard]] auto GetNullBlockId() const noexcept -> block_id_t { return null_block_id_; }

  /** Return the number of live sequence references to a physical block. */
  [[nodiscard]] auto GetBlockReferenceCount(block_id_t block_id) const -> uint32_t;

  [[nodiscard]] auto GetNumGpuBlocks() const noexcept -> size_t { return num_gpu_blocks_; }
  [[nodiscard]] auto GetNumFreeBlocks() const noexcept -> size_t { return free_queue_size_; }

 private:
  static constexpr block_id_t INVALID_BLOCK_ID = std::numeric_limits<block_id_t>::max();

  struct BlockRecord final {
    block_id_t block_id_{0};
    uint32_t reference_count_{0};
    std::vector<BlockHashWithGroupId> cached_hashes_;
    block_id_t previous_free_{INVALID_BLOCK_ID};
    block_id_t next_free_{INVALID_BLOCK_ID};
    bool is_null_{false};
  };

  [[nodiscard]] auto GetBlock(block_id_t block_id) -> BlockRecord &;
  [[nodiscard]] auto GetBlock(block_id_t block_id) const -> const BlockRecord &;

  void RemoveCachedHashes(BlockRecord &block);
  void EvictCachedBlock(BlockRecord &block);
  void RemoveFromFreeQueue(block_id_t block_id);
  void AppendToFreeQueue(block_id_t block_id);
  [[nodiscard]] auto PopFreeQueueFront() -> block_id_t;
  [[nodiscard]] auto ContainsCachedHash(const BlockRecord &block, const BlockHashWithGroupId &key) const -> bool;

  std::vector<BlockRecord> blocks_;
  block_id_t free_queue_head_{0};
  block_id_t free_queue_tail_{0};
  size_t free_queue_size_{0};
  std::unordered_map<BlockHashWithGroupId, std::vector<block_id_t>, BlockHashWithGroupIdHash> cached_blocks_;
  size_t num_gpu_blocks_{0};
  block_id_t null_block_id_{0};
};

}  // namespace zephyr::kv_cache
