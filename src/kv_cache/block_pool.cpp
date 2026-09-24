#include "kv_cache/block_pool.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "common/exception.hpp"

namespace zephyr::kv_cache {

BlockPool::BlockPool(size_t num_gpu_blocks) {
  constexpr auto max_block_count = static_cast<size_t>(std::numeric_limits<block_id_t>::max()) - 2;
  if (num_gpu_blocks == 0) {
    throw InvalidArgumentException("KV cache block pool must contain at least one GPU block");
  }
  if (num_gpu_blocks > max_block_count) {
    throw InvalidArgumentException("KV cache block pool is too large for block_id_t");
  }
  num_gpu_blocks_ = num_gpu_blocks;
  free_queue_head_ = static_cast<block_id_t>(num_gpu_blocks);
  free_queue_tail_ = static_cast<block_id_t>(num_gpu_blocks + 1);
  free_queue_size_ = num_gpu_blocks;
  blocks_.resize(num_gpu_blocks + 2);
  for (size_t index = 0; index < blocks_.size(); ++index) {
    blocks_[index].block_id_ = static_cast<block_id_t>(index);
  }

  blocks_[free_queue_head_].next_free_ = 0;
  for (block_id_t block_id = 0; block_id < num_gpu_blocks; ++block_id) {
    blocks_[block_id].previous_free_ = block_id == 0 ? free_queue_head_ : static_cast<block_id_t>(block_id - 1);
    blocks_[block_id].next_free_ = block_id + 1 == num_gpu_blocks ? free_queue_tail_ : block_id + 1;
  }
  blocks_[free_queue_tail_].previous_free_ = static_cast<block_id_t>(num_gpu_blocks - 1);
  blocks_[free_queue_head_].previous_free_ = INVALID_BLOCK_ID;
  blocks_[free_queue_tail_].next_free_ = INVALID_BLOCK_ID;

  null_block_id_ = PopFreeQueueFront();
  blocks_[null_block_id_].is_null_ = true;
}

auto BlockPool::GetNewBlocks(size_t count) -> std::optional<std::vector<block_id_t>> {
  if (count > free_queue_size_) {
    return std::nullopt;
  }

  std::vector<block_id_t> result;
  result.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    const auto block_id = PopFreeQueueFront();
    auto &block = GetBlock(block_id);
    EvictCachedBlock(block);
    block.reference_count_ = 1;
    result.push_back(block_id);
  }
  return result;
}

void BlockPool::Touch(std::span<const block_id_t> block_ids) {
  for (const auto block_id : block_ids) {
    auto &block = GetBlock(block_id);
    if (block.reference_count_ == 0 && !block.is_null_) {
      RemoveFromFreeQueue(block_id);
    }
    if (block.reference_count_ == std::numeric_limits<uint32_t>::max()) {
      throw InternalException("KV cache block reference count overflow");
    }
    ++block.reference_count_;
  }
}

void BlockPool::FreeBlocks(std::span<const block_id_t> ordered_block_ids) {
  std::unordered_map<block_id_t, uint32_t> release_counts;
  release_counts.reserve(ordered_block_ids.size());
  for (const auto block_id : ordered_block_ids) {
    static_cast<void>(GetBlock(block_id));
    auto &release_count = release_counts[block_id];
    if (release_count == std::numeric_limits<uint32_t>::max()) {
      throw InternalException("KV cache block release count overflow");
    }
    ++release_count;
  }

  for (const auto &[block_id, release_count] : release_counts) {
    if (GetBlock(block_id).reference_count_ < release_count) {
      throw InternalException("KV cache block reference count underflow for block " + std::to_string(block_id));
    }
  }

  for (const auto block_id : ordered_block_ids) {
    const auto release_count = release_counts.find(block_id);
    if (release_count == release_counts.end()) {
      continue;
    }
    auto &block = GetBlock(block_id);
    block.reference_count_ -= release_count->second;
    release_counts.erase(release_count);
    if (block.reference_count_ == 0 && !block.is_null_) {
      AppendToFreeQueue(block_id);
    }
  }
}

void BlockPool::CacheFullBlocks(std::span<const block_id_t> block_ids, std::span<const block_hash_t> block_hashes,
                                size_t num_cached_blocks, size_t num_full_blocks, cache_group_id_t group_id) {
  if (num_cached_blocks > num_full_blocks) {
    throw InvalidArgumentException("cached block count cannot exceed full block count");
  }
  if (num_full_blocks > block_ids.size() || num_full_blocks > block_hashes.size()) {
    throw InvalidArgumentException("full block metadata is shorter than the requested block count");
  }
  if (num_cached_blocks >= num_full_blocks) {
    return;
  }

  for (size_t index = num_cached_blocks; index < num_full_blocks; ++index) {
    static_cast<void>(GetBlock(block_ids[index]));
  }

  for (size_t index = num_cached_blocks; index < num_full_blocks; ++index) {
    const auto block_id = block_ids[index];
    auto &block = GetBlock(block_id);
    if (block.is_null_) {
      continue;
    }

    const BlockHashWithGroupId key{.block_hash_ = block_hashes[index], .group_id_ = group_id};
    if (ContainsCachedHash(block, key)) {
      continue;
    }
    block.cached_hashes_.push_back(key);
    auto &cached_blocks = cached_blocks_[key];
    if (std::ranges::find(cached_blocks, block_id) == cached_blocks.end()) {
      cached_blocks.push_back(block_id);
    }
  }
}

auto BlockPool::GetCommonCachedBlock(block_hash_t block_hash, std::span<const cache_group_id_t> group_ids) const
    -> std::optional<block_id_t> {
  if (group_ids.empty()) {
    return std::nullopt;
  }

  const auto first_key = BlockHashWithGroupId{.block_hash_ = block_hash, .group_id_ = group_ids.front()};
  const auto first_group = cached_blocks_.find(first_key);
  if (first_group == cached_blocks_.end()) {
    return std::nullopt;
  }

  for (const auto candidate : first_group->second) {
    const bool present_for_all_groups = std::ranges::all_of(group_ids, [&](cache_group_id_t group_id) {
      const auto key = BlockHashWithGroupId{.block_hash_ = block_hash, .group_id_ = group_id};
      const auto group_blocks = cached_blocks_.find(key);
      return group_blocks != cached_blocks_.end() &&
             std::ranges::find(group_blocks->second, candidate) != group_blocks->second.end();
    });
    if (present_for_all_groups) {
      return candidate;
    }
  }
  return std::nullopt;
}

auto BlockPool::GetBlockReferenceCount(block_id_t block_id) const -> uint32_t {
  return GetBlock(block_id).reference_count_;
}

auto BlockPool::GetBlock(block_id_t block_id) -> BlockRecord & {
  if (static_cast<size_t>(block_id) >= num_gpu_blocks_) {
    throw InvalidArgumentException("KV cache block ID is outside the physical block pool");
  }
  return blocks_[block_id];
}

auto BlockPool::GetBlock(block_id_t block_id) const -> const BlockRecord & {
  if (static_cast<size_t>(block_id) >= num_gpu_blocks_) {
    throw InvalidArgumentException("KV cache block ID is outside the physical block pool");
  }
  return blocks_[block_id];
}

void BlockPool::RemoveCachedHashes(BlockRecord &block) {
  for (const auto &key : block.cached_hashes_) {
    const auto cached_blocks = cached_blocks_.find(key);
    if (cached_blocks == cached_blocks_.end()) {
      continue;
    }
    auto &block_ids = cached_blocks->second;
    std::erase(block_ids, block.block_id_);
    if (block_ids.empty()) {
      cached_blocks_.erase(cached_blocks);
    }
  }
  block.cached_hashes_.clear();
}

void BlockPool::EvictCachedBlock(BlockRecord &block) {
  if (block.cached_hashes_.empty()) {
    return;
  }
  RemoveCachedHashes(block);
}

void BlockPool::RemoveFromFreeQueue(block_id_t block_id) {
  auto &block = blocks_[block_id];
  if (block.previous_free_ == INVALID_BLOCK_ID || block.next_free_ == INVALID_BLOCK_ID) {
    throw InternalException("KV cache free-list removal encountered an unlinked block");
  }
  blocks_[block.previous_free_].next_free_ = block.next_free_;
  blocks_[block.next_free_].previous_free_ = block.previous_free_;
  block.previous_free_ = INVALID_BLOCK_ID;
  block.next_free_ = INVALID_BLOCK_ID;
  --free_queue_size_;
}

void BlockPool::AppendToFreeQueue(block_id_t block_id) {
  auto &block = blocks_[block_id];
  if (block.is_null_) {
    throw InternalException("the KV cache null block cannot enter the free list");
  }
  if (block.previous_free_ != INVALID_BLOCK_ID || block.next_free_ != INVALID_BLOCK_ID) {
    throw InternalException("KV cache free-list append encountered a linked block");
  }
  const auto previous_tail = blocks_[free_queue_tail_].previous_free_;
  blocks_[previous_tail].next_free_ = block_id;
  block.previous_free_ = previous_tail;
  block.next_free_ = free_queue_tail_;
  blocks_[free_queue_tail_].previous_free_ = block_id;
  ++free_queue_size_;
}

auto BlockPool::PopFreeQueueFront() -> block_id_t {
  const auto block_id = blocks_[free_queue_head_].next_free_;
  if (block_id == free_queue_tail_) {
    throw OutOfMemoryException("KV cache block pool has no free blocks");
  }
  RemoveFromFreeQueue(block_id);
  return block_id;
}

auto BlockPool::ContainsCachedHash(const BlockRecord &block, const BlockHashWithGroupId &key) const -> bool {
  const auto &hashes = block.cached_hashes_;
  return std::ranges::find(hashes, key) != hashes.end();
}

}  // namespace zephyr::kv_cache
