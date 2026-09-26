#include "kv_cache/manager.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <ranges>
#include <unordered_set>
#include <utility>

#include "common/exception.hpp"

namespace zephyr::kv_cache {

KVCacheManager::KVCacheManager(CacheCapacity capacity, std::vector<cache_group_id_t> cache_group_ids)
    : block_pool_(capacity.num_gpu_blocks_), capacity_(capacity), cache_group_ids_(std::move(cache_group_ids)) {
  if (capacity_.block_size_ == 0) {
    throw InvalidArgumentException("KV cache block size must be positive");
  }
  if (cache_group_ids_.empty()) {
    throw InvalidArgumentException("KV cache must contain at least one cache group");
  }

  std::unordered_set<cache_group_id_t> unique_group_ids;
  unique_group_ids.reserve(cache_group_ids_.size());
  for (const auto group_id : cache_group_ids_) {
    if (!unique_group_ids.insert(group_id).second) {
      throw InvalidArgumentException("KV cache group IDs must be unique");
    }
  }
}

auto KVCacheManager::GetComputedBlocks(std::span<const block_hash_t> block_hashes, size_t num_tokens) const
    -> ComputedBlocks {
  ComputedBlocks result;
  if (block_hashes.empty() || num_tokens == 0) {
    return result;
  }

  const auto max_cache_tokens = num_tokens - 1;
  const auto max_cached_blocks = max_cache_tokens / capacity_.block_size_;
  const auto num_blocks_to_check = std::min(max_cached_blocks, block_hashes.size());
  result.block_ids_.reserve(num_blocks_to_check);

  for (size_t block_index = 0; block_index < num_blocks_to_check; ++block_index) {
    const auto block_id = block_pool_.GetCommonCachedBlock(block_hashes[block_index], cache_group_ids_);
    if (!block_id.has_value()) {
      break;
    }
    result.block_ids_.push_back(*block_id);
  }
  result.num_computed_tokens_ = result.block_ids_.size() * capacity_.block_size_;
  return result;
}

auto KVCacheManager::ReservePrompt(sequence_id_t sequence_id, size_t num_tokens,
                                   std::span<const block_id_t> computed_blocks) -> bool {
  if (sequence_blocks_.contains(sequence_id)) {
    return false;
  }

  const auto num_required_blocks = GetRequiredBlockCount(num_tokens);
  if (computed_blocks.size() > num_required_blocks) {
    return false;
  }
  ValidateBlockIds(computed_blocks);

  const auto num_reserved_blocks = num_required_blocks - computed_blocks.size();
  const auto num_evictable_blocks = GetNumEvictableBlocks(computed_blocks);
  const auto num_unreserved_blocks = GetNumUnreservedBlocks();
  if (num_evictable_blocks > num_unreserved_blocks ||
      num_reserved_blocks > num_unreserved_blocks - num_evictable_blocks) {
    return false;
  }

  const auto non_null_blocks = GetNonNullBlockIds(computed_blocks);
  auto [sequence, inserted] = sequence_blocks_.try_emplace(sequence_id);
  if (!inserted) {
    return false;
  }

  bool blocks_touched = false;
  try {
    sequence->second.block_ids_.assign(computed_blocks.begin(), computed_blocks.end());
    sequence->second.num_cached_blocks_ = computed_blocks.size();
    sequence->second.reserved_blocks_ = num_reserved_blocks;
    block_pool_.Touch(non_null_blocks);
    blocks_touched = true;
  } catch (...) {
    if (blocks_touched) {
      block_pool_.FreeBlocks(non_null_blocks);
    }
    sequence_blocks_.erase(sequence);
    throw;
  }

  total_reserved_blocks_ += num_reserved_blocks;
  return true;
}

auto KVCacheManager::AllocateSlots(sequence_id_t sequence_id, size_t num_tokens,
                                   std::span<const block_id_t> computed_blocks)
    -> std::optional<std::vector<block_id_t>> {
  const auto num_required_blocks = GetRequiredBlockCount(num_tokens);
  if (auto sequence = sequence_blocks_.find(sequence_id); sequence != sequence_blocks_.end()) {
    auto &state = sequence->second;
    if (num_required_blocks <= state.block_ids_.size()) {
      return std::vector<block_id_t>{};
    }

    const auto num_new_blocks = num_required_blocks - state.block_ids_.size();
    const auto num_reserved_blocks = std::min(num_new_blocks, state.reserved_blocks_);
    const auto num_unreserved_blocks = num_new_blocks - num_reserved_blocks;
    if (num_unreserved_blocks > GetNumUnreservedBlocks()) {
      return std::nullopt;
    }

    state.block_ids_.reserve(num_required_blocks);
    auto new_blocks = block_pool_.GetNewBlocks(num_new_blocks);
    if (!new_blocks.has_value()) {
      return std::nullopt;
    }

    state.block_ids_.insert(state.block_ids_.end(), new_blocks->begin(), new_blocks->end());
    state.reserved_blocks_ -= num_reserved_blocks;
    total_reserved_blocks_ -= num_reserved_blocks;
    return new_blocks;
  }

  if (computed_blocks.size() > num_required_blocks) {
    return std::nullopt;
  }
  ValidateBlockIds(computed_blocks);

  const auto num_new_blocks = num_required_blocks - computed_blocks.size();
  const auto num_evictable_blocks = GetNumEvictableBlocks(computed_blocks);
  const auto num_unreserved_blocks = GetNumUnreservedBlocks();
  if (num_new_blocks > num_unreserved_blocks || num_evictable_blocks > num_unreserved_blocks - num_new_blocks) {
    return std::nullopt;
  }

  const auto non_null_blocks = GetNonNullBlockIds(computed_blocks);
  auto [sequence, inserted] = sequence_blocks_.try_emplace(sequence_id);
  if (!inserted) {
    return std::nullopt;
  }

  auto &state = sequence->second;
  bool blocks_touched = false;
  try {
    state.block_ids_.reserve(num_required_blocks);
    state.block_ids_.assign(computed_blocks.begin(), computed_blocks.end());
    state.num_cached_blocks_ = computed_blocks.size();
    block_pool_.Touch(non_null_blocks);
    blocks_touched = true;
    auto new_blocks = block_pool_.GetNewBlocks(num_new_blocks);
    if (!new_blocks.has_value()) {
      block_pool_.FreeBlocks(non_null_blocks);
      blocks_touched = false;
      sequence_blocks_.erase(sequence);
      return std::nullopt;
    }
    state.block_ids_.insert(state.block_ids_.end(), new_blocks->begin(), new_blocks->end());
    return new_blocks;
  } catch (...) {
    if (blocks_touched) {
      block_pool_.FreeBlocks(non_null_blocks);
    }
    sequence_blocks_.erase(sequence);
    throw;
  }
}

void KVCacheManager::CacheBlocks(sequence_id_t sequence_id, std::span<const block_hash_t> block_hashes,
                                 size_t num_computed_tokens) {
  const auto sequence = sequence_blocks_.find(sequence_id);
  if (sequence == sequence_blocks_.end()) {
    return;
  }

  auto &state = sequence->second;
  const auto num_full_blocks = std::min(num_computed_tokens / capacity_.block_size_, state.block_ids_.size());
  if (state.num_cached_blocks_ >= num_full_blocks) {
    return;
  }
  if (num_full_blocks > block_hashes.size()) {
    throw InvalidArgumentException("KV cache block hash metadata is shorter than the computed block count");
  }

  for (const auto group_id : cache_group_ids_) {
    block_pool_.CacheFullBlocks(state.block_ids_, block_hashes, state.num_cached_blocks_, num_full_blocks, group_id);
  }
  state.num_cached_blocks_ = num_full_blocks;
}

void KVCacheManager::Free(sequence_id_t sequence_id) {
  const auto sequence = sequence_blocks_.find(sequence_id);
  if (sequence == sequence_blocks_.end()) {
    return;
  }

  const auto &state = sequence->second;
  std::vector<block_id_t> blocks_to_free;
  blocks_to_free.reserve(state.block_ids_.size());
  for (const auto block_id : std::ranges::reverse_view(state.block_ids_)) {
    if (block_id != block_pool_.GetNullBlockId()) {
      blocks_to_free.push_back(block_id);
    }
  }

  block_pool_.FreeBlocks(blocks_to_free);
  total_reserved_blocks_ -= state.reserved_blocks_;
  sequence_blocks_.erase(sequence);
}

auto KVCacheManager::GetSlotMapping(sequence_id_t sequence_id, size_t start_token, size_t num_tokens) const
    -> std::optional<std::vector<int64_t>> {
  const auto sequence = sequence_blocks_.find(sequence_id);
  if (sequence == sequence_blocks_.end()) {
    return std::nullopt;
  }
  if (num_tokens > std::numeric_limits<size_t>::max() - start_token) {
    throw InvalidArgumentException("KV cache slot mapping token range overflows");
  }

  const auto &block_ids = sequence->second.block_ids_;
  std::vector<int64_t> slots;
  slots.reserve(num_tokens);
  for (size_t token = start_token; token < start_token + num_tokens; ++token) {
    const auto block_index = token / capacity_.block_size_;
    const auto offset = token % capacity_.block_size_;
    if (block_index >= block_ids.size()) {
      slots.push_back(PADDING_SLOT_ID);
      continue;
    }
    const auto block_id = block_ids[block_index];
    if (block_id == block_pool_.GetNullBlockId()) {
      slots.push_back(PADDING_SLOT_ID);
      continue;
    }
    const auto slot = (static_cast<uint64_t>(block_id) * capacity_.block_size_) + offset;
    if (!std::in_range<int64_t>(slot)) {
      throw InvalidArgumentException("KV cache slot does not fit in int64");
    }
    slots.push_back(static_cast<int64_t>(slot));
  }
  return slots;
}

auto KVCacheManager::GetBlockTable(sequence_id_t sequence_id) const -> std::span<const block_id_t> {
  return sequence_blocks_.at(sequence_id).block_ids_;
}

auto KVCacheManager::GetNumUnreservedBlocks() const -> size_t {
  const auto num_free_blocks = GetNumFreeBlocks();
  if (num_free_blocks < total_reserved_blocks_) {
    throw InternalException("KV cache reservations exceed the number of free blocks");
  }
  return num_free_blocks - total_reserved_blocks_;
}

auto KVCacheManager::GetRequiredBlockCount(size_t num_tokens) const noexcept -> size_t {
  const auto quotient = num_tokens / capacity_.block_size_;
  return quotient + (num_tokens % capacity_.block_size_ == 0 ? 0 : 1);
}

void KVCacheManager::ValidateBlockIds(std::span<const block_id_t> block_ids) const {
  std::unordered_set<block_id_t> unique_block_ids;
  unique_block_ids.reserve(block_ids.size());
  for (const auto block_id : block_ids) {
    const auto reference_count = block_pool_.GetBlockReferenceCount(block_id);
    if (block_id == block_pool_.GetNullBlockId()) {
      continue;
    }
    if (!unique_block_ids.insert(block_id).second) {
      throw InvalidArgumentException("KV cache sequence contains a duplicate physical block");
    }
    if (reference_count == std::numeric_limits<uint32_t>::max()) {
      throw InternalException("KV cache block reference count overflow");
    }
  }
}

auto KVCacheManager::GetNumEvictableBlocks(std::span<const block_id_t> block_ids) const -> size_t {
  size_t result = 0;
  for (const auto block_id : block_ids) {
    if (block_id != block_pool_.GetNullBlockId() && block_pool_.GetBlockReferenceCount(block_id) == 0) {
      ++result;
    }
  }
  return result;
}

auto KVCacheManager::GetNonNullBlockIds(std::span<const block_id_t> block_ids) const -> std::vector<block_id_t> {
  std::vector<block_id_t> result;
  result.reserve(block_ids.size());
  for (const auto block_id : block_ids) {
    if (block_id != block_pool_.GetNullBlockId()) {
      result.push_back(block_id);
    }
  }
  return result;
}

}  // namespace zephyr::kv_cache
