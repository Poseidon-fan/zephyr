#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace zephyr::kv_cache {

/** Physical page capacity shared by the CPU block manager and GPU cache storage. */
struct CacheCapacity final {
  size_t block_size_;
  size_t num_gpu_blocks_;
};

/** Slot value used for a token that has no physical cache destination. */
inline constexpr int64_t PADDING_SLOT_ID = -1;

/** Identifies one physical KV cache block. */
using block_id_t = uint32_t;

/** Identifies a KV cache group when layers share a physical prefix cache. */
using cache_group_id_t = uint32_t;

/** Content hash of one complete logical KV cache block. */
using block_hash_t = uint64_t;

/** Associates a content hash with the KV cache group that owns it. */
struct BlockHashWithGroupId final {
  block_hash_t block_hash_;
  cache_group_id_t group_id_;

  [[nodiscard]] auto operator==(const BlockHashWithGroupId &) const noexcept -> bool = default;
};

/** Hash functor for a grouped prefix-cache key. */
struct BlockHashWithGroupIdHash final {
  [[nodiscard]] auto operator()(const BlockHashWithGroupId &key) const noexcept -> size_t {
    const auto hash = std::hash<block_hash_t>{}(key.block_hash_);
    const auto group = std::hash<cache_group_id_t>{}(key.group_id_);
    return hash ^ (group + static_cast<size_t>(0x9e3779b9U) + (hash << 6U) + (hash >> 2U));
  }
};

}  // namespace zephyr::kv_cache
