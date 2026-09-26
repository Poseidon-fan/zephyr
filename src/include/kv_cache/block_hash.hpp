#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "common/types.hpp"
#include "kv_cache/types.hpp"

namespace zephyr::kv_cache {

/**
 * Append XXH64 chain hashes for newly completed token blocks, ignoring an incomplete tail.
 * Existing hashes must describe an unchanged token prefix with the same positive block size.
 * Each hash includes its parent (zero for the first block) and all tokens in little-endian order.
 * These are probabilistic 64-bit cache keys, not collision-free identities or security boundaries;
 * the prefix cache does not compare token contents after a hash match.
 */
void AppendBlockHashes(std::span<const token_id_t> tokens, size_t block_size, std::vector<block_hash_t> &hashes);

}  // namespace zephyr::kv_cache
