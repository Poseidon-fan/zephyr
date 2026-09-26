#include "kv_cache/block_hash.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

#define XXH_INLINE_ALL
#define XXH_NO_XXH3
#include <xxhash.h>

#include "common/exception.hpp"

namespace zephyr::kv_cache {
namespace {

template <typename T>
void HashLittleEndian(XXH64_state_t &state, T value) {
  std::array<unsigned char, sizeof(T)> bytes{};
  for (size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<unsigned char>(value >> (index * 8U));
  }
  static_cast<void>(XXH64_update(&state, bytes.data(), bytes.size()));
}

}  // namespace

void AppendBlockHashes(std::span<const token_id_t> tokens, size_t block_size, std::vector<block_hash_t> &hashes) {
  if (block_size == 0) {
    throw InvalidArgumentException("KV cache block size must be positive");
  }
  const auto num_full_blocks = tokens.size() / block_size;
  if (hashes.size() > num_full_blocks) {
    throw InvalidArgumentException("KV cache block hashes exceed the complete token prefix");
  }

  if (num_full_blocks > hashes.capacity()) {
    const auto capacity = hashes.capacity();
    const auto grown_capacity = capacity <= hashes.max_size() / 2 ? capacity * 2 : hashes.max_size();
    hashes.reserve(std::max(num_full_blocks, grown_capacity));
  }
  auto parent = hashes.empty() ? block_hash_t{0} : hashes.back();
  for (auto block_index = hashes.size(); block_index < num_full_blocks; ++block_index) {
    const auto block_tokens = tokens.subspan(block_index * block_size, block_size);
    XXH64_state_t state{};
    static_cast<void>(XXH64_reset(&state, 0));
    HashLittleEndian(state, parent);
    if constexpr (std::endian::native == std::endian::little) {
      const auto bytes = std::as_bytes(block_tokens);
      static_cast<void>(XXH64_update(&state, bytes.data(), bytes.size()));
    } else {
      for (const auto token : block_tokens) {
        HashLittleEndian(state, static_cast<uint32_t>(token));
      }
    }
    parent = XXH64_digest(&state);
    hashes.push_back(parent);
  }
}

}  // namespace zephyr::kv_cache
