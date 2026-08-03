#include "ttl/internal/ops/matmul_plan.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>

namespace ttl::internal {
namespace {

template <typename T>
void HashCombine(size_t &seed, T value) noexcept {
  constexpr auto golden_ratio = size_t{0x9e3779b97f4a7c15ULL};
  seed ^= std::hash<T>{}(value) + golden_ratio + (seed << 6U) + (seed >> 2U);
}

void HashLayout(size_t &seed, const MatrixLayoutSignature &layout) noexcept {
  HashCombine(seed, layout.rows_);
  HashCombine(seed, layout.columns_);
  HashCombine(seed, std::bit_cast<uint64_t>(layout.leading_dimension_));
  HashCombine(seed, std::bit_cast<uint64_t>(layout.batch_stride_));
  HashCombine(seed, layout.order_);
}

}  // namespace

auto MatmulAlgorithmCache::KeyHash::operator()(const MatmulAlgorithmKey &key) const noexcept -> size_t {
  auto seed = size_t{0};
  HashLayout(seed, key.lhs_);
  HashLayout(seed, key.rhs_);
  HashLayout(seed, key.output_);
  HashCombine(seed, std::bit_cast<uint64_t>(key.batch_count_));
  HashCombine(seed, key.data_type_);
  HashCombine(seed, key.compute_type_);
  HashCombine(seed, key.lhs_operation_);
  HashCombine(seed, key.rhs_operation_);
  HashCombine(seed, key.epilogue_);
  HashCombine(seed, key.lhs_alignment_);
  HashCombine(seed, key.rhs_alignment_);
  HashCombine(seed, key.output_alignment_);
  HashCombine(seed, key.bias_alignment_);
  HashCombine(seed, key.workspace_limit_bytes_);
  return seed;
}

auto MatmulAlgorithmCache::Find(const MatmulAlgorithmKey &key) -> std::optional<MatmulAlgorithmChoice> {
  std::scoped_lock lock{latch_};
  const auto iterator = index_.find(key);
  if (iterator == index_.end()) {
    return std::nullopt;
  }
  entries_.splice(entries_.begin(), entries_, iterator->second);
  return iterator->second->choice_;
}

void MatmulAlgorithmCache::Insert(const MatmulAlgorithmKey &key, const MatmulAlgorithmChoice &choice) {
  std::scoped_lock lock{latch_};
  const auto existing = index_.find(key);
  if (existing != index_.end()) {
    existing->second->choice_ = choice;
    entries_.splice(entries_.begin(), entries_, existing->second);
    return;
  }

  if (entries_.size() == MAXIMUM_ENTRY_COUNT) {
    index_.erase(entries_.back().key_);
    entries_.pop_back();
  }
  entries_.push_front(Entry{.key_ = key, .choice_ = choice});
  try {
    index_.emplace(entries_.front().key_, entries_.begin());
  } catch (...) {
    entries_.pop_front();
    throw;
  }
}

}  // namespace ttl::internal
