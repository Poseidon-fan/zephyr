#include "kv_cache/cache_engine.hpp"

#include <cstdint>
#include <utility>

#include <ttl/tensor/dtype.hpp>
#include <ttl/tensor/shape.hpp>

#include "common/exception.hpp"

namespace zephyr::kv_cache {

CacheEngine::CacheEngine(ttl::Runtime &runtime, CacheConfig config, ttl::Device device)
    : config_(std::move(config)), device_(device) {
  if (config_.capacity_.block_size_ == 0) {
    throw InvalidArgumentException("KV cache block size must be positive");
  }
  if (config_.capacity_.num_gpu_blocks_ == 0) {
    throw InvalidArgumentException("KV cache GPU block count must be positive");
  }
  if (!ttl::IsFloating(config_.dtype_)) {
    throw InvalidArgumentException("KV cache dtype must be a floating-point type");
  }
  if (config_.layer_specs_.empty()) {
    throw InvalidArgumentException("KV cache must contain at least one layer specification");
  }

  auto context = runtime.CreateExecutionContext(device_);
  layer_caches_.reserve(config_.layer_specs_.size());
  for (const auto &spec : config_.layer_specs_) {
    layer_caches_.push_back(AllocateLayerCache(spec, context));
  }
  context.Synchronize();
}

auto CacheEngine::AllocateLayerCache(const LayerCacheSpec &spec, ttl::ExecutionContext &context) const -> LayerCache {
  if (spec.num_kv_heads_ == 0 || spec.key_head_dim_ == 0 || spec.value_head_dim_ == 0) {
    throw InvalidArgumentException("KV cache layer dimensions must be positive");
  }

  const auto element_size = ttl::GetDTypeInfo(config_.dtype_).size_bytes_;
  if (16 % element_size != 0) {
    throw InvalidArgumentException("KV cache dtype cannot use the 16-byte key packing");
  }
  const auto packed_dimension = 16 / element_size;
  if (spec.key_head_dim_ % packed_dimension != 0) {
    throw InvalidArgumentException("key head dimension must be divisible by KV cache packing");
  }

  if (!std::in_range<int64_t>(config_.capacity_.num_gpu_blocks_) ||
      !std::in_range<int64_t>(config_.capacity_.block_size_) || !std::in_range<int64_t>(spec.num_kv_heads_) ||
      !std::in_range<int64_t>(spec.key_head_dim_ / packed_dimension) || !std::in_range<int64_t>(spec.value_head_dim_)) {
    throw InvalidArgumentException("KV cache dimensions exceed the supported tensor range");
  }

  const auto num_gpu_blocks = static_cast<int64_t>(config_.capacity_.num_gpu_blocks_);
  const auto block_size = static_cast<int64_t>(config_.capacity_.block_size_);
  const auto num_kv_heads = static_cast<int64_t>(spec.num_kv_heads_);
  const auto packed_key_head_dim = static_cast<int64_t>(spec.key_head_dim_ / packed_dimension);
  const auto value_head_dim = static_cast<int64_t>(spec.value_head_dim_);
  const auto packing = static_cast<int64_t>(packed_dimension);

  const auto key_shape = ttl::Shape{
      num_gpu_blocks, num_kv_heads, packed_key_head_dim, block_size, packing,
  };
  const auto value_shape = ttl::Shape{
      num_gpu_blocks,
      num_kv_heads,
      value_head_dim,
      block_size,
  };

  return LayerCache{.key_cache_ = ttl::Empty(context, key_shape, config_.dtype_),
                    .value_cache_ = ttl::Empty(context, value_shape, config_.dtype_),
                    .device_ = device_};
}

}  // namespace zephyr::kv_cache
