#include "kv_cache/cache_engine.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

#include <ttl/tensor/dtype.hpp>
#include <ttl/tensor/shape.hpp>

#include "common/exception.hpp"

namespace zephyr::kv_cache {

auto GetCacheBlockBytes(std::span<const LayerCacheSpec> layer_specs, ttl::DType dtype, size_t block_size) -> size_t {
  if (block_size == 0) {
    throw InvalidArgumentException("KV cache block size must be positive");
  }
  if (!ttl::IsFloating(dtype)) {
    throw InvalidArgumentException("KV cache dtype must be a floating-point type");
  }
  if (layer_specs.empty()) {
    throw InvalidArgumentException("KV cache must contain at least one layer specification");
  }

  const auto element_size = ttl::GetDTypeInfo(dtype).size_bytes_;
  constexpr auto maximum = std::numeric_limits<size_t>::max();
  size_t total_bytes = 0;
  for (const auto &spec : layer_specs) {
    if (spec.num_kv_heads_ == 0 || spec.key_head_dim_ == 0 || spec.value_head_dim_ == 0) {
      throw InvalidArgumentException("KV cache layer dimensions must be positive");
    }
    if (spec.key_head_dim_ > maximum - spec.value_head_dim_) {
      throw InvalidArgumentException("KV cache block byte count overflows");
    }
    auto layer_bytes = spec.key_head_dim_ + spec.value_head_dim_;
    for (const auto factor : std::array{spec.num_kv_heads_, block_size, element_size}) {
      if (layer_bytes > maximum / factor) {
        throw InvalidArgumentException("KV cache block byte count overflows");
      }
      layer_bytes *= factor;
    }
    if (total_bytes > maximum - layer_bytes) {
      throw InvalidArgumentException("KV cache block byte count overflows");
    }
    total_bytes += layer_bytes;
  }
  return total_bytes;
}

CacheEngine::CacheEngine(ttl::Runtime &runtime, CacheConfig config, ttl::Device device)
    : config_(std::move(config)), device_(device) {
  if (config_.capacity_.num_gpu_blocks_ == 0) {
    throw InvalidArgumentException("KV cache GPU block count must be positive");
  }
  static_cast<void>(GetCacheBlockBytes(config_.layer_specs_, config_.dtype_, config_.capacity_.block_size_));

  auto context = runtime.CreateExecutionContext(device_);
  layer_caches_.reserve(config_.layer_specs_.size());
  for (const auto &spec : config_.layer_specs_) {
    layer_caches_.push_back(AllocateLayerCache(spec, context));
  }
  context.Synchronize();
}

auto CacheEngine::AllocateLayerCache(const LayerCacheSpec &spec, ttl::ExecutionContext &context) const -> LayerCache {
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
