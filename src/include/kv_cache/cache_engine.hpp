#pragma once

#include <cstddef>
#include <vector>

#include <ttl/common/device.hpp>
#include <ttl/runtime/execution_context.hpp>
#include <ttl/runtime/runtime.hpp>
#include <ttl/tensor/dtype.hpp>
#include <ttl/tensor/tensor.hpp>

#include "kv_cache/types.hpp"

namespace zephyr::kv_cache {

/** Layer-local model dimensions needed to allocate one paged KV cache. */
struct LayerCacheSpec final {
  size_t num_kv_heads_;
  size_t key_head_dim_;
  size_t value_head_dim_;
};

/** Parameters shared by every layer in one rank-local cache engine. */
struct CacheConfig final {
  CacheCapacity capacity_;
  ttl::DType dtype_;
  std::vector<LayerCacheSpec> layer_specs_;
};

/** GPU cache tensors and their device for one transformer layer. */
struct LayerCache final {
  ttl::Tensor key_cache_;
  ttl::Tensor value_cache_;
  ttl::Device device_;
};

/**
 * Allocates rank-local GPU KV pages for every attention layer.
 *
 * The runtime and device are borrowed by the engine. The engine owns its tensor
 * handles and releases them before the runtime is shut down.
 */
class CacheEngine final {
 public:
  CacheEngine(ttl::Runtime &runtime, CacheConfig config, ttl::Device device);

  CacheEngine(const CacheEngine &) = delete;
  auto operator=(const CacheEngine &) -> CacheEngine & = delete;
  CacheEngine(CacheEngine &&) = delete;
  auto operator=(CacheEngine &&) -> CacheEngine & = delete;
  ~CacheEngine() noexcept = default;

  [[nodiscard]] auto GetConfig() const noexcept -> const CacheConfig & { return config_; }
  [[nodiscard]] auto GetNumLayers() const noexcept -> size_t { return layer_caches_.size(); }
  [[nodiscard]] auto GetLayerCache(size_t layer_index) -> LayerCache & { return layer_caches_.at(layer_index); }
  [[nodiscard]] auto GetLayerCache(size_t layer_index) const -> const LayerCache & {
    return layer_caches_.at(layer_index);
  }

 private:
  [[nodiscard]] auto AllocateLayerCache(const LayerCacheSpec &spec, ttl::ExecutionContext &context) const -> LayerCache;

  CacheConfig config_;
  ttl::Device device_;
  std::vector<LayerCache> layer_caches_;
};

}  // namespace zephyr::kv_cache
