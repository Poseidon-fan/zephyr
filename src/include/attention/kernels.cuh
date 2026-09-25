#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include <ttl/runtime/device_error.hpp>
#include <ttl/tensor/dtype.hpp>
#include <ttl/tensor/tensor.hpp>

namespace zephyr::attention {

/**
 * Shared physical dimensions passed from the tensor wrappers to CUDA kernels.
 * Keys use [block, KV head, dim / packing, token, packing] and values use
 * [block, KV head, dim, token].
 */
struct CacheShape final {
  int64_t num_blocks_;
  int64_t num_kv_heads_;
  int64_t key_head_dim_;
  int64_t value_head_dim_;
  int64_t block_size_;
  int64_t key_packing_;
  ttl::DType dtype_;
};

/** Reads and validates the cache dimensions shared by all attention kernels. */
[[nodiscard]] auto GetCacheShape(const ttl::Tensor &key_cache, const ttl::Tensor &value_cache) -> CacheShape;

inline constexpr int64_t PAGED_DECODE_PARTITION_SIZE = 512;

void LaunchReshapeAndCache(cudaStream_t stream, ttl::DType dtype, const void *key, const void *value, void *key_cache,
                           void *value_cache, const int64_t *slot_mapping, int64_t num_tokens, int64_t key_stride,
                           int64_t value_stride, const CacheShape &cache_shape,
                           const ttl::CudaDeviceErrorContext &error_context);

void LaunchGatherKvCache(cudaStream_t stream, ttl::DType dtype, const void *key_cache, const void *value_cache,
                         const int32_t *block_tables, const int32_t *cu_seqlens_k, void *key_output, void *value_output,
                         int64_t total_tokens, int64_t batch_size, int64_t max_blocks, const CacheShape &cache_shape,
                         const ttl::CudaDeviceErrorContext &error_context);

void LaunchPagedDecode(cudaStream_t stream, ttl::DType dtype, const void *query, const void *key_cache,
                       const void *value_cache, const int32_t *block_tables, const int32_t *context_lengths,
                       void *output, int64_t batch_size, int64_t query_heads, int64_t max_blocks, int64_t query_stride,
                       float softmax_scale, const CacheShape &cache_shape, int64_t max_context_len, void *workspace,
                       const ttl::CudaDeviceErrorContext &error_context);

#ifdef __CUDACC__

/** Element offset of K[block, head, dimension / packing, token, dimension % packing]. */
__device__ inline auto GetKeyCacheOffset(int64_t physical_block, int64_t kv_head, int64_t dimension, int64_t offset,
                                         const CacheShape &cache_shape) -> int64_t {
  const auto packed_dimension = dimension / cache_shape.key_packing_;
  const auto packed_offset = dimension % cache_shape.key_packing_;
  const auto head_offset = (physical_block * cache_shape.num_kv_heads_) + kv_head;
  const auto dimension_offset =
      (head_offset * (cache_shape.key_head_dim_ / cache_shape.key_packing_)) + packed_dimension;
  const auto block_offset = (dimension_offset * cache_shape.block_size_) + offset;
  return (block_offset * cache_shape.key_packing_) + packed_offset;
}

/** Element offset of V[block, head, dimension, token]. */
__device__ inline auto GetValueCacheOffset(int64_t physical_block, int64_t kv_head, int64_t dimension, int64_t offset,
                                           const CacheShape &cache_shape) -> int64_t {
  const auto head_offset = (physical_block * cache_shape.num_kv_heads_) + kv_head;
  const auto dimension_offset = (head_offset * cache_shape.value_head_dim_) + dimension;
  return (dimension_offset * cache_shape.block_size_) + offset;
}

#endif

}  // namespace zephyr::attention
