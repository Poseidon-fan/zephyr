#include "attention/kernels.cuh"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include <ttl/runtime/cuda_dtype.hpp>
#include <ttl/runtime/device_error.cuh>

namespace zephyr::attention {

constexpr int THREADS_PER_COPY = 256;
constexpr int64_t MAX_COPY_BLOCKS = 65535;

template <ttl::CudaStorageType T>
__global__ void ReshapeAndCacheKernel(const T *key, const T *value, T *key_cache, T *value_cache,
                                      const int64_t *slot_mapping, int64_t num_tokens, int64_t key_stride,
                                      int64_t value_stride, CacheShape cache_shape,
                                      ttl::CudaDeviceErrorContext error_context) {
  if (error_context.record_ != nullptr &&
      error_context.record_->code_ != static_cast<uint32_t>(ttl::CudaDeviceErrorCode::NONE)) {
    return;
  }
  // Each block copies whole token rows. Negative slots leave padding rows unwritten.
  const auto key_elements = cache_shape.num_kv_heads_ * cache_shape.key_head_dim_;
  const auto value_elements = cache_shape.num_kv_heads_ * cache_shape.value_head_dim_;
  for (auto token = static_cast<int64_t>(blockIdx.x); token < num_tokens; token += gridDim.x) {
    const auto slot = slot_mapping[token];
    if (slot < 0) {
      continue;
    }
    const auto physical_block = slot / cache_shape.block_size_;
    if (physical_block >= cache_shape.num_blocks_) {
      if (threadIdx.x == 0) {
        ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INDEX_OUT_OF_BOUNDS, token,
                                   static_cast<uint64_t>(physical_block), cache_shape.num_blocks_);
      }
      continue;
    }
    const auto offset = slot % cache_shape.block_size_;
    for (auto index = static_cast<int64_t>(threadIdx.x); index < key_elements; index += blockDim.x) {
      const auto kv_head = index / cache_shape.key_head_dim_;
      const auto dimension = index % cache_shape.key_head_dim_;
      const auto cache_offset = GetKeyCacheOffset(physical_block, kv_head, dimension, offset, cache_shape);
      key_cache[cache_offset] = key[(token * key_stride) + index];
    }
    for (auto index = static_cast<int64_t>(threadIdx.x); index < value_elements; index += blockDim.x) {
      const auto kv_head = index / cache_shape.value_head_dim_;
      const auto dimension = index % cache_shape.value_head_dim_;
      const auto cache_offset = GetValueCacheOffset(physical_block, kv_head, dimension, offset, cache_shape);
      value_cache[cache_offset] = value[(token * value_stride) + index];
    }
  }
}

// Validate every sequence, including empty ones, before binary search can consume the cumulative lengths.
__global__ void ValidateGatherMetadataKernel(const int32_t *block_tables, const int32_t *cu_seqlens_k,
                                             int64_t total_tokens, int64_t batch_size, int64_t max_blocks,
                                             CacheShape cache_shape, ttl::CudaDeviceErrorContext error_context) {
  for (auto sequence = static_cast<int64_t>(blockIdx.x); sequence < batch_size; sequence += gridDim.x) {
    const auto begin = static_cast<int64_t>(cu_seqlens_k[sequence]);
    const auto end = static_cast<int64_t>(cu_seqlens_k[sequence + 1]);
    if (begin < 0 || begin > total_tokens || (sequence == 0 && begin != 0)) {
      if (threadIdx.x == 0) {
        ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INVALID_VALUE, sequence,
                                   static_cast<uint64_t>(begin), total_tokens);
      }
      continue;
    }
    if (end < begin || end > total_tokens || (sequence == batch_size - 1 && end != total_tokens)) {
      if (threadIdx.x == 0) {
        ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INVALID_VALUE, sequence + 1,
                                   static_cast<uint64_t>(end), total_tokens);
      }
      continue;
    }

    const auto length = end - begin;
    const auto num_blocks = (length / cache_shape.block_size_) + (length % cache_shape.block_size_ == 0 ? 0 : 1);
    if (num_blocks > max_blocks) {
      if (threadIdx.x == 0) {
        ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INDEX_OUT_OF_BOUNDS, sequence,
                                   static_cast<uint64_t>(num_blocks - 1), max_blocks);
      }
      continue;
    }
    // Padded table entries are unused and need not reference physical pages.
    for (auto block = static_cast<int64_t>(threadIdx.x); block < num_blocks; block += blockDim.x) {
      const auto table_index = (sequence * max_blocks) + block;
      const auto physical_block = block_tables[table_index];
      if (physical_block < 0 || physical_block >= cache_shape.num_blocks_) {
        ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INDEX_OUT_OF_BOUNDS, table_index,
                                   static_cast<uint64_t>(physical_block), cache_shape.num_blocks_);
      }
    }
  }
}

template <ttl::CudaStorageType T>
__global__ void GatherKvCacheKernel(const T *key_cache, const T *value_cache, const int32_t *block_tables,
                                    const int32_t *cu_seqlens_k, T *key_output, T *value_output, int64_t total_tokens,
                                    int64_t batch_size, int64_t max_blocks, CacheShape cache_shape,
                                    ttl::CudaDeviceErrorContext error_context) {
  // The validation launch precedes this kernel on the same stream. Its error record is now stable.
  if (error_context.record_ != nullptr &&
      error_context.record_->code_ != static_cast<uint32_t>(ttl::CudaDeviceErrorCode::NONE)) {
    return;
  }
  __shared__ int64_t physical_block;
  __shared__ int64_t offset;
  const auto key_elements = cache_shape.num_kv_heads_ * cache_shape.key_head_dim_;
  const auto value_elements = cache_shape.num_kv_heads_ * cache_shape.value_head_dim_;
  for (auto token = static_cast<int64_t>(blockIdx.x); token < total_tokens; token += gridDim.x) {
    // Cumulative lengths identify the sequence owning this packed output row.
    // Thread zero resolves its page once for all threads copying the K/V elements.
    if (threadIdx.x == 0) {
      int64_t low = 0;
      int64_t high = batch_size;
      while ((low + 1) < high) {
        const auto middle = low + ((high - low) / 2);
        if (token < cu_seqlens_k[middle]) {
          high = middle;
        } else {
          low = middle;
        }
      }
      const auto position = token - cu_seqlens_k[low];
      const auto logical_block = position / cache_shape.block_size_;
      physical_block = block_tables[(low * max_blocks) + logical_block];
      offset = position % cache_shape.block_size_;
    }
    __syncthreads();

    for (auto index = static_cast<int64_t>(threadIdx.x); index < key_elements; index += blockDim.x) {
      const auto kv_head = index / cache_shape.key_head_dim_;
      const auto dimension = index % cache_shape.key_head_dim_;
      const auto cache_offset = GetKeyCacheOffset(physical_block, kv_head, dimension, offset, cache_shape);
      key_output[(token * key_elements) + index] = key_cache[cache_offset];
    }
    for (auto index = static_cast<int64_t>(threadIdx.x); index < value_elements; index += blockDim.x) {
      const auto kv_head = index / cache_shape.value_head_dim_;
      const auto dimension = index % cache_shape.value_head_dim_;
      const auto cache_offset = GetValueCacheOffset(physical_block, kv_head, dimension, offset, cache_shape);
      value_output[(token * value_elements) + index] = value_cache[cache_offset];
    }
    // Finish reading this token's shared page address before resolving the next one.
    __syncthreads();
  }
}

void LaunchReshapeAndCache(cudaStream_t stream, ttl::DType dtype, const void *key, const void *value, void *key_cache,
                           void *value_cache, const int64_t *slot_mapping, int64_t num_tokens, int64_t key_stride,
                           int64_t value_stride, const CacheShape &cache_shape,
                           const ttl::CudaDeviceErrorContext &error_context) {
  if (num_tokens == 0) {
    return;
  }
  const auto blocks = static_cast<unsigned int>(std::min(num_tokens, MAX_COPY_BLOCKS));
  ttl::DispatchCudaFloatingDType(dtype, "ReshapeAndCache", [&]<ttl::CudaStorageType T>(std::type_identity<T>) {
    ReshapeAndCacheKernel<<<blocks, THREADS_PER_COPY, 0, stream>>>(
        static_cast<const T *>(key), static_cast<const T *>(value), static_cast<T *>(key_cache),
        static_cast<T *>(value_cache), slot_mapping, num_tokens, key_stride, value_stride, cache_shape, error_context);
  });
}

void LaunchGatherKvCache(cudaStream_t stream, ttl::DType dtype, const void *key_cache, const void *value_cache,
                         const int32_t *block_tables, const int32_t *cu_seqlens_k, void *key_output, void *value_output,
                         int64_t total_tokens, int64_t batch_size, int64_t max_blocks, const CacheShape &cache_shape,
                         const ttl::CudaDeviceErrorContext &error_context) {
  const auto metadata_blocks = static_cast<unsigned int>(std::min(batch_size, MAX_COPY_BLOCKS));
  ValidateGatherMetadataKernel<<<metadata_blocks, THREADS_PER_COPY, 0, stream>>>(
      block_tables, cu_seqlens_k, total_tokens, batch_size, max_blocks, cache_shape, error_context);
  if (total_tokens == 0) {
    return;
  }
  const auto blocks = static_cast<unsigned int>(std::min(total_tokens, MAX_COPY_BLOCKS));
  ttl::DispatchCudaFloatingDType(dtype, "GatherKvCache", [&]<ttl::CudaStorageType T>(std::type_identity<T>) {
    GatherKvCacheKernel<<<blocks, THREADS_PER_COPY, 0, stream>>>(
        static_cast<const T *>(key_cache), static_cast<const T *>(value_cache), block_tables, cu_seqlens_k,
        static_cast<T *>(key_output), static_cast<T *>(value_output), total_tokens, batch_size, max_blocks, cache_shape,
        error_context);
  });
}

}  // namespace zephyr::attention
