// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM paged attention for TTL's CUDA submission and storage interfaces.
// Reference: vllm-project/vllm@258f8de91f99b40a4dfb234e55a09e9493c6ece8, attention_kernels.cuh.
// Copyright (c) 2023, The vLLM team.
// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
// See LICENSE for attribution and the Apache-2.0 license terms.

#include "attention/kernels.cuh"
#include "attention/paged_decode.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include <ttl/runtime/cuda_dtype.hpp>
#include <ttl/runtime/cuda_math.cuh>
#include <ttl/runtime/device_error.cuh>

#include "common/exception.hpp"

namespace zephyr::attention {

/** Packed Q/K/V storage; BF16 vectors retain the alignment of their native pairs. */
template <typename T, int WIDTH>
struct alignas((std::is_same_v<T, __nv_bfloat16> && WIDTH > 1) ? alignof(__nv_bfloat162) : (sizeof(T) * WIDTH))
    CacheVector {
  T values_[WIDTH];
};

/** Accumulate each vector lane in FP32, then reduce lanes and the cooperating threads. */
template <int thread_group_size, typename T, int WIDTH, int NUM_VECTORS>
__device__ inline auto QkDot(const CacheVector<T, WIDTH> (&query)[NUM_VECTORS],
                             const CacheVector<T, WIDTH> (&key)[NUM_VECTORS]) -> float {
  float products[WIDTH];
#pragma unroll
  for (int lane = 0; lane < WIDTH; ++lane) {
    products[lane] = ttl::ToElementwiseFloat(query[0].values_[lane]) * ttl::ToElementwiseFloat(key[0].values_[lane]);
  }
#pragma unroll
  for (int vector = 1; vector < NUM_VECTORS; ++vector) {
#pragma unroll
    for (int lane = 0; lane < WIDTH; ++lane) {
      products[lane] = fmaf(ttl::ToElementwiseFloat(query[vector].values_[lane]),
                            ttl::ToElementwiseFloat(key[vector].values_[lane]), products[lane]);
    }
  }
  float result = products[0];
#pragma unroll
  for (int lane = 1; lane < WIDTH; ++lane) {
    result += products[lane];
  }
#pragma unroll
  for (int offset = thread_group_size / 2; offset >= 1; offset /= 2) {
    result += __shfl_xor_sync(0xffffffffU, result, offset);
  }
  return result;
}

/** Multiply probabilities and V in their storage dtype, preserving each dtype's reduction order. */
template <typename T, int WIDTH>
__device__ inline auto ValueDot(const CacheVector<T, WIDTH> &probability, const CacheVector<T, WIDTH> &value) -> float {
  if constexpr (std::is_same_v<T, __half>) {
    auto sum = __hmul2(__halves2half2(probability.values_[0], probability.values_[1]),
                       __halves2half2(value.values_[0], value.values_[1]));
#pragma unroll
    for (int lane = 2; lane < WIDTH; lane += 2) {
      const auto product = __hmul2(__halves2half2(probability.values_[lane], probability.values_[lane + 1]),
                                   __halves2half2(value.values_[lane], value.values_[lane + 1]));
      // Accumulate even and odd lanes in half before converting the final pair to float.
      sum = __hadd2(sum, product);
    }
    const auto result = __half22float2(sum);
    return result.x + result.y;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    float pairs[WIDTH / 2];
#pragma unroll
    for (int lane = 0; lane < WIDTH; lane += 2) {
      const auto product = __hmul2(__halves2bfloat162(probability.values_[lane], probability.values_[lane + 1]),
                                   __halves2bfloat162(value.values_[lane], value.values_[lane + 1]));
      const auto converted = __bfloat1622float2(product);
      pairs[lane / 2] = converted.x + converted.y;
    }
    float result = pairs[0];
#pragma unroll
    for (int pair = 1; pair < WIDTH / 2; ++pair) {
      result += pairs[pair];
    }
    return result;
  } else {
    float products[WIDTH];
#pragma unroll
    for (int lane = 0; lane < WIDTH; ++lane) {
      products[lane] = probability.values_[lane] * value.values_[lane];
    }
    float result = products[0];
#pragma unroll
    for (int lane = 1; lane < WIDTH; ++lane) {
      result += products[lane];
    }
    return result;
  }
}

constexpr int WARP_SIZE = 32;
constexpr int DECODE_THREADS = 128;
constexpr int DECODE_WARPS = DECODE_THREADS / WARP_SIZE;
constexpr int PARTITION_SIZE = static_cast<int>(PAGED_DECODE_PARTITION_SIZE);
constexpr float SOFTMAX_EPSILON = 1e-6F;

/** Validate each sequence once before attention reads its page table. Invalid rows become empty. */
__global__ void PrepareDecodeLengths(const int32_t *block_tables, const int32_t *context_lengths,
                                     int32_t *checked_lengths, int max_blocks, int block_size, int64_t num_blocks,
                                     int max_context_len, ttl::CudaDeviceErrorContext error_context) {
  const auto sequence = static_cast<int>(blockIdx.x);
  const auto thread = static_cast<int>(threadIdx.x);
  const auto length = context_lengths[sequence];
  // The host bound is already capped by the table capacity and must cover every actual length.
  if (length < 0 || length > max_context_len) {
    if (thread == 0) {
      checked_lengths[sequence] = 0;
      ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INVALID_VALUE, sequence,
                                 static_cast<uint64_t>(length));
    }
    return;
  }
  const auto pages = (length / block_size) + static_cast<int>(length % block_size != 0);
  bool invalid = false;
  for (auto page = thread; page < pages; page += DECODE_THREADS) {
    const auto index = (sequence * max_blocks) + page;
    const auto physical = block_tables[index];
    if (physical < 0 || physical >= num_blocks) {
      invalid = true;
      ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INDEX_OUT_OF_BOUNDS, index,
                                 static_cast<uint64_t>(physical), num_blocks);
    }
  }
  const auto invalid_sequence = __syncthreads_or(static_cast<int>(invalid));
  if (thread == 0) {
    checked_lengths[sequence] = invalid_sequence != 0 ? 0 : length;
  }
}

/** Sum each warp, then sum its leaders and broadcast to the entire block. */
__device__ auto BlockSum(float value, float *warp_sums) -> float {
  const auto lane = threadIdx.x % WARP_SIZE;
  const auto warp = threadIdx.x / WARP_SIZE;
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    value += __shfl_xor_sync(0xffffffffU, value, mask);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  if (lane < DECODE_WARPS) {
    value = warp_sums[lane];
  }
#pragma unroll
  for (int mask = DECODE_WARPS / 2; mask >= 1; mask /= 2) {
    value += __shfl_xor_sync(0xffffffffU, value, mask);
  }
  return __shfl_sync(0xffffffffU, value, 0);
}

/** One block computes one query head over a 512-token partition. Dimensions are compile-time constants. */
template <ttl::CudaStorageType T, int head_size, int block_size>
__global__ void PagedDecodeKernel(const T *__restrict__ query, const T *__restrict__ key_cache,
                                  const T *__restrict__ value_cache, const int32_t *__restrict__ block_tables,
                                  const int32_t *__restrict__ context_lengths, T *__restrict__ partial_output,
                                  float *__restrict__ exp_sums, float *__restrict__ max_logits, int kv_heads,
                                  int max_blocks, int query_stride, int kv_block_stride, int kv_head_stride,
                                  float softmax_scale) {
  constexpr int thread_group_size = WARP_SIZE / block_size;
  constexpr int thread_groups = DECODE_THREADS / thread_group_size;
  constexpr int key_vector_size = 16 / (thread_group_size * sizeof(T));
  constexpr int key_vectors_per_thread = head_size / (thread_group_size * key_vector_size);
  constexpr int key_packing = 16 / sizeof(T);
  using KeyVector = CacheVector<T, key_vector_size>;

  const auto sequence = static_cast<int>(blockIdx.y);
  const auto head = static_cast<int>(blockIdx.x);
  const auto heads = static_cast<int>(gridDim.x);
  const auto partition = static_cast<int>(blockIdx.z);
  const auto partitions = static_cast<int>(gridDim.z);
  const auto length = context_lengths[sequence];
  if (partition * PARTITION_SIZE >= length) {
    return;
  }
  const auto context_blocks = (length + block_size - 1) / block_size;
  const auto first_block = partition * (PARTITION_SIZE / block_size);
  const auto last_block = min(first_block + (PARTITION_SIZE / block_size), context_blocks);
  const auto first_token = first_block * block_size;
  const auto last_token = min(first_token + ((last_block - first_block) * block_size), length);
  const auto tokens = last_token - first_token;
  const auto kv_head = head / (heads / kv_heads);
  const auto thread = static_cast<int>(threadIdx.x);
  const auto warp = thread / WARP_SIZE;
  const auto lane = thread % WARP_SIZE;
  const auto group = thread / thread_group_size;
  const auto group_lane = thread % thread_group_size;

  // Q is reused by all pages. Each thread group cooperatively loads its vector lanes once.
  __shared__ KeyVector query_vectors[thread_group_size][key_vectors_per_thread];
  const auto *query_row = query + (sequence * query_stride) + (head * head_size);
#pragma unroll
  for (int index = group; index < key_vectors_per_thread; index += thread_groups) {
    const auto vector = group_lane + (index * thread_group_size);
    query_vectors[group_lane][index] = *reinterpret_cast<const KeyVector *>(query_row + (vector * key_vector_size));
  }
  __syncthreads();

  // Softmax logits and the cross-warp PV reduction reuse the same shared-memory region.
  extern __shared__ float attention_shared[];
  auto *logits = attention_shared;
  __shared__ float reductions[2 * DECODE_WARPS];
  const auto *block_table = block_tables + (sequence * max_blocks);
  float maximum = -FLT_MAX;
  for (auto page = first_block + warp; page < last_block; page += DECODE_WARPS) {
    const auto physical = static_cast<int64_t>(block_table[page]);
    const auto offset = group % block_size;
    const auto token = (page * block_size) + offset;
    const auto *key_row =
        key_cache + (physical * kv_block_stride) + (kv_head * kv_head_stride) + (offset * key_packing);
    KeyVector key_vectors[key_vectors_per_thread];
#pragma unroll
    for (int index = 0; index < key_vectors_per_thread; ++index) {
      const auto dimension = (group_lane + (index * thread_group_size)) * key_vector_size;
      const auto address = ((dimension / key_packing) * block_size * key_packing) + (dimension % key_packing);
      key_vectors[index] = *reinterpret_cast<const KeyVector *>(key_row + address);
    }
    const auto score = softmax_scale * QkDot<thread_group_size>(query_vectors[group_lane], key_vectors);
    if (group_lane == 0) {
      // The entire final page is loaded; unused token probabilities must remain zero for PV.
      const auto padding = token >= length;
      logits[token - first_token] = padding ? 0.0F : score;
      maximum = padding ? maximum : fmaxf(maximum, score);
    }
  }
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= thread_group_size; mask /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, mask));
  }
  if (lane == 0) {
    reductions[warp] = maximum;
  }
  __syncthreads();
  maximum = lane < DECODE_WARPS ? reductions[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = DECODE_WARPS / 2; mask >= 1; mask /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, mask));
  }
  maximum = __shfl_sync(0xffffffffU, maximum, 0);

  float sum = 0.0F;
  for (auto token = thread; token < tokens; token += DECODE_THREADS) {
    const auto weight = __expf(logits[token] - maximum);
    logits[token] = weight;
    sum += weight;
  }
  sum = BlockSum(sum, reductions + DECODE_WARPS);
  const auto inverse_sum = __fdividef(1.0F, sum + SOFTMAX_EPSILON);
  for (auto token = thread; token < tokens; token += DECODE_THREADS) {
    logits[token] *= inverse_sum;
  }
  __syncthreads();
  const auto task = (((sequence * heads) + head) * partitions) + partition;
  if (thread == 0) {
    max_logits[task] = maximum;
    exp_sums[task] = sum;
  }

  // Probabilities use the model dtype for vector PV; vector results accumulate in FP32.
  constexpr int value_vector_size = 16 / sizeof(T);
  constexpr int value_group_size = block_size / value_vector_size;
  constexpr int value_rows_per_warp = WARP_SIZE / value_group_size;
  constexpr int values_per_thread = (head_size + value_rows_per_warp - 1) / value_rows_per_warp;
  using ValueVector = CacheVector<T, value_vector_size>;
  float accumulators[values_per_thread]{};
  for (auto page = first_block + warp; page < last_block; page += DECODE_WARPS) {
    const auto physical = static_cast<int64_t>(block_table[page]);
    const auto offset = (lane % value_group_size) * value_vector_size;
    const auto token = (page * block_size) + offset;
    ValueVector probabilities;
#pragma unroll
    for (int element = 0; element < value_vector_size; ++element) {
      probabilities.values_[element] = ttl::FromElementwiseFloat<T>(logits[token - first_token + element]);
    }
    const auto *value_row = value_cache + (physical * kv_block_stride) + (kv_head * kv_head_stride);
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
      const auto dimension = (lane / value_group_size) + (index * value_rows_per_warp);
      if (dimension < head_size) {
        auto values = *reinterpret_cast<const ValueVector *>(value_row + (dimension * block_size) + offset);
        if (page == context_blocks - 1) {
#pragma unroll
          for (int element = 0; element < value_vector_size; ++element) {
            // Padding cache contents may contain NaNs; zero the values before multiplying by zero probability.
            values.values_[element] =
                token + element < length ? values.values_[element] : ttl::FromElementwiseFloat<T>(0.0F);
          }
        }
        accumulators[index] += ValueDot(probabilities, values);
      }
    }
  }
#pragma unroll
  for (int index = 0; index < values_per_thread; ++index) {
#pragma unroll
    for (int mask = value_group_size / 2; mask >= 1; mask /= 2) {
      accumulators[index] += __shfl_xor_sync(0xffffffffU, accumulators[index], mask);
    }
  }
  __syncthreads();

  // Pair upper and lower warps repeatedly, preserving the order of the partial-output additions.
#pragma unroll
  for (int warps = DECODE_WARPS; warps > 1; warps /= 2) {
    const auto half = warps / 2;
    if (warp >= half && warp < warps) {
#pragma unroll
      for (int index = 0; index < values_per_thread; ++index) {
        const auto dimension = (lane / value_group_size) + (index * value_rows_per_warp);
        if (dimension < head_size && lane % value_group_size == 0) {
          attention_shared[((warp - half) * head_size) + dimension] = accumulators[index];
        }
      }
    }
    __syncthreads();
    if (warp < half) {
#pragma unroll
      for (int index = 0; index < values_per_thread; ++index) {
        const auto dimension = (lane / value_group_size) + (index * value_rows_per_warp);
        if (dimension < head_size && lane % value_group_size == 0) {
          accumulators[index] += attention_shared[(warp * head_size) + dimension];
        }
      }
    }
    __syncthreads();
  }
  if (warp == 0) {
#pragma unroll
    for (int index = 0; index < values_per_thread; ++index) {
      const auto dimension = (lane / value_group_size) + (index * value_rows_per_warp);
      if (dimension < head_size && lane % value_group_size == 0) {
        partial_output[(task * head_size) + dimension] = ttl::FromElementwiseFloat<T>(accumulators[index]);
      }
    }
  }
}

/** Merge partition outputs using their maxima and unnormalized probability sums. */
template <ttl::CudaStorageType T, int head_size>
__global__ void ReducePartitionsKernel(const T *__restrict__ partial_output, const float *__restrict__ exp_sums,
                                       const float *__restrict__ max_logits,
                                       const int32_t *__restrict__ context_lengths, T *__restrict__ output,
                                       int max_partitions) {
  const auto head = static_cast<int>(blockIdx.x);
  const auto heads = static_cast<int>(gridDim.x);
  const auto sequence = static_cast<int>(blockIdx.y);
  const auto thread = static_cast<int>(threadIdx.x);
  const auto lane = thread % WARP_SIZE;
  const auto warp = thread / WARP_SIZE;
  const auto length = context_lengths[sequence];
  const auto partitions = (length + PARTITION_SIZE - 1) / PARTITION_SIZE;
  const auto row = (sequence * heads) + head;
  const auto stats_offset = row * max_partitions;
  const auto *partials = partial_output + (stats_offset * head_size);
  auto *output_row = output + (row * head_size);
  if (partitions == 1) {
    for (auto dimension = thread; dimension < head_size; dimension += DECODE_THREADS) {
      output_row[dimension] = partials[dimension];
    }
    return;
  }

  extern __shared__ float partition_shared[];
  auto *partition_maxima = partition_shared;
  auto *partition_sums = partition_shared + partitions;
  __shared__ float reductions[2 * DECODE_WARPS];
  float maximum = -FLT_MAX;
  for (auto partition = thread; partition < partitions; partition += DECODE_THREADS) {
    const auto value = max_logits[stats_offset + partition];
    partition_maxima[partition] = value;
    maximum = fmaxf(maximum, value);
  }
  __syncthreads();
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, mask));
  }
  if (lane == 0) {
    reductions[warp] = maximum;
  }
  __syncthreads();
  maximum = lane < DECODE_WARPS ? reductions[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = DECODE_WARPS / 2; mask >= 1; mask /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, mask));
  }
  maximum = __shfl_sync(0xffffffffU, maximum, 0);
  float sum = 0.0F;
  for (auto partition = thread; partition < partitions; partition += DECODE_THREADS) {
    const auto weight = exp_sums[stats_offset + partition] * expf(partition_maxima[partition] - maximum);
    partition_sums[partition] = weight;
    sum += weight;
  }
  __syncthreads();
  sum = BlockSum(sum, reductions + DECODE_WARPS);
  const auto inverse_sum = __fdividef(1.0F, sum + SOFTMAX_EPSILON);
#pragma unroll
  for (auto dimension = thread; dimension < head_size; dimension += DECODE_THREADS) {
    float accumulator = 0.0F;
    for (int partition = 0; partition < partitions; ++partition) {
      accumulator += ttl::ToElementwiseFloat(partials[(partition * head_size) + dimension]) *
                     partition_sums[partition] * inverse_sum;
    }
    output_row[dimension] = ttl::FromElementwiseFloat<T>(accumulator);
  }
}

/** Keep capability queries and CUDA instantiation on the same specialization list. */
template <typename Callback>
auto DispatchPagedDecode(int64_t head_dim, int64_t page_size, const Callback &callback) -> bool {
  const auto dispatch_head = [&]<int block_size>() {
    switch (head_dim) {
      case 32:
        callback.template operator()<32, block_size>();
        break;
      case 64:
        callback.template operator()<64, block_size>();
        break;
      case 80:
        callback.template operator()<80, block_size>();
        break;
      case 96:
        callback.template operator()<96, block_size>();
        break;
      case 112:
        callback.template operator()<112, block_size>();
        break;
      case 120:
        callback.template operator()<120, block_size>();
        break;
      case 128:
        callback.template operator()<128, block_size>();
        break;
      case 192:
        callback.template operator()<192, block_size>();
        break;
      case 256:
        callback.template operator()<256, block_size>();
        break;
      case 512:
        callback.template operator()<512, block_size>();
        break;
      default:
        return false;
    }
    return true;
  };
  switch (page_size) {
    case 8:
      return dispatch_head.template operator()<8>();
    case 16:
      return dispatch_head.template operator()<16>();
    case 32:
      return dispatch_head.template operator()<32>();
    default:
      return false;
  }
}

auto SupportsPagedDecode(int64_t head_dim, int64_t block_size) noexcept -> bool {
  return DispatchPagedDecode(head_dim, block_size, [&]<int, int>() {});
}

void LaunchPagedDecode(cudaStream_t stream, ttl::DType dtype, const void *query, const void *key_cache,
                       const void *value_cache, const int32_t *block_tables, const int32_t *context_lengths,
                       void *output, int64_t batch_size, int64_t query_heads, int64_t max_blocks, int64_t query_stride,
                       float softmax_scale, const CacheShape &cache_shape, int64_t max_context_len, void *workspace,
                       const ttl::CudaDeviceErrorContext &error_context) {
  const auto num_partitions = std::max<int64_t>(1, (max_context_len + PARTITION_SIZE - 1) / PARTITION_SIZE);
  const auto tasks = batch_size * query_heads * num_partitions;
  auto *exp_sums = static_cast<float *>(workspace);
  auto *max_logits = exp_sums + tasks;
  auto *checked_lengths = reinterpret_cast<int32_t *>(max_logits + tasks);
  auto *partial_output = checked_lengths + batch_size;
  const dim3 grid(static_cast<unsigned int>(query_heads), static_cast<unsigned int>(batch_size),
                  static_cast<unsigned int>(num_partitions));
  const dim3 reduce_grid(static_cast<unsigned int>(query_heads), static_cast<unsigned int>(batch_size));
  const auto kv_head_stride = static_cast<int>(cache_shape.key_head_dim_ * cache_shape.block_size_);
  const auto kv_block_stride = static_cast<int>(cache_shape.num_kv_heads_ * kv_head_stride);
  const auto shared_bytes =
      std::max(PARTITION_SIZE, (DECODE_WARPS / 2) * static_cast<int>(cache_shape.key_head_dim_)) * sizeof(float);
  const auto reduce_shared_bytes = 2 * static_cast<size_t>(num_partitions) * sizeof(float);

  PrepareDecodeLengths<<<static_cast<unsigned int>(batch_size), DECODE_THREADS, 0, stream>>>(
      block_tables, context_lengths, checked_lengths, static_cast<int>(max_blocks),
      static_cast<int>(cache_shape.block_size_), cache_shape.num_blocks_, static_cast<int>(max_context_len),
      error_context);
  ttl::DispatchCudaFloatingDType(dtype, "PagedDecode", [&]<ttl::CudaStorageType T>(std::type_identity<T>) {
    const auto launch = [&]<int head_size, int block_size>() {
      PagedDecodeKernel<T, head_size, block_size><<<grid, DECODE_THREADS, shared_bytes, stream>>>(
          static_cast<const T *>(query), static_cast<const T *>(key_cache), static_cast<const T *>(value_cache),
          block_tables, checked_lengths, reinterpret_cast<T *>(partial_output), exp_sums, max_logits,
          static_cast<int>(cache_shape.num_kv_heads_), static_cast<int>(max_blocks), static_cast<int>(query_stride),
          kv_block_stride, kv_head_stride, softmax_scale);
      ReducePartitionsKernel<T, head_size><<<reduce_grid, DECODE_THREADS, reduce_shared_bytes, stream>>>(
          reinterpret_cast<const T *>(partial_output), exp_sums, max_logits, checked_lengths, static_cast<T *>(output),
          static_cast<int>(num_partitions));
    };
    if (!DispatchPagedDecode(cache_shape.key_head_dim_, cache_shape.block_size_, launch)) {
      throw InvalidArgumentException("unsupported paged decode head dimension or page size");
    }
  });
}

}  // namespace zephyr::attention
