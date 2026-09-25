// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Eric Buehler
// The MIT permission and warranty notice is provided in LICENSE.

#include "layer/rotary_embedding.cuh"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

#include <ttl/runtime/cuda_dtype.hpp>
#include <ttl/runtime/device_error.cuh>

namespace zephyr::layer {

/** One block rotates one token, preserving the storage dtype at every scalar operation. */
template <typename T, typename Index, bool IS_NEOX>
__global__ void RotaryEmbeddingKernel(T *input, const T *cos_cache, const T *sin_cache, const Index *positions,
                                      int64_t num_heads, int64_t head_dim, int64_t rotary_half_dim,
                                      int64_t max_position_embeddings, ttl::CudaDeviceErrorContext error_context) {
  const auto token = static_cast<int64_t>(blockIdx.x);
  const auto position = static_cast<int64_t>(positions[token]);
  if (position < 0 || position >= max_position_embeddings) {
    if (threadIdx.x == 0) {
      ttl::ReportCudaDeviceError(error_context, ttl::CudaDeviceErrorCode::INDEX_OUT_OF_BOUNDS, token,
                                 static_cast<uint64_t>(position), max_position_embeddings);
    }
    return;
  }

  const auto *cos = cos_cache + (position * rotary_half_dim);
  const auto *sin = sin_cache + (position * rotary_half_dim);
  const auto pairs = num_heads * rotary_half_dim;
  for (auto index = static_cast<int64_t>(threadIdx.x); index < pairs; index += blockDim.x) {
    const auto head = index / rotary_half_dim;
    const auto offset = index % rotary_half_dim;
    auto *row = input + (((token * num_heads) + head) * head_dim);
    const auto x_index = IS_NEOX ? offset : 2 * offset;
    const auto y_index = IS_NEOX ? rotary_half_dim + offset : (2 * offset) + 1;
    const T cosine = __ldg(cos + offset);
    const T sine = __ldg(sin + offset);
    const T x = row[x_index];
    const T y = row[y_index];
    row[x_index] = (x * cosine) - (y * sine);
    row[y_index] = (y * cosine) + (x * sine);
  }
}

void LaunchRotaryEmbedding(cudaStream_t stream, ttl::DType dtype, void *input, const void *cos, const void *sin,
                           const void *positions, ttl::DType position_dtype, int64_t num_tokens, int64_t num_heads,
                           int64_t head_dim, int64_t rotary_half_dim, int64_t max_position_embeddings, bool is_gpt_neox,
                           const ttl::CudaDeviceErrorContext &error_context) {
  if (num_tokens == 0 || num_heads == 0) {
    return;
  }
  const dim3 grid(static_cast<uint32_t>(num_tokens));
  const dim3 block(static_cast<uint32_t>(std::min<int64_t>(num_heads * rotary_half_dim, 512)));
  ttl::DispatchCudaFloatingDType(dtype, "RotaryEmbedding", [&]<typename T>(std::type_identity<T>) {
    const auto launch = [&]<typename Index>() {
      if (is_gpt_neox) {
        RotaryEmbeddingKernel<T, Index, true>
            <<<grid, block, 0, stream>>>(static_cast<T *>(input), static_cast<const T *>(cos),
                                         static_cast<const T *>(sin), static_cast<const Index *>(positions), num_heads,
                                         head_dim, rotary_half_dim, max_position_embeddings, error_context);
      } else {
        RotaryEmbeddingKernel<T, Index, false>
            <<<grid, block, 0, stream>>>(static_cast<T *>(input), static_cast<const T *>(cos),
                                         static_cast<const T *>(sin), static_cast<const Index *>(positions), num_heads,
                                         head_dim, rotary_half_dim, max_position_embeddings, error_context);
      }
    };
    if (position_dtype == ttl::DType::INT32) {
      launch.template operator()<int32_t>();
    } else {
      launch.template operator()<int64_t>();
    }
  });
}

}  // namespace zephyr::layer
