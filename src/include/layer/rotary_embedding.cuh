#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include <ttl/runtime/device_error.hpp>
#include <ttl/tensor/dtype.hpp>

namespace zephyr::layer {

void LaunchRotaryEmbedding(cudaStream_t stream, ttl::DType dtype, void *input, const void *cos, const void *sin,
                           const void *positions, ttl::DType position_dtype, int64_t num_tokens, int64_t num_heads,
                           int64_t head_dim, int64_t rotary_half_dim, int64_t max_position_embeddings, bool is_gpt_neox,
                           const ttl::CudaDeviceErrorContext &error_context);

}  // namespace zephyr::layer
