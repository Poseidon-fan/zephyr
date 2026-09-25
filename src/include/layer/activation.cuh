#pragma once

#include <cuda_runtime_api.h>

#include <ttl/tensor/dtype.hpp>
#include <ttl/tensor/shape.hpp>

namespace zephyr::layer {

/** Evaluate x / (1 + exp(-x)) in FP32, then cast to the output dtype. */
void LaunchSilu(cudaStream_t stream, ttl::DType dtype, const void *input, void *output, const ttl::Shape &shape,
                const ttl::Strides &strides);

}  // namespace zephyr::layer
