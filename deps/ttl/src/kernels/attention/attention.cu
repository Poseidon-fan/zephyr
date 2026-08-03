#include "ttl/internal/ops/attention.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_math.cuh"
#include "ttl/internal/runtime/library/cuda_dtype.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl::internal {
namespace {

constexpr uint32_t MAXIMUM_SDPA_BLOCKS = 65535;

template <CudaStorageType T>
__device__ auto LoadFloat(const std::byte *pointer) -> float {
  return ToElementwiseFloat(*reinterpret_cast<const T *>(pointer));
}

__device__ auto LoadMaskFloat(const SdpaParameters &parameters, uint64_t offset) -> float {
  const auto *pointer = parameters.mask_ + offset;
  switch (parameters.mask_dtype_) {
    case DType::FLOAT16:
      return LoadFloat<__half>(pointer);
    case DType::BFLOAT16:
      return LoadFloat<__nv_bfloat16>(pointer);
    case DType::FLOAT32:
      return LoadFloat<float>(pointer);
    case DType::BOOL:
    case DType::UINT8:
    case DType::INT32:
    case DType::INT64:
      break;
  }
  return CUDART_NAN_F;
}

__device__ auto IsCausallyAllowed(const SdpaParameters &parameters, uint64_t query_index, uint64_t key_index) noexcept
    -> bool {
  if (!parameters.causal_) {
    return true;
  }
  auto boundary = static_cast<int64_t>(query_index);
  if (parameters.lower_right_) {
    boundary += static_cast<int64_t>(parameters.key_length_) - static_cast<int64_t>(parameters.query_length_);
  }
  return boundary >= 0 && key_index <= static_cast<uint64_t>(boundary);
}

template <CudaStorageType T, uint32_t block_size>
__global__ void SdpaKernel(SdpaParameters parameters) {
  __shared__ float reduction[block_size];
  __shared__ float old_scale;
  __shared__ float current_scale;
  __shared__ float normalizer;
  __shared__ float maximum;
  __shared__ bool process_score;

  const auto value_tiles = (parameters.value_dimension_ + block_size - 1U) / block_size;
  const auto row_count = parameters.batch_size_ * parameters.query_head_count_ * parameters.query_length_;
  const auto task_count = row_count * value_tiles;
  for (auto task = static_cast<uint64_t>(blockIdx.x); task < task_count; task += gridDim.x) {
    const auto row = task / value_tiles;
    const auto value_tile = task % value_tiles;
    const auto query_index = row % parameters.query_length_;
    const auto head_linear = row / parameters.query_length_;
    const auto query_head = head_linear % parameters.query_head_count_;
    const auto batch = head_linear / parameters.query_head_count_;
    const auto heads_per_key_value = parameters.query_head_count_ / parameters.key_value_head_count_;
    const auto key_value_head = query_head / heads_per_key_value;
    const auto value_index = (value_tile * block_size) + threadIdx.x;

    const auto query_base = (batch * parameters.query_strides_bytes_[0]) +
                            (query_head * parameters.query_strides_bytes_[1]) +
                            (query_index * parameters.query_strides_bytes_[2]);
    auto accumulator = 0.0F;
    if (threadIdx.x == 0) {
      normalizer = 0.0F;
      maximum = -CUDART_INF_F;
    }
    __syncthreads();

    for (uint64_t key_index = 0; key_index < parameters.key_length_; ++key_index) {
      auto allowed = IsCausallyAllowed(parameters, query_index, key_index);
      auto additive_mask = 0.0F;
      if (parameters.has_mask_) {
        const auto mask_offset =
            (batch * parameters.mask_strides_bytes_[0]) + (query_head * parameters.mask_strides_bytes_[1]) +
            (query_index * parameters.mask_strides_bytes_[2]) + (key_index * parameters.mask_strides_bytes_[3]);
        if (parameters.mask_dtype_ == DType::BOOL) {
          allowed = allowed && parameters.mask_[mask_offset] != std::byte{0};
        } else {
          additive_mask = LoadMaskFloat(parameters, mask_offset);
        }
      }

      auto partial = 0.0F;
      if (allowed) {
        const auto key_base = (batch * parameters.key_strides_bytes_[0]) +
                              (key_value_head * parameters.key_strides_bytes_[1]) +
                              (key_index * parameters.key_strides_bytes_[2]);
        for (auto dimension = static_cast<uint64_t>(threadIdx.x); dimension < parameters.head_dimension_;
             dimension += block_size) {
          partial += LoadFloat<T>(parameters.query_ + query_base + (dimension * parameters.query_strides_bytes_[3])) *
                     LoadFloat<T>(parameters.key_ + key_base + (dimension * parameters.key_strides_bytes_[3]));
        }
      }
      reduction[threadIdx.x] = partial;
      __syncthreads();
      for (uint32_t width = block_size / 2; width > 0; width >>= 1U) {
        if (threadIdx.x < width) {
          reduction[threadIdx.x] += reduction[threadIdx.x + width];
        }
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        if (allowed) {
          const auto score = (reduction[0] * parameters.scale_) + additive_mask;
          process_score = score != -CUDART_INF_F;
          if (process_score) {
            const auto new_maximum = isnan(score) || isnan(maximum) ? CUDART_NAN_F : fmaxf(maximum, score);
            old_scale = maximum == -CUDART_INF_F ? 0.0F : expf(maximum - new_maximum);
            current_scale = expf(score - new_maximum);
            normalizer = (normalizer * old_scale) + current_scale;
            maximum = new_maximum;
          } else {
            old_scale = 1.0F;
            current_scale = 0.0F;
          }
        } else {
          process_score = false;
          old_scale = 1.0F;
          current_scale = 0.0F;
        }
      }
      __syncthreads();

      if (value_index < parameters.value_dimension_) {
        auto value_element = 0.0F;
        if (process_score) {
          const auto value_offset =
              (batch * parameters.value_strides_bytes_[0]) + (key_value_head * parameters.value_strides_bytes_[1]) +
              (key_index * parameters.value_strides_bytes_[2]) + (value_index * parameters.value_strides_bytes_[3]);
          value_element = LoadFloat<T>(parameters.value_ + value_offset);
        }
        accumulator = (accumulator * old_scale) + (current_scale * value_element);
      }
      __syncthreads();
    }

    if (value_index < parameters.value_dimension_) {
      const auto output_offset =
          (batch * parameters.output_strides_bytes_[0]) + (query_head * parameters.output_strides_bytes_[1]) +
          (query_index * parameters.output_strides_bytes_[2]) + (value_index * parameters.output_strides_bytes_[3]);
      const auto result = normalizer == 0.0F ? 0.0F : accumulator / normalizer;
      *reinterpret_cast<T *>(parameters.output_ + output_offset) = FromElementwiseFloat<T>(result);
    }
    __syncthreads();
  }
}

template <CudaStorageType T, uint32_t block_size>
void LaunchTyped(cudaStream_t stream, const SdpaParameters &parameters) {
  const auto value_tiles = (parameters.value_dimension_ + block_size - 1U) / block_size;
  const auto tasks = parameters.batch_size_ * parameters.query_head_count_ * parameters.query_length_ * value_tiles;
  const auto blocks = static_cast<uint32_t>(tasks < MAXIMUM_SDPA_BLOCKS ? tasks : MAXIMUM_SDPA_BLOCKS);
  SdpaKernel<T, block_size><<<blocks, block_size, 0, stream>>>(parameters);
}

}  // namespace

void LaunchSdpa(cudaStream_t stream, DType dtype, const SdpaParameters &parameters, std::source_location location) {
  if (stream == nullptr || parameters.output_ == nullptr || parameters.query_ == nullptr ||
      parameters.query_head_count_ == 0 || parameters.key_value_head_count_ == 0 || parameters.query_length_ == 0 ||
      parameters.value_dimension_ == 0) {
    throw InternalError("invalid SDPA launch parameters", location);
  }
  DispatchCudaFloatingDType(
      dtype, "ScaledDotProductAttentionOut",
      [&]<CudaStorageType T>(std::type_identity<T>) {
        if (parameters.head_dimension_ <= 128 && parameters.value_dimension_ <= 128) {
          LaunchTyped<T, 128>(stream, parameters);
        } else {
          LaunchTyped<T, 256>(stream, parameters);
        }
      },
      location);
}

}  // namespace ttl::internal
