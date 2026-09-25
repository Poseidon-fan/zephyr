#include "attention/cache.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

#include <ttl/runtime/kernel_launch.hpp>

#include "attention/kernels.cuh"
#include "common/exception.hpp"

namespace zephyr::attention {

auto GetCacheShape(const ttl::Tensor &key_cache, const ttl::Tensor &value_cache) -> CacheShape {
  if (!key_cache.IsContiguous() || !value_cache.IsContiguous() ||
      ttl::ClassifyAlias(key_cache, value_cache) != ttl::AliasKind::DISJOINT) {
    throw InvalidArgumentException("paged KV caches must be contiguous and disjoint");
  }

  const auto dtype = key_cache.GetDType();
  if ((dtype != ttl::DType::FLOAT16 && dtype != ttl::DType::BFLOAT16 && dtype != ttl::DType::FLOAT32) ||
      value_cache.GetDType() != dtype || key_cache.GetRank() != 5 || value_cache.GetRank() != 4) {
    throw InvalidArgumentException("paged KV caches must use one supported floating-point dtype and shape");
  }

  const auto key_shape = key_cache.GetShape();
  const auto value_shape = value_cache.GetShape();
  const auto key_blocks = key_shape.GetDimension(0);
  const auto key_heads = key_shape.GetDimension(1);
  const auto packed_key_dim = key_shape.GetDimension(2);
  const auto block_size = key_shape.GetDimension(3);
  const auto key_packing = key_shape.GetDimension(4);
  const auto value_blocks = value_shape.GetDimension(0);
  const auto value_heads = value_shape.GetDimension(1);
  const auto value_dim = value_shape.GetDimension(2);
  const auto value_block_size = value_shape.GetDimension(3);
  const auto expected_key_packing = static_cast<int64_t>(16 / ttl::GetDTypeInfo(dtype).size_bytes_);

  if (key_blocks <= 0 || key_heads <= 0 || packed_key_dim <= 0 || block_size <= 0 || key_packing <= 0 ||
      value_dim <= 0 || value_blocks != key_blocks || value_heads != key_heads || value_block_size != block_size ||
      key_packing != expected_key_packing) {
    throw InvalidArgumentException("paged KV cache dimensions do not match the standard page shape");
  }

  return CacheShape{.num_blocks_ = key_blocks,
                    .num_kv_heads_ = key_heads,
                    .key_head_dim_ = packed_key_dim * key_packing,
                    .value_head_dim_ = value_dim,
                    .block_size_ = block_size,
                    .key_packing_ = key_packing,
                    .dtype_ = dtype};
}

void ReshapeAndCache(ttl::ExecutionContext &context, const ttl::Tensor &key, const ttl::Tensor &value,
                     ttl::Tensor &key_cache, ttl::Tensor &value_cache, const ttl::Tensor &slot_mapping) {
  const auto cache_shape = GetCacheShape(key_cache, value_cache);
  if (slot_mapping.GetRank() != 1) {
    throw InvalidArgumentException("slot mapping must have rank one");
  }

  const auto key_shape = key.GetShape();
  const auto value_shape = value.GetShape();
  const auto slot_shape = slot_mapping.GetShape();
  const auto get_input_layout = [](const ttl::Tensor &input, const char *name) -> std::array<int64_t, 2> {
    const auto rank = input.GetRank();
    if (rank != 3 && rank != 4) {
      throw InvalidArgumentException(std::string{name} + " must have rank three or four");
    }
    const auto shape = input.GetShape();
    const auto head_dim = shape.GetDimension(rank - 1);
    const auto num_heads = shape.GetDimension(rank - 2);
    const auto strides = input.GetStrides();
    if (head_dim <= 0 || num_heads <= 0 || strides.GetStride(rank - 1) != 1 ||
        (num_heads > 1 && strides.GetStride(rank - 2) != head_dim) ||
        num_heads > std::numeric_limits<int64_t>::max() / head_dim) {
      throw InvalidArgumentException(std::string{name} + " must have dense head dimensions");
    }

    int64_t num_tokens;
    int64_t row_stride;
    if (rank == 3) {
      num_tokens = shape.GetDimension(0);
      row_stride = strides.GetStride(0);
    } else {
      const auto batch_size = shape.GetDimension(0);
      const auto sequence_length = shape.GetDimension(1);
      if (batch_size > 0 && sequence_length > std::numeric_limits<int64_t>::max() / batch_size) {
        throw InvalidArgumentException(std::string{name} + " token count overflows the kernel index range");
      }
      num_tokens = batch_size * sequence_length;
      row_stride = sequence_length == 1 ? strides.GetStride(0) : strides.GetStride(1);
      if (batch_size > 1 && sequence_length > 1 &&
          (row_stride > std::numeric_limits<int64_t>::max() / sequence_length ||
           strides.GetStride(0) != sequence_length * row_stride)) {
        throw InvalidArgumentException(std::string{name} + " batch and sequence rows are not uniformly strided");
      }
    }
    if (row_stride < num_heads * head_dim) {
      throw InvalidArgumentException(std::string{name} + " rows are smaller than their dense head data");
    }
    return {num_tokens, row_stride};
  };
  const auto key_layout = get_input_layout(key, "key");
  const auto value_layout = get_input_layout(value, "value");
  if (!slot_mapping.IsContiguous() || key.GetDType() != cache_shape.dtype_ || value.GetDType() != cache_shape.dtype_ ||
      slot_mapping.GetDType() != ttl::DType::INT64 || key_layout[0] != value_layout[0] ||
      key_layout[0] != slot_shape.GetDimension(0) ||
      key_shape.GetDimension(key.GetRank() - 2) != cache_shape.num_kv_heads_ ||
      value_shape.GetDimension(value.GetRank() - 2) != cache_shape.num_kv_heads_ ||
      key_shape.GetDimension(key.GetRank() - 1) != cache_shape.key_head_dim_ ||
      value_shape.GetDimension(value.GetRank() - 1) != cache_shape.value_head_dim_) {
    throw InvalidArgumentException("K/V rows do not match the paged cache");
  }

  const std::array<ttl::Tensor, 3> inputs{key, value, slot_mapping};
  for (const auto &input : inputs) {
    if (ttl::ClassifyAlias(key_cache, input) != ttl::AliasKind::DISJOINT ||
        ttl::ClassifyAlias(value_cache, input) != ttl::AliasKind::DISJOINT) {
      throw InvalidArgumentException("cache writes must not alias their inputs");
    }
  }
  const std::array<ttl::Tensor *, 2> outputs{&key_cache, &value_cache};
  ttl::SubmitCudaKernel(
      context, "ReshapeAndCache", std::span<const ttl::Tensor>{inputs}, std::span<ttl::Tensor *const>{outputs},
      [&](ttl::CudaKernelLaunch &launch) {
        LaunchReshapeAndCache(launch.GetStream(), cache_shape.dtype_, launch.GetInputData(inputs[0]),
                              launch.GetInputData(inputs[1]), launch.GetOutputData(key_cache),
                              launch.GetOutputData(value_cache), launch.GetInputDataAs<int64_t>(inputs[2]),
                              key_layout[0], key_layout[1], value_layout[1], cache_shape,
                              launch.GetDeviceErrorContext(ttl::DType::INT64));
      },
      ttl::CudaKernelLaunchOptions{.capture_policy_ = ttl::CudaCapturePolicy::SAFE});
}

void GatherKvCache(ttl::ExecutionContext &context, ttl::Tensor &key_output, ttl::Tensor &value_output,
                   const ttl::Tensor &key_cache, const ttl::Tensor &value_cache, const ttl::Tensor &block_tables,
                   const ttl::Tensor &cu_seqlens_k) {
  const auto cache_shape = GetCacheShape(key_cache, value_cache);
  if (block_tables.GetRank() != 2 || cu_seqlens_k.GetRank() != 1 || key_output.GetRank() != 3 ||
      value_output.GetRank() != 3) {
    throw InvalidArgumentException("gathered KV tensors have invalid ranks");
  }

  const auto block_shape = block_tables.GetShape();
  const auto key_shape = key_output.GetShape();
  const auto value_shape = value_output.GetShape();
  const auto cu_shape = cu_seqlens_k.GetShape();
  const auto batch_size = block_shape.GetDimension(0);
  if (batch_size <= 0 || !block_tables.IsContiguous() || block_tables.GetDType() != ttl::DType::INT32 ||
      !cu_seqlens_k.IsContiguous() || cu_seqlens_k.GetDType() != ttl::DType::INT32 ||
      block_shape.GetDimension(1) <= 0 || cu_shape.GetDimension(0) != batch_size + 1 || !key_output.IsContiguous() ||
      !value_output.IsContiguous() || key_output.GetDType() != cache_shape.dtype_ ||
      value_output.GetDType() != cache_shape.dtype_ || key_shape.GetDimension(0) != value_shape.GetDimension(0) ||
      key_shape.GetDimension(1) != cache_shape.num_kv_heads_ ||
      value_shape.GetDimension(1) != cache_shape.num_kv_heads_ ||
      key_shape.GetDimension(2) != cache_shape.key_head_dim_ ||
      value_shape.GetDimension(2) != cache_shape.value_head_dim_) {
    throw InvalidArgumentException("gathered KV tensors do not match the paged cache");
  }

  if (ttl::ClassifyAlias(key_output, value_output) != ttl::AliasKind::DISJOINT) {
    throw InvalidArgumentException("gathered K and V outputs must be disjoint");
  }
  const std::array<ttl::Tensor, 4> inputs{key_cache, value_cache, block_tables, cu_seqlens_k};
  for (const auto &input : inputs) {
    if (ttl::ClassifyAlias(key_output, input) != ttl::AliasKind::DISJOINT ||
        ttl::ClassifyAlias(value_output, input) != ttl::AliasKind::DISJOINT) {
      throw InvalidArgumentException("gathered KV outputs must not alias their inputs");
    }
  }
  const std::array<ttl::Tensor *, 2> outputs{&key_output, &value_output};
  ttl::SubmitCudaKernel(
      context, "GatherKvCache", std::span<const ttl::Tensor>{inputs}, std::span<ttl::Tensor *const>{outputs},
      [&](ttl::CudaKernelLaunch &launch) {
        LaunchGatherKvCache(launch.GetStream(), cache_shape.dtype_, launch.GetInputData(inputs[0]),
                            launch.GetInputData(inputs[1]), launch.GetInputDataAs<int32_t>(inputs[2]),
                            launch.GetInputDataAs<int32_t>(inputs[3]), launch.GetOutputData(key_output),
                            launch.GetOutputData(value_output), key_shape.GetDimension(0), batch_size,
                            block_shape.GetDimension(1), cache_shape, launch.GetDeviceErrorContext(ttl::DType::INT32));
      },
      ttl::CudaKernelLaunchOptions{.capture_policy_ = ttl::CudaCapturePolicy::SAFE});
}

}  // namespace zephyr::attention
