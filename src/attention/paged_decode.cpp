#include "attention/paged_decode.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#include <ttl/runtime/cuda_dtype.hpp>
#include <ttl/runtime/kernel_launch.hpp>

#include "attention/kernels.cuh"
#include "common/exception.hpp"

namespace zephyr::attention {

void PagedDecode(ttl::ExecutionContext &context, ttl::Tensor &output, const ttl::Tensor &query,
                 const ttl::Tensor &key_cache, const ttl::Tensor &value_cache, const ttl::Tensor &block_tables,
                 const ttl::Tensor &context_lens, int64_t max_context_len, const AttentionParams &params) {
  const auto cache_shape = GetCacheShape(key_cache, value_cache);
  const auto head_dim = cache_shape.key_head_dim_;
  const auto block_size = cache_shape.block_size_;
  constexpr std::array head_sizes{32, 64, 80, 96, 112, 120, 128, 192, 256};
  if (std::ranges::find(head_sizes, head_dim) == head_sizes.end() || head_dim != cache_shape.value_head_dim_ ||
      (block_size != 8 && block_size != 16 && block_size != 32)) {
    throw InvalidArgumentException("paged decode does not support the cache head dimension or page size");
  }
  if (block_tables.GetRank() != 2 || context_lens.GetRank() != 1 || query.GetRank() != 3 || output.GetRank() != 3) {
    throw InvalidArgumentException("paged decode tensors have invalid ranks");
  }

  const auto query_shape = query.GetShape();
  const auto output_shape = output.GetShape();
  const auto table_shape = block_tables.GetShape();
  const auto length_shape = context_lens.GetShape();
  const auto batch_size = table_shape.GetDimension(0);
  const auto query_heads = query_shape.GetDimension(1);
  const auto query_stride = batch_size == 1 ? 0 : query.GetStrides().GetStride(0);
  const auto max_blocks = table_shape.GetDimension(1);
  if (!block_tables.IsContiguous() || block_tables.GetDType() != ttl::DType::INT32 || !context_lens.IsContiguous() ||
      context_lens.GetDType() != ttl::DType::INT32 || batch_size <= 0 || max_blocks <= 0 || max_context_len < 0 ||
      length_shape.GetDimension(0) != batch_size || !output.IsContiguous() || query.GetDType() != cache_shape.dtype_ ||
      output.GetDType() != cache_shape.dtype_ || query_shape.GetDimension(0) != batch_size ||
      output_shape.GetDimension(0) != batch_size || query_heads <= 0 || query_heads % cache_shape.num_kv_heads_ != 0 ||
      query_shape.GetDimension(2) != cache_shape.key_head_dim_ || query.GetStrides().GetStride(2) != 1 ||
      (query_heads > 1 && query.GetStrides().GetStride(1) != cache_shape.key_head_dim_) ||
      output_shape.GetDimension(1) != query_heads || output_shape.GetDimension(2) != cache_shape.value_head_dim_) {
    throw InvalidArgumentException("paged decode tensors do not match the KV cache");
  }

  if (!std::isfinite(params.softmax_scale_)) {
    throw InvalidArgumentException("attention softmax scale must be finite");
  }

  // The specialized kernels use int32 for row offsets and strides, and grid.y for the batch.
  constexpr auto max_index = std::numeric_limits<int32_t>::max();
  constexpr int64_t max_grid_dimension = 65535;
  const auto query_width = query_heads * head_dim;
  if (batch_size > max_grid_dimension || block_tables.GetNumElements() > max_index || query_width > max_index ||
      cache_shape.num_kv_heads_ > max_index / (head_dim * block_size) ||
      (batch_size > 1 && query_stride > (max_index - query_width) / (batch_size - 1))) {
    throw InvalidArgumentException("paged decode dimensions exceed the kernel index range");
  }

  // The host bound sizes the launch without copying device lengths. Keep one partition for empty contexts.
  const auto effective_max_context_len = std::min(max_context_len, max_blocks * block_size);
  const auto num_partitions =
      std::max<int64_t>(1, (effective_max_context_len + PAGED_DECODE_PARTITION_SIZE - 1) / PAGED_DECODE_PARTITION_SIZE);
  if (num_partitions > max_grid_dimension || output.GetNumElements() > max_index / num_partitions) {
    throw InvalidArgumentException("paged decode partitions exceed the kernel index range");
  }

  // Validate actual view addresses before submission; a callback failure invalidates the execution context.
  const auto query_key_alignment =
      static_cast<uintptr_t>(cache_shape.dtype_ == ttl::DType::BFLOAT16 ? 4 : block_size / 2);
  const auto value_alignment = static_cast<uintptr_t>(cache_shape.dtype_ == ttl::DType::BFLOAT16 ? 4 : 16);
  ttl::DispatchCudaFloatingDType(cache_shape.dtype_, "PagedDecode", [&]<ttl::CudaStorageType T>(std::type_identity<T>) {
    using Storage = ttl::StorageTypeForT<ttl::CUDA_DTYPE_OF<T>>;
    if (reinterpret_cast<uintptr_t>(query.GetData<Storage>()) % query_key_alignment != 0 ||
        reinterpret_cast<uintptr_t>(key_cache.GetData<Storage>()) % query_key_alignment != 0 ||
        reinterpret_cast<uintptr_t>(value_cache.GetData<Storage>()) % value_alignment != 0 ||
        static_cast<uintptr_t>(query_stride) % (query_key_alignment / sizeof(Storage)) != 0) {
      throw InvalidArgumentException("paged decode cache and query views must satisfy vector alignment");
    }
  });

  const std::array<ttl::Tensor, 5> inputs{query, key_cache, value_cache, block_tables, context_lens};
  for (const auto &input : inputs) {
    if (ttl::ClassifyAlias(output, input) != ttl::AliasKind::DISJOINT) {
      throw InvalidArgumentException("paged decode output must not alias an input");
    }
  }

  const std::array<ttl::Tensor *, 1> outputs{&output};
  // Workspace contains two FP32 statistics per partition, checked lengths, and input-dtype partition outputs.
  const auto tasks = batch_size * query_heads * num_partitions;
  const auto statistics_bytes = static_cast<size_t>(2 * tasks) * sizeof(float);
  const auto lengths_bytes = static_cast<size_t>(batch_size) * sizeof(int32_t);
  const auto partition_bytes =
      static_cast<size_t>(tasks * head_dim) * ttl::GetDTypeInfo(cache_shape.dtype_).size_bytes_;
  const auto workspace_bytes = statistics_bytes + lengths_bytes + partition_bytes;
  ttl::SubmitCudaKernel(
      context, "PagedDecode", std::span<const ttl::Tensor>{inputs}, std::span<ttl::Tensor *const>{outputs},
      [&](ttl::CudaKernelLaunch &launch) {
        LaunchPagedDecode(launch.GetStream(), cache_shape.dtype_, launch.GetInputData(inputs[0]),
                          launch.GetInputData(inputs[1]), launch.GetInputData(inputs[2]),
                          launch.GetInputDataAs<int32_t>(inputs[3]), launch.GetInputDataAs<int32_t>(inputs[4]),
                          launch.GetOutputData(output), batch_size, query_heads, max_blocks, query_stride,
                          params.softmax_scale_, cache_shape, effective_max_context_len, launch.GetWorkspace().data_,
                          launch.GetDeviceErrorContext(ttl::DType::INT32));
      },
      ttl::CudaKernelLaunchOptions{.workspace_ = {.size_bytes_ = workspace_bytes},
                                   .capture_policy_ = ttl::CudaCapturePolicy::SAFE});
}

}  // namespace zephyr::attention
