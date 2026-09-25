#pragma once

#include <cstdint>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

namespace zephyr::attention {

/** Parameters used by one paged decode invocation. */
struct AttentionParams final {
  /** Multiplicative scale applied to every query-key dot product. */
  float softmax_scale_;
};

/**
 * Computes one-token-per-sequence decode directly from the paged KV cache.
 * Query and output are [batch, query_heads, head_dim], with the same head dimension in K and V.
 * Supported head dimensions are 32, 64, 80, 96, 112, 120, 128, 192, and 256; page sizes are 8, 16, and 32.
 * Query heads are contiguous; batches may be strided or broadcast. For FP16/FP32, Q/K addresses and
 * query batch strides must align to page_size / 2 bytes, and V addresses to 16 bytes. BF16 requires
 * 4-byte alignment for all three tensors. Kernel row offsets and strides must fit signed 32-bit indexing.
 * The key cache has fewer heads for grouped-query attention when applicable. Context lengths must be
 * non-negative and fit their block-table rows; empty contexts produce zeros.
 * max_context_len is a host-side upper bound on context_lens, including the current token. Normally it
 * is the batch maximum; CUDA Graph capture may use a larger bound covering subsequent replays.
 * The bound, capped by the block-table capacity, determines partition count and workspace size.
 * It must be non-negative; zero is valid only when all contexts are empty. It does not truncate attention.
 * Referenced block IDs must be in the supplied cache's physical page range.
 * Partition outputs use the input dtype; softmax statistics use FP32. Warm up
 * the required shapes before capturing a CUDA graph so its scratch capacity is sufficient.
 * Invalid metadata stops the affected row and is reported through
 * context.Synchronize() or context.CheckAsyncErrors(); its output must be discarded.
 */
void PagedDecode(ttl::ExecutionContext &context, ttl::Tensor &output, const ttl::Tensor &query,
                 const ttl::Tensor &key_cache, const ttl::Tensor &value_cache, const ttl::Tensor &block_tables,
                 const ttl::Tensor &context_lens, int64_t max_context_len, const AttentionParams &params);

}  // namespace zephyr::attention
