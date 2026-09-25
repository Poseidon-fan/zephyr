#pragma once

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

namespace zephyr::attention {

/**
 * Writes dense-head K/V rows into existing physical page slots. Inputs may be
 * rank 3 [tokens, heads, dim] or rank 4 [batch, sequence, heads, dim].
 * Negative slot mapping entries are padding rows and are skipped.
 * Out-of-range non-negative slots are skipped and reported through
 * context.Synchronize() or context.CheckAsyncErrors().
 */
void ReshapeAndCache(ttl::ExecutionContext &context, const ttl::Tensor &key, const ttl::Tensor &value,
                     ttl::Tensor &key_cache, ttl::Tensor &value_cache, const ttl::Tensor &slot_mapping);

/**
 * Gathers paged K/V data into packed token-major tensors for dense SDPA.
 * cu_seqlens_k must start at zero, be nondecreasing, and end at the output
 * token count. Empty sequences are allowed. Each sequence must fit its table
 * row and reference physical cache pages; unused table entries are ignored.
 * Invalid metadata prevents the gather and is reported through
 * context.Synchronize() or context.CheckAsyncErrors().
 */
void GatherKvCache(ttl::ExecutionContext &context, ttl::Tensor &key_output, ttl::Tensor &value_output,
                   const ttl::Tensor &key_cache, const ttl::Tensor &value_cache, const ttl::Tensor &block_tables,
                   const ttl::Tensor &cu_seqlens_k);

}  // namespace zephyr::attention
