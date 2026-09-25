#include "layer/paged_attention.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <ttl/ops/attention.hpp>
#include <ttl/ops/cast.hpp>
#include <ttl/ops/composition.hpp>
#include <ttl/ops/copy.hpp>
#include <ttl/ops/creation.hpp>
#include <ttl/tensor/layout.hpp>

#include "attention/cache.hpp"
#include "attention/kernels.cuh"
#include "attention/paged_decode.hpp"
#include "common/exception.hpp"

namespace zephyr::layer {

auto PagedAttention::Forward(ttl::ExecutionContext &context, const ttl::Tensor &query, const ttl::Tensor &key,
                             const ttl::Tensor &value, const attention::AttentionMask &attention_mask,
                             std::optional<ttl::Tensor> key_cache, std::optional<ttl::Tensor> value_cache,
                             const PagedAttentionInputMetadata &input_metadata,
                             const attention::SdpaParams &sdpa_params, const attention::FlashParams *flash_params) const
    -> ttl::Tensor {
  if (query.GetRank() != 4 || key.GetRank() != 4 || value.GetRank() != 4 ||
      key_cache.has_value() != value_cache.has_value()) {
    throw InvalidArgumentException("paged attention requires rank-four Q/K/V and either both caches or neither");
  }
  const auto batch = query.GetShape().GetDimension(0);
  const auto heads = query.GetShape().GetDimension(1);
  const auto sequence_length = query.GetShape().GetDimension(2);
  const auto head_dim = query.GetShape().GetDimension(3);
  const auto kv_heads = key.GetShape().GetDimension(1);
  if (key.GetShape().GetDimension(0) != batch || key.GetShape().GetDimension(2) != sequence_length ||
      key.GetShape().GetDimension(3) != head_dim || value.GetShape() != key.GetShape() || kv_heads <= 0 || heads <= 0 ||
      heads % kv_heads != 0 || sdpa_params.n_kv_groups_ != heads / kv_heads) {
    throw InvalidArgumentException("paged Q/K/V shapes and KV group count must describe the same new tokens");
  }
  const auto slots = ttl::Flatten(context, ttl::Contiguous(context, input_metadata.slot_mappings_));
  const auto gather =
      input_metadata.block_tables_.has_value() &&
      (input_metadata.num_cached_tokens_.has_value() ||
       (!attention_mask.IsNone() && input_metadata.query_lens_.has_value() && !input_metadata.is_first_prompt_chunk_));
  const auto single_token_first_prompt = input_metadata.is_first_prompt_chunk_ && sequence_length == 1;
  const auto custom_decode =
      attention_mask.IsCustom() && !input_metadata.query_lens_.has_value() && !input_metadata.is_first_prompt_chunk_;
  const auto prompt = !custom_decode && (!attention_mask.IsNone() || single_token_first_prompt);

  const auto write_cache = [&] {
    if (!key_cache.has_value()) {
      return;
    }
    // Cache kernels accept token rows with dense heads. Materialize all other layouts with canonical strides.
    const auto cache_input = [&](const ttl::Tensor &input) {
      auto transposed = ttl::Transpose(input, 1, 2);
      const auto strides = transposed.GetStrides();
      const auto row_stride = sequence_length == 1 ? strides.GetStride(0) : strides.GetStride(1);
      if (strides.GetStride(3) != 1 || (kv_heads > 1 && strides.GetStride(2) != head_dim) ||
          row_stride < kv_heads * head_dim ||
          (batch > 1 && sequence_length > 1 && strides.GetStride(0) != sequence_length * row_stride)) {
        // Contiguous may preserve singleton strides, so this path requires an unconditional copy.
        return ttl::View(ttl::Clone(context, transposed), ttl::Shape{batch * sequence_length, kv_heads, head_dim});
      }
      return transposed;
    };
    attention::ReshapeAndCache(context, cache_input(key), cache_input(value), *key_cache, *value_cache, slots);
  };

  // An uncached prompt consumes its Q/K/V before the current tokens are written to the optional cache.
  if (!gather && prompt) {
    auto output = attention::RunAttention(context, query, key, value, attention_mask, flash_params, sdpa_params);
    write_cache();
    return output;
  }
  if (!key_cache.has_value() || !input_metadata.block_tables_.has_value()) {
    throw InvalidArgumentException("cached attention requires K/V caches and block tables");
  }
  if (ttl::ClassifyAlias(query, *key_cache) != ttl::AliasKind::DISJOINT ||
      ttl::ClassifyAlias(query, *value_cache) != ttl::AliasKind::DISJOINT) {
    throw InvalidArgumentException("cache writes must not overwrite the attention query");
  }
  write_cache();
  if (gather) {
    const auto output = GatherAttention(context, query, *key_cache, *value_cache, attention_mask, input_metadata,
                                        sdpa_params, flash_params, false);
    // Only prefix attention with no explicit mask uses token-major output.
    return attention_mask.IsNone() ? ttl::Contiguous(context, ttl::Transpose(output, 1, 2)) : output;
  }

  const auto token_major = sequence_length > 1 ? ttl::Transpose(query, 1, 2) : query;
  const auto packed_query =
      ttl::View(ttl::Contiguous(context, token_major), ttl::Shape{batch * sequence_length, heads, head_dim});
  const auto cache_shape = attention::GetCacheShape(*key_cache, *value_cache);
  if (attention_mask.IsCustom() || !attention::SupportsPagedDecode(head_dim, cache_shape.block_size_)) {
    const auto query_rows = ttl::Unsqueeze(packed_query, 2);
    // Decode consumes each row's complete context. Prompt flash metadata does not define this call's boundaries.
    return GatherAttention(context, query_rows, *key_cache, *value_cache, attention_mask, input_metadata, sdpa_params,
                           nullptr, true);
  }
  if (!input_metadata.context_lens_.has_value() || !input_metadata.max_context_len_.has_value()) {
    throw InvalidArgumentException("paged decode requires context lengths and their host maximum");
  }
  auto output = ttl::Empty(context, packed_query.GetShape(), query.GetDType());
  attention::PagedDecode(context, output, packed_query, *key_cache, *value_cache, *input_metadata.block_tables_,
                         *input_metadata.context_lens_, *input_metadata.max_context_len_,
                         {.softmax_scale_ = sdpa_params.softmax_scale_});
  return output;
}

auto PagedAttention::GatherAttention(ttl::ExecutionContext &context, const ttl::Tensor &query,
                                     const ttl::Tensor &key_cache, const ttl::Tensor &value_cache,
                                     const attention::AttentionMask &attention_mask,
                                     const PagedAttentionInputMetadata &input_metadata,
                                     const attention::SdpaParams &sdpa_params,
                                     const attention::FlashParams *flash_params, bool decode) -> ttl::Tensor {
  const auto batch = query.GetShape().GetDimension(0);
  const auto heads = query.GetShape().GetDimension(1);
  const auto query_max = query.GetShape().GetDimension(2);
  const auto head_dim = query.GetShape().GetDimension(3);
  const auto cache_shape = attention::GetCacheShape(key_cache, value_cache);
  if (cache_shape.key_head_dim_ != head_dim || cache_shape.value_head_dim_ != head_dim) {
    throw InvalidArgumentException("gathered KV head dimensions must match the query");
  }

  std::vector<int64_t> query_lengths;
  if (decode) {
    query_lengths.assign(static_cast<size_t>(batch), 1);
  } else if (input_metadata.query_lens_.has_value()) {
    query_lengths = *input_metadata.query_lens_;
  } else {
    // Input preparation normally supplies host lengths. Fall back to counting non-padding slots.
    const auto slots = ttl::Contiguous(context, input_metadata.slot_mappings_);
    if (slots.GetDType() != ttl::DType::INT64 || slots.GetNumElements() != batch * query_max) {
      throw InvalidArgumentException("prompt slot mappings must cover the physical query rows");
    }
    std::vector<int64_t> host_slots(static_cast<size_t>(slots.GetNumElements()));
    ttl::CopyToHostBlocking(context, std::as_writable_bytes(std::span{host_slots}), slots);
    for (int64_t row = 0; row < batch; ++row) {
      const auto first = host_slots.begin() + (row * query_max);
      query_lengths.push_back(std::count_if(first, first + query_max, [](auto slot) { return slot != -1; }));
    }
  }

  std::vector<int64_t> key_lengths;
  if (input_metadata.paged_context_lens_cpu_.has_value()) {
    key_lengths = *input_metadata.paged_context_lens_cpu_;
  } else if (decode) {
    if (!input_metadata.context_lens_.has_value() || input_metadata.context_lens_->GetDType() != ttl::DType::INT32 ||
        input_metadata.context_lens_->GetRank() != 1) {
      throw InvalidArgumentException("decode gather requires an INT32 context-length vector");
    }
    std::vector<int32_t> host_lengths(static_cast<size_t>(input_metadata.context_lens_->GetNumElements()));
    ttl::CopyToHostBlocking(context, std::as_writable_bytes(std::span{host_lengths}), *input_metadata.context_lens_);
    key_lengths.assign(host_lengths.begin(), host_lengths.end());
  } else {
    key_lengths = query_lengths;
    if (input_metadata.num_cached_tokens_.has_value()) {
      if (input_metadata.num_cached_tokens_->size() != key_lengths.size()) {
        throw InvalidArgumentException("cached token counts must match the logical batch");
      }
      for (size_t row = 0; row < key_lengths.size(); ++row) {
        const auto cached = (*input_metadata.num_cached_tokens_)[row];
        if (cached < 0 || key_lengths[row] < 0 || cached > std::numeric_limits<int32_t>::max() - key_lengths[row]) {
          throw InvalidArgumentException("cached and new token counts must fit INT32 sequence lengths");
        }
        key_lengths[row] += cached;
      }
    }
  }
  if (query_lengths.empty() || query_lengths.size() != key_lengths.size() ||
      input_metadata.block_tables_->GetRank() != 2 ||
      input_metadata.block_tables_->GetShape().GetDimension(0) != static_cast<int64_t>(key_lengths.size())) {
    throw InvalidArgumentException("gathered query, context, and block-table rows must have the same logical batch");
  }

  int64_t query_tokens = 0;
  int64_t key_tokens = 0;
  std::vector<int32_t> cumulative{0};
  cumulative.reserve(key_lengths.size() + 1);
  for (size_t row = 0; row < key_lengths.size(); ++row) {
    if (query_lengths[row] < 0 || key_lengths[row] < 0 ||
        key_lengths[row] > std::numeric_limits<int32_t>::max() - key_tokens ||
        query_lengths[row] > (batch * query_max) - query_tokens) {
      throw InvalidArgumentException("gathered sequence lengths exceed their physical tensors or kernel index range");
    }
    query_tokens += query_lengths[row];
    key_tokens += key_lengths[row];
    cumulative.push_back(static_cast<int32_t>(key_tokens));
  }
  const auto dense_queries = query_tokens == batch * query_max;
  if (!dense_queries && (query_lengths.size() != static_cast<size_t>(batch) ||
                         std::ranges::any_of(query_lengths, [&](auto length) { return length > query_max; }))) {
    throw InvalidArgumentException("padded query lengths must fit their physical batch rows");
  }
  auto cu_k = !decode ? input_metadata.cu_seqlens_kv_ : std::nullopt;
  if (!cu_k.has_value()) {
    cu_k = ttl::Empty(context, ttl::Shape{static_cast<int64_t>(cumulative.size())}, ttl::DType::INT32);
    ttl::CopyFromHostBlocking(context, *cu_k, std::as_bytes(std::span{cumulative}));
  }
  auto key = ttl::Empty(context, ttl::Shape{key_tokens, cache_shape.num_kv_heads_, head_dim}, query.GetDType());
  auto value = ttl::Empty(context, key.GetShape(), query.GetDType());
  attention::GatherKvCache(context, key, value, key_cache, value_cache, *input_metadata.block_tables_, *cu_k);

  const auto declared_causal = flash_params == nullptr || flash_params->causal_;
  const auto prefix_causal =
      declared_causal && std::ranges::any_of(query_lengths, [](auto length) { return length > 1; });
  const auto max_key = *std::ranges::max_element(key_lengths);
  const auto has_padding = std::ranges::any_of(query_lengths, [&](auto length) { return length != query_max; }) ||
                           std::ranges::any_of(key_lengths, [&](auto length) { return length != max_key; });
  // Unmasked prefix batches with padding still reconstruct their prefix causal mask.
  const auto reconstruct_prefix_mask = attention_mask.IsNone() && has_padding && !prefix_causal;
  if (dense_queries && !attention_mask.IsCustom() && !reconstruct_prefix_mask) {
    // A packed context is a collection of independent sequences, with each query aligned to its suffix.
    const auto packed_query =
        ttl::View(ttl::Contiguous(context, ttl::Transpose(query, 1, 2)), ttl::Shape{query_tokens, heads, head_dim});
    const ttl::SdpaOptions options{
        .scale_ = sdpa_params.softmax_scale_,
        .causal_ = flash_params != nullptr ? prefix_causal : !attention_mask.IsNone() || prefix_causal,
        .causal_alignment_ = ttl::CausalAlignment::LOWER_RIGHT};
    std::vector<ttl::Tensor> outputs;
    outputs.reserve(query_lengths.size());
    int64_t query_offset = 0;
    for (size_t row = 0; row < query_lengths.size(); ++row) {
      const auto q_len = query_lengths[row];
      const auto k_len = key_lengths[row];
      if (q_len != 0) {
        const auto q = ttl::Unsqueeze(ttl::Transpose(ttl::Narrow(packed_query, 0, query_offset, q_len), 0, 1), 0);
        const auto k = ttl::Unsqueeze(ttl::Transpose(ttl::Narrow(key, 0, cumulative[row], k_len), 0, 1), 0);
        const auto v = ttl::Unsqueeze(ttl::Transpose(ttl::Narrow(value, 0, cumulative[row], k_len), 0, 1), 0);
        const auto output = ttl::ScaledDotProductAttention(context, q, k, v, std::nullopt, options);
        outputs.push_back(ttl::Squeeze(ttl::Transpose(output, 1, 2), 0));
      }
      query_offset += q_len;
    }
    const auto packed_output = outputs.empty() ? ttl::Empty(context, ttl::Shape{0, heads, head_dim}, query.GetDType())
                                               : ttl::Concat(context, outputs, 0);
    const auto output = ttl::View(packed_output, ttl::Shape{batch, query_max, heads, head_dim});
    return ttl::Transpose(output, 1, 2);
  }

  const auto unpack = [&](const ttl::Tensor &packed) {
    std::vector<ttl::Tensor> rows;
    rows.reserve(key_lengths.size());
    for (size_t row = 0; row < key_lengths.size(); ++row) {
      const auto length = key_lengths[row];
      auto unpacked = ttl::Unsqueeze(ttl::Transpose(ttl::Narrow(packed, 0, cumulative[row], length), 0, 1), 0);
      if (length < max_key) {
        const std::array parts{unpacked,
                               ttl::Zeros(context, ttl::Shape{1, cache_shape.num_kv_heads_, max_key - length, head_dim},
                                          query.GetDType())};
        unpacked = ttl::Concat(context, parts, 2);
      }
      rows.push_back(std::move(unpacked));
    }
    return ttl::Concat(context, rows, 0);
  };
  auto mask = attention_mask;
  if (mask.IsCustom()) {
    auto tensor = *mask.AsOptionTensor();
    if (tensor.GetRank() >= 2 && tensor.GetRank() <= 4) {
      const auto axis = static_cast<int64_t>(tensor.GetRank()) - 1;
      const auto extent = tensor.GetShape().GetDimension(axis);
      if (extent > max_key) {
        tensor = ttl::Narrow(tensor, axis, extent - max_key, max_key);
      }
    }
    mask = attention::AttentionMask::Custom(std::move(tensor));
  } else {
    // Padding queries attend only to the first KV row; real queries cannot see padding or future tokens.
    const ttl::Shape mask_shape{static_cast<int64_t>(key_lengths.size()), 1, query_max, max_key};
    std::vector<float> values(static_cast<size_t>(mask_shape.GetNumElements()));
    size_t index = 0;
    for (size_t row = 0; row < key_lengths.size(); ++row) {
      const auto prefix = std::max(key_lengths[row] - query_lengths[row], int64_t{0});
      for (int64_t q_index = 0; q_index < query_max; ++q_index) {
        for (int64_t k_index = 0; k_index < max_key; ++k_index) {
          const auto masked =
              q_index >= query_lengths[row] ? k_index != 0 : k_index >= key_lengths[row] || k_index > prefix + q_index;
          values[index++] = masked ? -std::numeric_limits<float>::infinity() : 0.0F;
        }
      }
    }
    auto tensor = ttl::Empty(context, mask_shape, ttl::DType::FLOAT32);
    ttl::CopyFromHostBlocking(context, tensor, std::as_bytes(std::span{values}));
    mask = attention::AttentionMask::Custom(ttl::Cast(context, tensor, query.GetDType()));
  }
  return attention::RunAttention(context, query, unpack(key), unpack(value), mask, nullptr, sdpa_params);
}

}  // namespace zephyr::layer
