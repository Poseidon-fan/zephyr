#include "attention/sdpa.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <ttl/ops/attention.hpp>
#include <ttl/ops/composition.hpp>
#include <ttl/ops/copy.hpp>
#include <ttl/tensor/layout.hpp>

#include "common/exception.hpp"

namespace zephyr::attention {

auto RunAttention(ttl::ExecutionContext &context, const ttl::Tensor &query, const ttl::Tensor &key,
                  const ttl::Tensor &value, const AttentionMask &mask, const FlashParams *flash_params,
                  const SdpaParams &sdpa_params) -> ttl::Tensor {
  if (query.GetRank() != 4 || key.GetRank() != 4 || value.GetRank() != 4) {
    throw InvalidArgumentException("SDPA requires [batch, heads, sequence, head_dim] tensors");
  }
  const auto query_heads = query.GetShape().GetDimension(1);
  const auto kv_heads = key.GetShape().GetDimension(1);
  if (kv_heads <= 0 || query_heads <= 0 || query_heads % kv_heads != 0 ||
      sdpa_params.n_kv_groups_ != query_heads / kv_heads) {
    throw InvalidArgumentException("SDPA KV group count does not match the query and key heads");
  }
  if (mask.IsCustom() && mask.AsOptionTensor()->GetDType() != query.GetDType()) {
    throw InvalidArgumentException("an additive attention mask must use the query dtype");
  }
  if (flash_params != nullptr && flash_params->packed_ &&
      (mask.IsNone() || mask.IsCustom() || !flash_params->causal_)) {
    throw InvalidArgumentException("packed prefill requires causal variable-length attention");
  }

  const ttl::SdpaOptions options{
      .scale_ = sdpa_params.softmax_scale_,
      .causal_ = flash_params != nullptr ? flash_params->causal_ : !mask.IsNone() && !mask.IsCustom(),
      .causal_alignment_ = ttl::CausalAlignment::LOWER_RIGHT};
  const auto has_boundaries = flash_params != nullptr && flash_params->cumulative_seqlens_q_.has_value() &&
                              flash_params->logical_k_.cumulative_seqlens_.has_value();
  if (mask.IsCustom() || !has_boundaries) {
    if (flash_params != nullptr && flash_params->packed_) {
      throw InvalidArgumentException("packed attention requires cumulative query and key lengths");
    }
    const auto additive_mask = mask.IsCustom() ? std::optional{*mask.AsOptionTensor()} : std::nullopt;
    return ttl::ScaledDotProductAttention(context, query, key, value, additive_mask, options);
  }

  // TTL provides rectangular SDPA. Keep the same logical sequence boundaries when adapting a varlen call.
  const auto read_boundaries = [&](const ttl::Tensor &tensor, int64_t total, int32_t maximum) {
    if (tensor.GetRank() != 1 || tensor.GetDType() != ttl::DType::INT32 || tensor.GetNumElements() < 2) {
      throw InvalidArgumentException("cumulative attention lengths must be an INT32 vector with at least two entries");
    }
    std::vector<int32_t> lengths(static_cast<size_t>(tensor.GetNumElements()));
    ttl::CopyToHostBlocking(context, std::as_writable_bytes(std::span{lengths}), tensor);
    if (lengths.front() != 0 || lengths.back() != total) {
      throw InvalidArgumentException("cumulative attention lengths must cover the physical tensor exactly");
    }
    for (size_t index = 1; index < lengths.size(); ++index) {
      if (lengths[index] < lengths[index - 1] || lengths[index] - lengths[index - 1] > maximum) {
        throw InvalidArgumentException("attention lengths must be nondecreasing and respect their declared maximum");
      }
    }
    return lengths;
  };
  const auto tokens = query.GetShape().GetDimension(0) * query.GetShape().GetDimension(2);
  const auto key_tokens = key.GetShape().GetDimension(0) * key.GetShape().GetDimension(2);
  const auto cu_q = read_boundaries(*flash_params->cumulative_seqlens_q_, tokens, flash_params->max_q_);
  const auto cu_k =
      read_boundaries(*flash_params->logical_k_.cumulative_seqlens_, key_tokens, flash_params->logical_k_.max_);
  if (cu_q.size() != cu_k.size() || value.GetShape().GetDimension(0) != key.GetShape().GetDimension(0) ||
      value.GetShape().GetDimension(2) != key.GetShape().GetDimension(2)) {
    throw InvalidArgumentException("variable-length Q/K/V must describe the same logical sequences");
  }

  const auto key_dim = key.GetShape().GetDimension(3);
  const auto value_dim = value.GetShape().GetDimension(3);
  const auto q = ttl::Reshape(context, ttl::Transpose(query, 1, 2), ttl::Shape{tokens, query_heads, key_dim});
  const auto k = ttl::Reshape(context, ttl::Transpose(key, 1, 2), ttl::Shape{key_tokens, kv_heads, key_dim});
  const auto v = ttl::Reshape(context, ttl::Transpose(value, 1, 2), ttl::Shape{key_tokens, kv_heads, value_dim});
  std::vector<ttl::Tensor> outputs;
  outputs.reserve(cu_q.size() - 1);
  for (size_t sequence = 0; sequence + 1 < cu_q.size(); ++sequence) {
    const auto q_length = cu_q[sequence + 1] - cu_q[sequence];
    if (q_length == 0) {
      continue;
    }
    const auto k_length = cu_k[sequence + 1] - cu_k[sequence];
    const auto q_row = ttl::Unsqueeze(ttl::Transpose(ttl::Narrow(q, 0, cu_q[sequence], q_length), 0, 1), 0);
    const auto k_row = ttl::Unsqueeze(ttl::Transpose(ttl::Narrow(k, 0, cu_k[sequence], k_length), 0, 1), 0);
    const auto v_row = ttl::Unsqueeze(ttl::Transpose(ttl::Narrow(v, 0, cu_k[sequence], k_length), 0, 1), 0);
    auto output = ttl::ScaledDotProductAttention(context, q_row, k_row, v_row, std::nullopt, options);
    outputs.push_back(ttl::Squeeze(ttl::Transpose(output, 1, 2), 0));
  }
  if (outputs.empty()) {
    return ttl::Empty(
        context, ttl::Shape{query.GetShape().GetDimension(0), query_heads, query.GetShape().GetDimension(2), value_dim},
        query.GetDType());
  }
  const auto packed_output = ttl::Concat(context, outputs, 0);
  return ttl::Transpose(ttl::View(packed_output, ttl::Shape{query.GetShape().GetDimension(0),
                                                            query.GetShape().GetDimension(2), query_heads, value_dim}),
                        1, 2);
}

}  // namespace zephyr::attention
