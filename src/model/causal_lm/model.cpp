#include "model/causal_lm/model.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include <ttl/ops/copy.hpp>
#include <ttl/ops/creation.hpp>
#include <ttl/ops/indexing.hpp>
#include <ttl/tensor/layout.hpp>

#include "common/exception.hpp"

namespace zephyr::model::causal_lm {

auto ModelForwardContext::SelectLogits(ttl::ExecutionContext &context, const ttl::Tensor &hidden_states) const
    -> ttl::Tensor {
  if (hidden_states.GetRank() != 3 || logits_ranges_.empty()) {
    throw InvalidArgumentException("logits selection requires [batch, sequence, hidden] and nonempty output ranges");
  }
  const auto &shape = hidden_states.GetShape();
  const auto batch = shape.GetDimension(0);
  const auto sequence = shape.GetDimension(1);
  const auto hidden = shape.GetDimension(2);
  const auto tokens = batch * sequence;
  const auto packed = flash_params_.packed_;
  if ((packed &&
       (!paged_attention_.query_lens_.has_value() || paged_attention_.query_lens_->size() != logits_ranges_.size())) ||
      (!packed && logits_ranges_.size() != static_cast<size_t>(batch))) {
    throw InvalidArgumentException("logits ranges must match the logical query batch");
  }

  const auto output_length = logits_ranges_.front().length_;
  int64_t total = 0;
  for (size_t row = 0; row < logits_ranges_.size(); ++row) {
    const auto length = packed ? (*paged_attention_.query_lens_)[row] : sequence;
    const auto &range = logits_ranges_[row];
    if (length < 0 || length > tokens - total || range.start_ < 0 || range.start_ > length || range.length_ < 0 ||
        range.length_ > length - range.start_ || range.length_ != output_length) {
      throw InvalidArgumentException("logits ranges must fit their queries and select the same length");
    }
    total += length;
  }
  if (total != tokens) {
    throw InvalidArgumentException("packed query lengths must cover the physical input tokens");
  }

  // A common rectangular span, including all tokens and single-token decode, needs only a view.
  const auto start = logits_ranges_.front().start_;
  if (!packed && std::ranges::all_of(logits_ranges_, [&](const auto &range) { return range.start_ == start; })) {
    return ttl::Narrow(hidden_states, 1, start, output_length);
  }

  std::vector<int64_t> indices;
  indices.reserve(logits_ranges_.size() * static_cast<size_t>(output_length));
  int64_t base = 0;
  for (size_t row = 0; row < logits_ranges_.size(); ++row) {
    const auto &range = logits_ranges_[row];
    for (int64_t index = 0; index < output_length; ++index) {
      indices.push_back(base + range.start_ + index);
    }
    base += packed ? (*paged_attention_.query_lens_)[row] : sequence;
  }
  auto device_indices = ttl::Empty(context, ttl::Shape{static_cast<int64_t>(indices.size())}, ttl::DType::INT64);
  ttl::CopyFromHostBlocking(context, device_indices, std::as_bytes(std::span{indices}));
  const auto flat = ttl::Reshape(context, hidden_states, ttl::Shape{tokens, hidden});
  return ttl::View(ttl::IndexSelect(context, flat, 0, device_indices),
                   ttl::Shape{static_cast<int64_t>(logits_ranges_.size()), output_length, hidden});
}

}  // namespace zephyr::model::causal_lm
