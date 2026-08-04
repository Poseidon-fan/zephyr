#pragma once

#include <cstdint>
#include <source_location>
#include <utility>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/** @brief Selects the axis, count, ordering direction, and requested result ordering for `TopK`. */
struct TopKOptions final {
  /** Axis along which values are selected. */
  int64_t axis_{-1};
  /** Number of values selected from each axis slice. */
  int64_t k_{0};
  /** Select greatest values when true and least values when false. */
  bool largest_{true};
  /** Reserved ordering control. The current implementation always returns values in selection order. */
  bool sorted_{true};
};

/**
 * @brief Select the `k_` largest or smallest values and their INT64 source indices along `axis_`.
 *
 * `k_` lies in `[0, input.shape[axis_]]`. Ties preserve lower source indices, and floating NaNs compare as greater
 * than numeric values. `values` and `indices` must be non-overlapping dense, mutually disjoint, and disjoint from the
 * input. Results are returned in selection order.
 */
void TopKOut(ExecutionContext &context, Tensor &values, Tensor &indices, const Tensor &input,
             const TopKOptions &options, std::source_location location = std::source_location::current());

/** @brief Allocate canonical contiguous value and INT64 index tensors and perform top-k selection. */
[[nodiscard]] auto TopK(ExecutionContext &context, const Tensor &input, const TopKOptions &options,
                        std::source_location location = std::source_location::current()) -> std::pair<Tensor, Tensor>;

}  // namespace ttl
