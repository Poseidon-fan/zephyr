#pragma once

#include <cstdint>
#include <source_location>
#include <vector>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

/** @brief Selects reduction axes and whether reduced dimensions remain with extent one. */
struct ReductionOptions final {
  /** Axes to reduce. An empty list means every input axis. */
  std::vector<int64_t> axes_;
  bool keep_dimensions_{false};
};

/**
 * @brief Sum values over the selected axes into a caller-provided output.
 *
 * Supported dtypes are INT32, INT64, FLOAT16, BFLOAT16, and FLOAT32. Integer overflow uses modular arithmetic. The
 * output must be non-overlapping dense and disjoint from the input.
 */
void SumOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
            std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and sum over the selected axes. */
[[nodiscard]] auto Sum(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                       std::source_location location = std::source_location::current()) -> Tensor;

/** @brief Compute the arithmetic mean of a floating tensor over the selected axes. */
void MeanOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
             std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and compute means over the selected axes. */
[[nodiscard]] auto Mean(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                        std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Compute minimum values over the selected axes.
 *
 * The reduced extent must be nonzero. Floating NaNs propagate to the result.
 */
void MinimumOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
                std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and compute minima over the selected axes. */
[[nodiscard]] auto Minimum(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                           std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Compute maximum values over the selected axes.
 *
 * The reduced extent must be nonzero. Floating NaNs propagate to the result.
 */
void MaximumOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
                std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous output and compute maxima over the selected axes. */
[[nodiscard]] auto Maximum(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                           std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Return INT64 indices of minimum values along one nonempty axis.
 *
 * Ties select the lowest source index. A floating NaN wins over numeric values, with the lowest NaN index selected.
 */
void ArgMinOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis,
               bool keep_dimension = false, std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous INT64 tensor containing indices of minima. */
[[nodiscard]] auto ArgMin(ExecutionContext &context, const Tensor &input, int64_t axis, bool keep_dimension = false,
                          std::source_location location = std::source_location::current()) -> Tensor;

/**
 * @brief Return INT64 indices of maximum values along one nonempty axis.
 *
 * Ties select the lowest source index. A floating NaN wins over numeric values, with the lowest NaN index selected.
 */
void ArgMaxOut(ExecutionContext &context, Tensor &output, const Tensor &input, int64_t axis,
               bool keep_dimension = false, std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous INT64 tensor containing indices of maxima. */
[[nodiscard]] auto ArgMax(ExecutionContext &context, const Tensor &input, int64_t axis, bool keep_dimension = false,
                          std::source_location location = std::source_location::current()) -> Tensor;

/** @brief Reduce a BOOL tensor with logical OR over the selected axes. */
void AnyOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
            std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous BOOL tensor and reduce with logical OR. */
[[nodiscard]] auto Any(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                       std::source_location location = std::source_location::current()) -> Tensor;

/** @brief Reduce a BOOL tensor with logical AND over the selected axes. */
void AllOut(ExecutionContext &context, Tensor &output, const Tensor &input, const ReductionOptions &options = {},
            std::source_location location = std::source_location::current());

/** @brief Allocate a canonical contiguous BOOL tensor and reduce with logical AND. */
[[nodiscard]] auto All(ExecutionContext &context, const Tensor &input, const ReductionOptions &options = {},
                       std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
