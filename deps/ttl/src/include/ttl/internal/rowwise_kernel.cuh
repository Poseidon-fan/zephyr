#pragma once

#include <cstddef>
#include <type_traits>

#include <cuda_runtime.h>  // IWYU pragma: keep

namespace ttl::internal {

template <typename Parameters>
using RowwiseIndexType = std::remove_cvref_t<decltype(Parameters::group_count_)>;

template <typename Parameters>
__device__ auto GetRowwiseGroupInputOffset(const Parameters &parameters, RowwiseIndexType<Parameters> group_index)
    -> RowwiseIndexType<Parameters> {
  using Index = RowwiseIndexType<Parameters>;
  auto offset = Index{0};
  for (size_t remaining = parameters.group_rank_; remaining > 0; --remaining) {
    const auto axis = remaining - 1;
    const auto coordinate = static_cast<Index>(group_index % parameters.group_shape_[axis]);
    group_index = static_cast<Index>(group_index / parameters.group_shape_[axis]);
    offset = static_cast<Index>(offset + coordinate * parameters.group_input_strides_bytes_[axis]);
  }
  return offset;
}

template <typename Parameters>
__device__ auto GetRowwiseGroupOutputOffset(const Parameters &parameters, RowwiseIndexType<Parameters> group_index)
    -> RowwiseIndexType<Parameters> {
  using Index = RowwiseIndexType<Parameters>;
  auto offset = Index{0};
  for (size_t remaining = parameters.group_rank_; remaining > 0; --remaining) {
    const auto axis = remaining - 1;
    const auto coordinate = static_cast<Index>(group_index % parameters.group_shape_[axis]);
    group_index = static_cast<Index>(group_index / parameters.group_shape_[axis]);
    offset = static_cast<Index>(offset + coordinate * parameters.group_output_strides_bytes_[axis]);
  }
  return offset;
}

template <typename Parameters>
__device__ auto GetRowwiseReductionOffset(const Parameters &parameters, RowwiseIndexType<Parameters> reduction_index,
                                          bool output) -> RowwiseIndexType<Parameters> {
  using Index = RowwiseIndexType<Parameters>;
  const auto contiguous = output ? parameters.contiguous_output_reduction_ : parameters.contiguous_input_reduction_;
  if (contiguous) {
    for (size_t remaining = parameters.reduction_rank_; remaining > 0; --remaining) {
      const auto axis = remaining - 1;
      if (parameters.reduction_shape_[axis] > 1) {
        const auto stride =
            output ? parameters.reduction_output_strides_bytes_[axis] : parameters.reduction_input_strides_bytes_[axis];
        return static_cast<Index>(reduction_index * stride);
      }
    }
    return Index{0};
  }

  auto offset = Index{0};
  for (size_t remaining = parameters.reduction_rank_; remaining > 0; --remaining) {
    const auto axis = remaining - 1;
    const auto coordinate = static_cast<Index>(reduction_index % parameters.reduction_shape_[axis]);
    reduction_index = static_cast<Index>(reduction_index / parameters.reduction_shape_[axis]);
    const auto stride =
        output ? parameters.reduction_output_strides_bytes_[axis] : parameters.reduction_input_strides_bytes_[axis];
    offset = static_cast<Index>(offset + coordinate * stride);
  }
  return offset;
}

}  // namespace ttl::internal
