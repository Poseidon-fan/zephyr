#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <driver_types.h>

#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {

/**
 * @brief Trivially copyable host-to-kernel ABI for a TopK launch.
 *
 * shape excludes no dimensions; slice_count is the product of dimensions other than axis. Value and input strides are
 * byte offsets, while index strides are measured in int64_t elements. The sort launcher fills sort_item_count and
 * output_item_count in its launch-local copy.
 */
struct TopKParameters final {
  std::byte *values_;
  int64_t *indices_;
  const std::byte *input_;
  uint64_t shape_[TTL_MAX_RANK]{};
  uint64_t value_strides_bytes_[TTL_MAX_RANK]{};
  uint64_t index_strides_elements_[TTL_MAX_RANK]{};
  uint64_t input_strides_bytes_[TTL_MAX_RANK]{};
  uint64_t slice_count_;
  uint64_t axis_size_;
  uint64_t k_;
  uint64_t sort_item_count_;
  uint64_t output_item_count_;
  uint8_t rank_;
  uint8_t axis_;
  bool largest_;
};

static_assert(std::is_trivially_copyable_v<TopKParameters>);
static_assert(std::is_standard_layout_v<TopKParameters>);

[[nodiscard]] auto GetTopKSortWorkspaceBytes(DType dtype, int32_t num_items, int32_t num_segments, int32_t axis_size,
                                             bool largest,
                                             std::source_location location = std::source_location::current()) -> size_t;

/** Launch the fixed-capacity shared-memory sorting network for axis_size in [1, 1024] and k in [1, axis_size]. */
void LaunchTopKSmall(cudaStream_t stream, DType dtype, const TopKParameters &parameters,
                     std::source_location location = std::source_location::current());
/** Launch CUB radix sort with packed row keys; full-width INT64 keys retain segmented sorting. */
void LaunchTopKSort(cudaStream_t stream, DType dtype, const TopKParameters &parameters, uint64_t *keys_input,
                    uint64_t *keys_output, int64_t *indices_input, int64_t *indices_output, void *workspace,
                    size_t workspace_bytes, std::source_location location = std::source_location::current());
/** Launch the workspace-free fallback that scans one complete axis slice per CUDA thread. */
void LaunchTopKSerial(cudaStream_t stream, DType dtype, const TopKParameters &parameters,
                      std::source_location location = std::source_location::current());

}  // namespace ttl::internal
