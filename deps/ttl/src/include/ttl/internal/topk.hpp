#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <driver_types.h>

#include "ttl/dtype.hpp"
#include "ttl/shape.hpp"

namespace ttl::internal {

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
  uint8_t rank_;
  uint8_t axis_;
  bool largest_;
};

static_assert(std::is_trivially_copyable_v<TopKParameters>);
static_assert(std::is_standard_layout_v<TopKParameters>);

[[nodiscard]] auto GetTopKSortWorkspaceBytes(int32_t num_items, int32_t num_segments, int32_t axis_size, bool largest,
                                             std::source_location location = std::source_location::current()) -> size_t;

void LaunchTopKSmall(cudaStream_t stream, DType dtype, const TopKParameters &parameters,
                     std::source_location location = std::source_location::current());
void LaunchTopKSort(cudaStream_t stream, DType dtype, const TopKParameters &parameters, uint64_t *keys_input,
                    uint64_t *keys_output, int64_t *indices_input, int64_t *indices_output, void *workspace,
                    size_t workspace_bytes, std::source_location location = std::source_location::current());
void LaunchTopKSerial(cudaStream_t stream, DType dtype, const TopKParameters &parameters,
                      std::source_location location = std::source_location::current());

}  // namespace ttl::internal
