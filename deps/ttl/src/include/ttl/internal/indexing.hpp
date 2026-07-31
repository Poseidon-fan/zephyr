#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <driver_types.h>

#include "ttl/dtype.hpp"
#include "ttl/internal/device_error.hpp"
#include "ttl/internal/index_width.hpp"
#include "ttl/shape.hpp"

namespace ttl::internal {

template <typename Index>
struct IndexingParameters final {
  std::byte *output_;
  const std::byte *input_;
  const std::byte *indices_;
  Index shape_[TTL_MAX_RANK]{};
  Index output_strides_bytes_[TTL_MAX_RANK]{};
  Index input_strides_bytes_[TTL_MAX_RANK]{};
  Index index_strides_bytes_[TTL_MAX_RANK]{};
  Index num_elements_;
  int64_t axis_bound_;
  uint8_t rank_;
  uint8_t axis_;
};

template <typename Index>
struct GatherRowsParameters final {
  std::byte *output_;
  const std::byte *table_;
  const std::byte *indices_;
  Index output_shape_[TTL_MAX_RANK]{};
  Index output_strides_bytes_[TTL_MAX_RANK]{};
  Index table_tail_strides_bytes_[TTL_MAX_RANK]{};
  Index index_strides_bytes_[TTL_MAX_RANK]{};
  Index table_row_stride_bytes_;
  Index num_elements_;
  int64_t row_count_;
  uint8_t output_rank_;
  uint8_t index_rank_;
};

using IndexingParameters32 = IndexingParameters<uint32_t>;
using IndexingParameters64 = IndexingParameters<uint64_t>;
using GatherRowsParameters32 = GatherRowsParameters<uint32_t>;
using GatherRowsParameters64 = GatherRowsParameters<uint64_t>;

static_assert(std::is_trivially_copyable_v<IndexingParameters32>);
static_assert(std::is_standard_layout_v<IndexingParameters32>);
static_assert(std::is_trivially_copyable_v<IndexingParameters64>);
static_assert(std::is_standard_layout_v<IndexingParameters64>);
static_assert(std::is_trivially_copyable_v<GatherRowsParameters32>);
static_assert(std::is_standard_layout_v<GatherRowsParameters32>);
static_assert(std::is_trivially_copyable_v<GatherRowsParameters64>);
static_assert(std::is_standard_layout_v<GatherRowsParameters64>);

void LaunchIndexing(cudaStream_t stream, DType dtype, DType index_dtype, IndexWidth index_width,
                    const IndexingParameters64 &parameters, const DeviceErrorLaunchContext &error_context,
                    std::source_location location);
void LaunchGatherRows(cudaStream_t stream, DType dtype, DType index_dtype, IndexWidth index_width,
                      const GatherRowsParameters64 &parameters, const DeviceErrorLaunchContext &error_context,
                      std::source_location location);

}  // namespace ttl::internal
