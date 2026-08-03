#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <driver_types.h>

#include "ttl/internal/common/index_width.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {

template <typename Index>
struct CumulativeSumParameters final {
  std::byte *output_;
  const std::byte *input_;
  Index shape_[TTL_MAX_RANK]{};
  Index output_strides_bytes_[TTL_MAX_RANK]{};
  Index input_strides_bytes_[TTL_MAX_RANK]{};
  Index slice_count_;
  Index axis_size_;
  uint8_t rank_;
  uint8_t axis_;
};

using CumulativeSumParameters32 = CumulativeSumParameters<uint32_t>;
using CumulativeSumParameters64 = CumulativeSumParameters<uint64_t>;

static_assert(std::is_trivially_copyable_v<CumulativeSumParameters32>);
static_assert(std::is_standard_layout_v<CumulativeSumParameters32>);
static_assert(std::is_trivially_copyable_v<CumulativeSumParameters64>);
static_assert(std::is_standard_layout_v<CumulativeSumParameters64>);

void LaunchCumulativeSum(cudaStream_t stream, DType dtype, IndexWidth index_width,
                         const CumulativeSumParameters64 &parameters,
                         std::source_location location = std::source_location::current());

}  // namespace ttl::internal
