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

/**
 * @brief Trivially copyable host-to-kernel ABI for a strided composition copy.
 *
 * shape is the common logical iteration space and both stride arrays contain byte offsets. The host selects the
 * narrowest index width that can represent every reachable offset and the element count.
 */
template <typename Index>
struct CompositionParameters final {
  std::byte *output_;
  const std::byte *input_;
  Index shape_[TTL_MAX_RANK]{};
  Index output_strides_bytes_[TTL_MAX_RANK]{};
  Index input_strides_bytes_[TTL_MAX_RANK]{};
  Index num_elements_;
  uint8_t rank_;
};

using CompositionParameters32 = CompositionParameters<uint32_t>;
using CompositionParameters64 = CompositionParameters<uint64_t>;

static_assert(std::is_trivially_copyable_v<CompositionParameters32>);
static_assert(std::is_standard_layout_v<CompositionParameters32>);
static_assert(std::is_trivially_copyable_v<CompositionParameters64>);
static_assert(std::is_standard_layout_v<CompositionParameters64>);

[[nodiscard]] auto GetCompositionIndexWidth(const CompositionParameters64 &parameters) noexcept -> IndexWidth;

void LaunchCompositionCopy(cudaStream_t stream, DType dtype, IndexWidth index_width,
                           const CompositionParameters64 &parameters,
                           std::source_location location = std::source_location::current());

}  // namespace ttl::internal
