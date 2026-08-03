#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <driver_types.h>

#include "ttl/internal/ops/rowwise.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {

enum class NormalizationOp : uint8_t {
  LAYER_NORM,
  RMS_NORM,
};

template <typename Index>
struct NormalizationAuxiliaryParameters final {
  const std::byte *weight_;
  const std::byte *bias_;
  Index weight_strides_bytes_[TTL_MAX_RANK]{};
  Index bias_strides_bytes_[TTL_MAX_RANK]{};
  float epsilon_;
};

using NormalizationAuxiliaryParameters64 = NormalizationAuxiliaryParameters<uint64_t>;

static_assert(std::is_trivially_copyable_v<NormalizationAuxiliaryParameters64>);
static_assert(std::is_standard_layout_v<NormalizationAuxiliaryParameters64>);

void LaunchNormalization(cudaStream_t stream, DType dtype, NormalizationOp operation, const RowwisePlan &plan,
                         const NormalizationAuxiliaryParameters64 &auxiliary, void *scratch,
                         std::source_location location);

}  // namespace ttl::internal
