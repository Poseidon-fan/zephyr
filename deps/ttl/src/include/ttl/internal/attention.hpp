#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <driver_types.h>

#include "ttl/dtype.hpp"

namespace ttl::internal {

struct SdpaParameters final {
  std::byte *output_;
  const std::byte *query_;
  const std::byte *key_;
  const std::byte *value_;
  const std::byte *mask_;
  uint64_t output_strides_bytes_[4]{};
  uint64_t query_strides_bytes_[4]{};
  uint64_t key_strides_bytes_[4]{};
  uint64_t value_strides_bytes_[4]{};
  uint64_t mask_strides_bytes_[4]{};
  uint64_t batch_size_;
  uint64_t query_head_count_;
  uint64_t key_value_head_count_;
  uint64_t query_length_;
  uint64_t key_length_;
  uint64_t head_dimension_;
  uint64_t value_dimension_;
  float scale_;
  DType mask_dtype_;
  bool has_mask_;
  bool causal_;
  bool lower_right_;
};

static_assert(std::is_trivially_copyable_v<SdpaParameters>);
static_assert(std::is_standard_layout_v<SdpaParameters>);

void LaunchSdpa(cudaStream_t stream, DType dtype, const SdpaParameters &parameters,
                std::source_location location = std::source_location::current());

}  // namespace ttl::internal
