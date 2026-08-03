#pragma once

#include <cstdint>
#include <source_location>

#include <driver_types.h>

#include "ttl/tensor/dtype.hpp"

namespace ttl::internal {

struct ArangeParameters final {
  void *output_;
  int64_t num_elements_;
  int64_t integer_start_;
  int64_t integer_step_;
  double floating_start_;
  double floating_step_;
};

/** Launch an already validated contiguous arange operation. */
void LaunchArange(cudaStream_t stream, DType dtype, const ArangeParameters &parameters,
                  std::source_location location = std::source_location::current());

}  // namespace ttl::internal
