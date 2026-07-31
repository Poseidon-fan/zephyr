#pragma once

#include <cstdint>
#include <source_location>

#include <driver_types.h>

#include "ttl/dtype.hpp"
#include "ttl/internal/rowwise.hpp"

namespace ttl::internal {

enum class SoftmaxOp : uint8_t {
  SOFTMAX,
  LOG_SOFTMAX,
};

void LaunchSoftmax(cudaStream_t stream, DType dtype, SoftmaxOp operation, const RowwisePlan &plan, void *scratch,
                   std::source_location location);

}  // namespace ttl::internal
