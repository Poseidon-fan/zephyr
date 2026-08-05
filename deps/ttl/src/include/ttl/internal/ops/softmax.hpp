#pragma once

#include <cstdint>
#include <source_location>

#include <driver_types.h>

#include "ttl/internal/ops/rowwise.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl::internal {

enum class SoftmaxOp : uint8_t {
  SOFTMAX,
  LOG_SOFTMAX,
};

/** Launch the rowwise plan as softmax or log-softmax, using scratch only for a partitioned plan. */
void LaunchSoftmax(cudaStream_t stream, DType dtype, SoftmaxOp operation, const RowwisePlan &plan, void *scratch,
                   std::source_location location);

}  // namespace ttl::internal
