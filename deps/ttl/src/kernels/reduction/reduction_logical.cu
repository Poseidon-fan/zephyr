#include "ttl/internal/ops/reduction.hpp"

#include <source_location>

#include <cuda_runtime.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/kernels/reduction/reduction_kernel.cuh"
#include "ttl/internal/kernels/reduction/reduction_operations.cuh"

namespace ttl::internal {

void LaunchLogicalReduction(cudaStream_t stream, ReductionOp operation, const ReductionPlan &plan, void *scratch,
                            std::source_location location) {
  switch (operation) {
    case ReductionOp::ANY:
      LaunchTypedReduction<bool, bool>(stream, plan, scratch, LogicalReductionOperation<ReductionOp::ANY>{}, location);
      return;
    case ReductionOp::ALL:
      LaunchTypedReduction<bool, bool>(stream, plan, scratch, LogicalReductionOperation<ReductionOp::ALL>{}, location);
      return;
    case ReductionOp::SUM:
    case ReductionOp::MEAN:
    case ReductionOp::MINIMUM:
    case ReductionOp::MAXIMUM:
    case ReductionOp::ARG_MIN:
    case ReductionOp::ARG_MAX:
      break;
  }
  throw InternalError("logical launcher received a non-logical operation", location);
}

}  // namespace ttl::internal
