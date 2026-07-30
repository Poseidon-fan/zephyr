#include "ttl/internal/reduction.hpp"

#include <concepts>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/internal/cuda_dtype.hpp"
#include "ttl/internal/reduction_kernel.cuh"
#include "ttl/internal/reduction_operations.cuh"

namespace ttl::internal {

void LaunchSummationReduction(cudaStream_t stream, DType dtype, ReductionOp operation, const ReductionPlan &plan,
                              void *scratch, std::source_location location) {
  switch (operation) {
    case ReductionOp::SUM:
      DispatchCudaDType(dtype, "SumOut", [&]<CudaStorageType T>(std::type_identity<T>) {
        if constexpr (IsCudaFloatingType<T>() || std::same_as<T, int32_t> || std::same_as<T, int64_t>) {
          LaunchTypedReduction<T, T>(stream, plan, scratch, SumReductionOperation<T>{}, location);
        } else {
          throw InternalError("SumOut launch received an unsupported dtype", location);
        }
      });
      return;
    case ReductionOp::MEAN:
      DispatchCudaDType(dtype, "MeanOut", [&]<CudaStorageType T>(std::type_identity<T>) {
        if constexpr (IsCudaFloatingType<T>()) {
          LaunchTypedReduction<T, T>(stream, plan, scratch, MeanReductionOperation<T>{}, location);
        } else {
          throw InternalError("MeanOut launch received an unsupported dtype", location);
        }
      });
      return;
    case ReductionOp::MINIMUM:
    case ReductionOp::MAXIMUM:
    case ReductionOp::ARG_MIN:
    case ReductionOp::ARG_MAX:
    case ReductionOp::ANY:
    case ReductionOp::ALL:
      break;
  }
  throw InternalError("summation launcher received a non-summation operation", location);
}

}  // namespace ttl::internal
