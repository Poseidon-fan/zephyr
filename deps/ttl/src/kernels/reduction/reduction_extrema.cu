#include "ttl/internal/ops/reduction.hpp"

#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda/std/type_traits>

#include "ttl/common/error.hpp"
#include "ttl/internal/kernels/reduction/reduction_kernel.cuh"
#include "ttl/internal/kernels/reduction/reduction_operations.cuh"
#include "ttl/internal/runtime/library/cuda_dtype.hpp"
#include "ttl/tensor/dtype.hpp"

namespace ttl::internal {
namespace {

template <CudaStorageType T, ReductionOp operation>
void LaunchOneExtrema(cudaStream_t stream, const ReductionPlan &plan, void *scratch, std::source_location location) {
  using Output =
      cuda::std::conditional_t<operation == ReductionOp::ARG_MIN || operation == ReductionOp::ARG_MAX, int64_t, T>;
  LaunchTypedReduction<T, Output>(stream, plan, scratch, ExtremaReductionOperation<T, operation>{}, location);
}

template <CudaStorageType T>
void DispatchExtremaOperation(cudaStream_t stream, ReductionOp operation, const ReductionPlan &plan, void *scratch,
                              std::source_location location) {
  switch (operation) {
    case ReductionOp::MINIMUM:
      LaunchOneExtrema<T, ReductionOp::MINIMUM>(stream, plan, scratch, location);
      return;
    case ReductionOp::MAXIMUM:
      LaunchOneExtrema<T, ReductionOp::MAXIMUM>(stream, plan, scratch, location);
      return;
    case ReductionOp::ARG_MIN:
      LaunchOneExtrema<T, ReductionOp::ARG_MIN>(stream, plan, scratch, location);
      return;
    case ReductionOp::ARG_MAX:
      LaunchOneExtrema<T, ReductionOp::ARG_MAX>(stream, plan, scratch, location);
      return;
    case ReductionOp::SUM:
    case ReductionOp::MEAN:
    case ReductionOp::ANY:
    case ReductionOp::ALL:
      break;
  }
  throw InternalError("extrema launcher received a non-extrema operation", location);
}

}  // namespace

void LaunchExtremaReduction(cudaStream_t stream, DType dtype, ReductionOp operation, const ReductionPlan &plan,
                            void *scratch, std::source_location location) {
  DispatchCudaNumericDType(
      dtype, GetReductionName(operation),
      [&]<CudaStorageType T>(std::type_identity<T>) {
        DispatchExtremaOperation<T>(stream, operation, plan, scratch, location);
      },
      location);
}

}  // namespace ttl::internal
