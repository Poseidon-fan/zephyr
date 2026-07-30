#include "ttl/internal/elementwise_ops.hpp"

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/internal/cuda_dtype.hpp"
#include "ttl/internal/elementwise_apply.cuh"
#include "ttl/internal/elementwise_iterator.hpp"

namespace ttl::internal {
namespace {

template <LogicalElementwiseOp operation>
struct LogicalOperation final {
  __device__ auto operator()(bool lhs, bool rhs, int64_t /*index*/) const -> ElementwiseResult<bool> {
    if constexpr (operation == LogicalElementwiseOp::AND) {
      return {.value_ = lhs && rhs};
    } else {
      static_assert(operation == LogicalElementwiseOp::OR);
      return {.value_ = lhs || rhs};
    }
  }
};

struct LogicalNotOperation final {
  __device__ auto operator()(bool input, int64_t /*index*/) const -> ElementwiseResult<bool> {
    return {.value_ = !input};
  }
};

template <CudaStorageType T>
struct WhereOperation final {
  __device__ auto operator()(bool condition, T true_value, T false_value, int64_t /*index*/) const
      -> ElementwiseResult<T> {
    return {.value_ = condition ? true_value : false_value};
  }
};

void ValidateLogicalLaunch(cudaStream_t stream, const ElementwiseIterator &iterator, size_t operand_count,
                           std::source_location location) {
  if (stream == nullptr || iterator.GetOperandCount() != operand_count || iterator.GetNumElements() <= 0) {
    throw InternalError("invalid logical elementwise launch plan", location);
  }
}

}  // namespace

void LaunchLogicalElementwise(cudaStream_t stream, LogicalElementwiseOp operation, const ElementwiseIterator &iterator,
                              std::source_location location) {
  const auto operand_count = operation == LogicalElementwiseOp::NOT ? size_t{2} : size_t{3};
  ValidateLogicalLaunch(stream, iterator, operand_count, location);
  switch (operation) {
    case LogicalElementwiseOp::AND:
      LaunchBinary<bool, bool, bool>(stream, iterator, LogicalOperation<LogicalElementwiseOp::AND>{}, location);
      return;
    case LogicalElementwiseOp::OR:
      LaunchBinary<bool, bool, bool>(stream, iterator, LogicalOperation<LogicalElementwiseOp::OR>{}, location);
      return;
    case LogicalElementwiseOp::NOT:
      LaunchUnary<bool, bool>(stream, iterator, LogicalNotOperation{}, location);
      return;
  }
  throw InternalError("invalid logical elementwise operation", location);
}

void LaunchWhereElementwise(cudaStream_t stream, DType value_dtype, const ElementwiseIterator &iterator,
                            std::source_location location) {
  ValidateLogicalLaunch(stream, iterator, 4, location);
  DispatchCudaDType(value_dtype, "WhereOut", [&]<CudaStorageType T>(std::type_identity<T>) {
    LaunchTernary<T, bool, T, T>(stream, iterator, WhereOperation<T>{}, location);
  });
}

}  // namespace ttl::internal
