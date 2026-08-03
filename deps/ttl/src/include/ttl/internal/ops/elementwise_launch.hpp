#pragma once

#include <source_location>

#include <driver_types.h>

#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/runtime/execution/device_error.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/scalar.hpp"

namespace ttl::internal {

/** Launch a typed fill using an already validated iteration plan. */
void LaunchFill(cudaStream_t stream, DType dtype, const ElementwiseIterator &iterator, const Scalar &value,
                std::source_location location = std::source_location::current());

/** Launch a typed identity copy using an already validated two-operand iteration plan. */
void LaunchCopy(cudaStream_t stream, DType dtype, const ElementwiseIterator &iterator,
                std::source_location location = std::source_location::current());

/** Launch a checked dtype conversion using an already validated two-operand iteration plan. */
void LaunchCast(cudaStream_t stream, DType source_dtype, DType target_dtype, const ElementwiseIterator &iterator,
                const DeviceErrorLaunchContext &error_context,
                std::source_location location = std::source_location::current());

}  // namespace ttl::internal
