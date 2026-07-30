#include "ttl/ops/copy.hpp"

#include <source_location>

#include <cuda_runtime_api.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/internal/elementwise_launch.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

void CopyOutImpl(internal::OpGuard &guard, Tensor &output, const Tensor &input, bool require_contiguous_output,
                 std::source_location location) {
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  if (output.GetShape() != input.GetShape()) {
    throw InvalidArgumentError("CopyOut requires equal input and output shapes", location);
  }
  if (output.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("CopyOut requires equal input and output dtypes", location);
  }
  if (require_contiguous_output && !output.IsContiguous()) {
    throw InvalidArgumentError("ContiguousOut requires a canonical contiguous output", location);
  }

  auto iterator = internal::ElementwiseIterator::Builder{}
                      .AddOutput(output)
                      .AddInput(input)
                      .SetAliasPolicy(internal::AliasPolicy::COPY)
                      .SetRequireSameDType(true)
                      .Build(require_contiguous_output ? "ContiguousOut" : "CopyOut", location);
  if (ClassifyAlias(output, input, location) == AliasKind::EXACT || output.GetNumElements() == 0) {
    return;
  }

  guard.RecordTensor(output);
  guard.RecordTensor(input);
  if (output.IsContiguous() && input.IsContiguous()) {
    const auto bytes =
        internal::CheckedBytes(output.GetNumElements(), GetDTypeSize(output.GetDType(), location), location);
    internal::CheckCuda(internal::GetCudaApi().memcpy_async_(internal::TensorAccess::GetMutableData(output, location),
                                                             internal::TensorAccess::GetData(input, location), bytes,
                                                             cudaMemcpyDeviceToDevice, guard.GetNativeStream()),
                        "cudaMemcpyAsync (CopyOut)", location);
  } else {
    internal::LaunchCopy(guard.GetNativeStream(), output.GetDType(), iterator, location);
  }
  guard.CheckLaunch();
}

}  // namespace

void CopyOut(ExecutionContext &context, Tensor &output, const Tensor &input, std::source_location location) {
  internal::OpGuard guard{context, "CopyOut", location};
  CopyOutImpl(guard, output, input, false, location);
}

auto Clone(ExecutionContext &context, const Tensor &input, std::source_location location) -> Tensor {
  const auto &input_impl = internal::TensorAccess::GetImpl(input, location);
  {
    internal::OpGuard guard{context, "Clone", location};
    guard.ValidateTensor(input);
  }
  auto output = Empty(context, input_impl.GetShape(), input_impl.GetDType(), location);
  CopyOut(context, output, input, location);
  return output;
}

void ContiguousOut(ExecutionContext &context, Tensor &output, const Tensor &input, std::source_location location) {
  internal::OpGuard guard{context, "ContiguousOut", location};
  CopyOutImpl(guard, output, input, true, location);
}

auto Contiguous(ExecutionContext &context, const Tensor &input, std::source_location location) -> Tensor {
  const auto &input_impl = internal::TensorAccess::GetImpl(input, location);
  if (input_impl.HasFlag(internal::TensorFlag::CONTIGUOUS)) {
    internal::OpGuard guard{context, "Contiguous", location};
    guard.ValidateTensor(input);
    return input;
  }

  {
    internal::OpGuard guard{context, "Contiguous", location};
    guard.ValidateTensor(input);
  }
  auto output = Empty(context, input_impl.GetShape(), input_impl.GetDType(), location);
  ContiguousOut(context, output, input, location);
  return output;
}

}  // namespace ttl
