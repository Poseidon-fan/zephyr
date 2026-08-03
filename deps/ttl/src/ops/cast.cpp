#include "ttl/ops/cast.hpp"

#include <source_location>

#include "ttl/common/error.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/elementwise_launch.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

void CastOut(ExecutionContext &context, Tensor &output, const Tensor &input, std::source_location location) {
  auto same_dtype = false;
  {
    internal::OpGuard guard{context, "CastOut", location, internal::CapturePolicy::SAFE};
    guard.ValidateTensor(output);
    guard.ValidateTensor(input);
    if (output.GetShape() != input.GetShape()) {
      throw InvalidArgumentError("CastOut requires equal input and output shapes", location);
    }
    same_dtype = output.GetDType() == input.GetDType();
    if (!same_dtype) {
      auto iterator = internal::ElementwiseIterator::Builder{}
                          .AddOutput(output)
                          .AddInput(input)
                          .SetAliasPolicy(internal::AliasPolicy::NO_ALIAS)
                          .Build("CastOut", location);
      if (output.GetNumElements() == 0) {
        return;
      }

      const auto error_context = guard.RegisterDeviceError(input.GetDType(), output.GetDType());
      guard.RecordTensor(output);
      guard.RecordTensor(input);
      internal::LaunchCast(guard.GetNativeStream(), input.GetDType(), output.GetDType(), iterator, error_context,
                           location);
      guard.CheckLaunch();
      return;
    }
  }
  CopyOut(context, output, input, location);
}

auto Cast(ExecutionContext &context, const Tensor &input, DType dtype, std::source_location location) -> Tensor {
  const auto &input_impl = internal::TensorAccess::GetImpl(input, location);
  const auto target_dtype = GetDTypeInfo(dtype, location).dtype_;
  {
    internal::OpGuard guard{context, "Cast", location};
    guard.ValidateTensor(input);
  }
  auto output = Empty(context, input_impl.GetShape(), target_dtype, location);
  CastOut(context, output, input, location);
  return output;
}

}  // namespace ttl
