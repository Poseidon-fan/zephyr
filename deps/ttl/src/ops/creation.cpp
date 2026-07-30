#include "ttl/ops/creation.hpp"

#include <bit>
#include <cstdint>
#include <source_location>

#include <cuda_runtime_api.h>

#include "ttl/dtype.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/internal/elementwise_launch.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/scalar.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

[[nodiscard]] auto HasZeroBitPattern(const Scalar &value, DType dtype, std::source_location location) -> bool {
  switch (dtype) {
    case DType::BOOL:
      return !value.Cast<bool>(location);
    case DType::UINT8:
      return value.Cast<uint8_t>(location) == 0;
    case DType::INT32:
      return value.Cast<int32_t>(location) == 0;
    case DType::INT64:
      return value.Cast<int64_t>(location) == 0;
    case DType::FLOAT16:
      return value.Cast<Float16>(location).bits_ == 0;
    case DType::BFLOAT16:
      return value.Cast<BFloat16>(location).bits_ == 0;
    case DType::FLOAT32:
      return std::bit_cast<uint32_t>(value.Cast<float>(location)) == 0;
  }
  throw InternalError("FillOut received an invalid dtype after tensor validation", location);
}

}  // namespace

void FillOut(ExecutionContext &context, Tensor &output, Scalar value, std::source_location location) {
  internal::OpGuard guard{context, "FillOut", location};
  guard.ValidateTensor(output);

  auto iterator = internal::ElementwiseIterator::Builder{}
                      .AddOutput(output)
                      .SetAliasPolicy(internal::AliasPolicy::NO_ALIAS)
                      .Build("FillOut", location);
  const auto has_zero_bit_pattern = HasZeroBitPattern(value, output.GetDType(), location);
  const auto use_memset = output.IsContiguous() && has_zero_bit_pattern;
  if (output.GetNumElements() == 0) {
    return;
  }

  guard.RecordTensor(output);
  if (use_memset) {
    const auto bytes =
        internal::CheckedBytes(output.GetNumElements(), GetDTypeSize(output.GetDType(), location), location);
    internal::CheckCuda(internal::GetCudaApi().memset_async_(internal::TensorAccess::GetMutableData(output, location),
                                                             0, bytes, guard.GetNativeStream()),
                        "cudaMemsetAsync (FillOut)", location);
  } else {
    internal::LaunchFill(guard.GetNativeStream(), output.GetDType(), iterator, value, location);
  }
  guard.CheckLaunch();
}

}  // namespace ttl
