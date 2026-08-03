#include "ttl/ops/creation.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <source_location>

#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/ops/creation.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/elementwise_launch.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/scalar.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

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

[[nodiscard]] auto GetIntegerArangeLength(int64_t start, int64_t end, int64_t step, std::source_location location)
    -> int64_t {
  if (step == 0) {
    throw InvalidArgumentError("Arange step must not be zero", location);
  }
  if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
    return 0;
  }

  const auto distance = step > 0 ? static_cast<uint64_t>(end) - static_cast<uint64_t>(start)
                                 : static_cast<uint64_t>(start) - static_cast<uint64_t>(end);
  const auto step_magnitude = step > 0 ? static_cast<uint64_t>(step) : uint64_t{0} - static_cast<uint64_t>(step);
  const auto length = (distance / step_magnitude) + static_cast<uint64_t>(distance % step_magnitude != 0);
  return internal::CheckedNarrow<int64_t>(length, "Arange element count", location);
}

[[nodiscard]] auto GetFloatingArangeLength(double start, double end, double step, std::source_location location)
    -> int64_t {
  if (!std::isfinite(start) || !std::isfinite(end) || !std::isfinite(step)) {
    throw InvalidArgumentError("floating Arange start, end, and step must be finite", location);
  }
  if (step == 0.0) {
    throw InvalidArgumentError("Arange step must not be zero", location);
  }
  if ((step > 0.0 && start >= end) || (step < 0.0 && start <= end)) {
    return 0;
  }

  const auto length = std::ceil((end - start) / step);
  if (!std::isfinite(length) || length > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw OverflowError("Arange element count exceeds int64 range", location);
  }
  return static_cast<int64_t>(length);
}

[[nodiscard]] auto GetLastIntegerArangeValue(int64_t start, int64_t step, int64_t num_elements) noexcept -> int64_t {
  const auto index = static_cast<uint64_t>(num_elements - 1);
  return std::bit_cast<int64_t>(static_cast<uint64_t>(start) + (index * static_cast<uint64_t>(step)));
}

void ValidateInt32Arange(int64_t start, int64_t step, int64_t num_elements, std::source_location location) {
  if (num_elements == 0) {
    return;
  }
  static_cast<void>(internal::CheckedNarrow<int32_t>(start, "Arange start", location));
  static_cast<void>(
      internal::CheckedNarrow<int32_t>(GetLastIntegerArangeValue(start, step, num_elements), "Arange end", location));
}

}  // namespace

auto EmptyLike(ExecutionContext &context, const Tensor &input, std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, "EmptyLike", location};
    guard.ValidateTensor(input);
  }
  const auto &input_impl = internal::TensorAccess::GetImpl(input, location);
  return Empty(context, input_impl.GetShape(), input_impl.GetDType(), location);
}

auto Full(ExecutionContext &context, const Shape &shape, Scalar value, DType dtype, std::source_location location)
    -> Tensor {
  auto output = Empty(context, shape, dtype, location);
  FillOut(context, output, value, location);
  return output;
}

auto Zeros(ExecutionContext &context, const Shape &shape, DType dtype, std::source_location location) -> Tensor {
  return Full(context, shape, Scalar{int64_t{0}}, dtype, location);
}

auto Ones(ExecutionContext &context, const Shape &shape, DType dtype, std::source_location location) -> Tensor {
  return Full(context, shape, Scalar{int64_t{1}}, dtype, location);
}

auto Arange(ExecutionContext &context, Scalar start, Scalar end, Scalar step, DType dtype,
            std::source_location location) -> Tensor {
  {
    internal::OpGuard guard{context, "Arange", location};
  }
  auto parameters = internal::ArangeParameters{};
  switch (dtype) {
    case DType::INT32:
    case DType::INT64:
      parameters.integer_start_ = start.Cast<int64_t>(location);
      parameters.integer_step_ = step.Cast<int64_t>(location);
      parameters.num_elements_ = GetIntegerArangeLength(parameters.integer_start_, end.Cast<int64_t>(location),
                                                        parameters.integer_step_, location);
      if (dtype == DType::INT32) {
        ValidateInt32Arange(parameters.integer_start_, parameters.integer_step_, parameters.num_elements_, location);
      }
      break;
    case DType::FLOAT32:
      parameters.floating_start_ = start.ToDouble();
      parameters.floating_step_ = step.ToDouble();
      parameters.num_elements_ =
          GetFloatingArangeLength(parameters.floating_start_, end.ToDouble(), parameters.floating_step_, location);
      break;
    case DType::BOOL:
    case DType::UINT8:
    case DType::FLOAT16:
    case DType::BFLOAT16:
      throw InvalidArgumentError("Arange supports only INT32, INT64, and FLOAT32", location);
  }

  auto output = Empty(context, Shape{parameters.num_elements_}, dtype, location);
  if (parameters.num_elements_ == 0) {
    return output;
  }

  internal::OpGuard guard{context, "Arange", location};
  parameters.output_ = internal::TensorAccess::GetMutableData(output, location);
  guard.RecordTensor(output);
  internal::LaunchArange(guard.GetNativeStream(), dtype, parameters, location);
  guard.CheckLaunch();
  return output;
}

void FillOut(ExecutionContext &context, Tensor &output, Scalar value, std::source_location location) {
  internal::OpGuard guard{context, "FillOut", location, internal::CapturePolicy::SAFE};
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
