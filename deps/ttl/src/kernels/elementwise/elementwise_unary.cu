#include "ttl/internal/ops/elementwise_ops.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <source_location>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda/std/type_traits>

#include "ttl/common/error.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_apply.cuh"
#include "ttl/internal/kernels/elementwise/elementwise_math.cuh"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/runtime/library/cuda_dtype.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/scalar.hpp"

namespace ttl::internal {
namespace {

[[nodiscard]] __device__ auto StableSigmoid(float value) -> float {
  if (value >= 0.0F) {
    return 1.0F / (1.0F + expf(-value));
  }
  const auto exponential = expf(value);
  return exponential / (1.0F + exponential);
}

template <CudaStorageType T, UnaryElementwiseOp operation>
struct UnaryOperation final {
  __device__ auto operator()(T input, int64_t /*index*/) const -> ElementwiseResult<T> {
    if constexpr (operation == UnaryElementwiseOp::NEGATE) {
      if constexpr (IsCudaFloatingType<T>()) {
        return {.value_ = FromElementwiseFloat<T>(-ToElementwiseFloat(input))};
      } else {
        using Unsigned = cuda::std::make_unsigned_t<T>;
        const auto bits = Unsigned{0} - static_cast<Unsigned>(input);
        return {.value_ = __builtin_bit_cast(T, bits)};
      }
    } else if constexpr (operation == UnaryElementwiseOp::ABS) {
      if constexpr (IsCudaFloatingType<T>()) {
        return {.value_ = FromElementwiseFloat<T>(fabsf(ToElementwiseFloat(input)))};
      } else {
        using Unsigned = cuda::std::make_unsigned_t<T>;
        const auto bits = input < 0 ? Unsigned{0} - static_cast<Unsigned>(input) : static_cast<Unsigned>(input);
        return {.value_ = __builtin_bit_cast(T, bits)};
      }
    } else if constexpr (operation == UnaryElementwiseOp::RELU && !IsCudaFloatingType<T>()) {
      return {.value_ = input > 0 ? input : T{0}};
    } else if constexpr (operation == UnaryElementwiseOp::EXP) {
      const auto value = ToElementwiseFloat(input);
      return {.value_ = FromElementwiseFloat<T>(expf(value))};
    } else if constexpr (operation == UnaryElementwiseOp::LOG) {
      const auto value = ToElementwiseFloat(input);
      return {.value_ = FromElementwiseFloat<T>(logf(value))};
    } else if constexpr (operation == UnaryElementwiseOp::SQRT) {
      const auto value = ToElementwiseFloat(input);
      return {.value_ = FromElementwiseFloat<T>(sqrtf(value))};
    } else if constexpr (operation == UnaryElementwiseOp::RSQRT) {
      const auto value = ToElementwiseFloat(input);
      return {.value_ = FromElementwiseFloat<T>(rsqrtf(value))};
    } else if constexpr (operation == UnaryElementwiseOp::SIN) {
      const auto value = ToElementwiseFloat(input);
      return {.value_ = FromElementwiseFloat<T>(sinf(value))};
    } else if constexpr (operation == UnaryElementwiseOp::COS) {
      const auto value = ToElementwiseFloat(input);
      return {.value_ = FromElementwiseFloat<T>(cosf(value))};
    } else if constexpr (operation == UnaryElementwiseOp::TANH) {
      const auto value = ToElementwiseFloat(input);
      return {.value_ = FromElementwiseFloat<T>(tanhf(value))};
    } else if constexpr (operation == UnaryElementwiseOp::SIGMOID) {
      const auto value = ToElementwiseFloat(input);
      return {.value_ = FromElementwiseFloat<T>(StableSigmoid(value))};
    } else if constexpr (operation == UnaryElementwiseOp::RELU) {
      const auto value = ToElementwiseFloat(input);
      return {.value_ = isnan(value) ? input : FromElementwiseFloat<T>(value > 0.0F ? value : 0.0F)};
    } else {
      static_assert(operation == UnaryElementwiseOp::SILU);
      const auto value = ToElementwiseFloat(input);
      return {.value_ = FromElementwiseFloat<T>(value * StableSigmoid(value))};
    }
  }
};

template <CudaStorageType T, GeluApproximation approximation>
struct GeluOperation final {
  __device__ auto operator()(T input, int64_t /*index*/) const -> ElementwiseResult<T> {
    constexpr auto sqrt_one_half = 0.70710678118654752440F;
    constexpr auto sqrt_two_over_pi = 0.79788456080286535588F;
    constexpr auto kappa = 0.044715F;
    const auto value = ToElementwiseFloat(input);
    if constexpr (approximation == GeluApproximation::NONE) {
      return {.value_ = FromElementwiseFloat<T>(0.5F * value * (1.0F + erff(value * sqrt_one_half)))};
    } else {
      const auto cube = value * value * value;
      const auto inner = sqrt_two_over_pi * (value + kappa * cube);
      return {.value_ = FromElementwiseFloat<T>(0.5F * value * (1.0F + tanhf(inner)))};
    }
  }
};

template <CudaStorageType T>
struct ClampOperation final {
  float minimum_;
  float maximum_;
  bool has_minimum_;
  bool has_maximum_;

  __device__ auto operator()(T input, int64_t /*index*/) const -> ElementwiseResult<T> {
    const auto value = ToElementwiseFloat(input);
    if (isnan(value)) {
      return {.value_ = input};
    }
    if (has_minimum_ && value < minimum_) {
      return {.value_ = FromElementwiseFloat<T>(minimum_)};
    }
    if (has_maximum_ && value > maximum_) {
      return {.value_ = FromElementwiseFloat<T>(maximum_)};
    }
    return {.value_ = input};
  }
};

template <CudaStorageType T, UnaryElementwiseOp operation>
void LaunchUnaryOperation(cudaStream_t stream, const ElementwiseIterator &iterator, std::source_location location) {
  LaunchUnary<T, T>(stream, iterator, UnaryOperation<T, operation>{}, location);
}

template <CudaStorageType T>
void DispatchUnaryOperation(cudaStream_t stream, UnaryElementwiseOp operation, const ElementwiseIterator &iterator,
                            std::source_location location) {
  switch (operation) {
    case UnaryElementwiseOp::NEGATE:
      LaunchUnaryOperation<T, UnaryElementwiseOp::NEGATE>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::ABS:
      LaunchUnaryOperation<T, UnaryElementwiseOp::ABS>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::EXP:
      LaunchUnaryOperation<T, UnaryElementwiseOp::EXP>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::LOG:
      LaunchUnaryOperation<T, UnaryElementwiseOp::LOG>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::SQRT:
      LaunchUnaryOperation<T, UnaryElementwiseOp::SQRT>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::RSQRT:
      LaunchUnaryOperation<T, UnaryElementwiseOp::RSQRT>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::SIN:
      LaunchUnaryOperation<T, UnaryElementwiseOp::SIN>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::COS:
      LaunchUnaryOperation<T, UnaryElementwiseOp::COS>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::TANH:
      LaunchUnaryOperation<T, UnaryElementwiseOp::TANH>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::SIGMOID:
      LaunchUnaryOperation<T, UnaryElementwiseOp::SIGMOID>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::RELU:
      LaunchUnaryOperation<T, UnaryElementwiseOp::RELU>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::SILU:
      LaunchUnaryOperation<T, UnaryElementwiseOp::SILU>(stream, iterator, location);
      return;
  }
  throw InternalError("invalid unary elementwise operation", location);
}

template <CudaStorageType T>
void DispatchSignedUnaryOperation(cudaStream_t stream, UnaryElementwiseOp operation,
                                  const ElementwiseIterator &iterator, std::source_location location) {
  switch (operation) {
    case UnaryElementwiseOp::NEGATE:
      LaunchUnaryOperation<T, UnaryElementwiseOp::NEGATE>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::ABS:
      LaunchUnaryOperation<T, UnaryElementwiseOp::ABS>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::RELU:
      LaunchUnaryOperation<T, UnaryElementwiseOp::RELU>(stream, iterator, location);
      return;
    case UnaryElementwiseOp::EXP:
    case UnaryElementwiseOp::LOG:
    case UnaryElementwiseOp::SQRT:
    case UnaryElementwiseOp::RSQRT:
    case UnaryElementwiseOp::SIN:
    case UnaryElementwiseOp::COS:
    case UnaryElementwiseOp::TANH:
    case UnaryElementwiseOp::SIGMOID:
    case UnaryElementwiseOp::SILU:
      throw InternalError("signed integer unary launch received a floating-point operation", location);
  }
  throw InternalError("invalid signed integer unary operation", location);
}

template <CudaStorageType T>
void LaunchTypedGelu(cudaStream_t stream, GeluApproximation approximation, const ElementwiseIterator &iterator,
                     std::source_location location) {
  switch (approximation) {
    case GeluApproximation::NONE:
      LaunchUnary<T, T>(stream, iterator, GeluOperation<T, GeluApproximation::NONE>{}, location);
      return;
    case GeluApproximation::TANH:
      LaunchUnary<T, T>(stream, iterator, GeluOperation<T, GeluApproximation::TANH>{}, location);
      return;
  }
  throw InternalError("invalid Gelu approximation reached the CUDA launcher", location);
}

template <CudaStorageType T>
void LaunchTypedClamp(cudaStream_t stream, const ElementwiseIterator &iterator, const std::optional<Scalar> &minimum,
                      const std::optional<Scalar> &maximum, std::source_location location) {
  const auto minimum_value =
      minimum.has_value() ? ToElementwiseFloat(ConvertElementwiseScalar<T>(*minimum, location)) : 0.0F;
  const auto maximum_value =
      maximum.has_value() ? ToElementwiseFloat(ConvertElementwiseScalar<T>(*maximum, location)) : 0.0F;
  LaunchUnary<T, T>(stream, iterator,
                    ClampOperation<T>{
                        .minimum_ = minimum_value,
                        .maximum_ = maximum_value,
                        .has_minimum_ = minimum.has_value(),
                        .has_maximum_ = maximum.has_value(),
                    },
                    location);
}

void ValidateUnaryLaunch(cudaStream_t stream, const ElementwiseIterator &iterator, std::source_location location) {
  if (stream == nullptr || iterator.GetOperandCount() != 2 || iterator.GetNumElements() <= 0) {
    throw InternalError("invalid unary elementwise launch plan", location);
  }
}

}  // namespace

void LaunchUnaryElementwise(cudaStream_t stream, DType dtype, UnaryElementwiseOp operation,
                            const ElementwiseIterator &iterator, std::source_location location) {
  ValidateUnaryLaunch(stream, iterator, location);
  DispatchCudaDType(dtype, "unary elementwise", [&]<CudaStorageType T>(std::type_identity<T>) {
    if constexpr (IsCudaFloatingType<T>()) {
      DispatchUnaryOperation<T>(stream, operation, iterator, location);
    } else if constexpr (IsCudaSignedIntegerType<T>()) {
      DispatchSignedUnaryOperation<T>(stream, operation, iterator, location);
    } else {
      throw InternalError("unary elementwise launch received an unsupported dtype", location);
    }
  });
}

void LaunchGeluElementwise(cudaStream_t stream, DType dtype, GeluApproximation approximation,
                           const ElementwiseIterator &iterator, std::source_location location) {
  ValidateUnaryLaunch(stream, iterator, location);
  DispatchCudaDType(dtype, "GeluOut", [&]<CudaStorageType T>(std::type_identity<T>) {
    if constexpr (IsCudaFloatingType<T>()) {
      LaunchTypedGelu<T>(stream, approximation, iterator, location);
    } else {
      throw InternalError("Gelu launch received a non-floating dtype", location);
    }
  });
}

void LaunchClampElementwise(cudaStream_t stream, DType dtype, const ElementwiseIterator &iterator,
                            const std::optional<Scalar> &minimum, const std::optional<Scalar> &maximum,
                            std::source_location location) {
  ValidateUnaryLaunch(stream, iterator, location);
  DispatchCudaDType(dtype, "ClampOut", [&]<CudaStorageType T>(std::type_identity<T>) {
    if constexpr (IsCudaFloatingType<T>()) {
      LaunchTypedClamp<T>(stream, iterator, minimum, maximum, location);
    } else {
      throw InternalError("Clamp launch received a non-floating dtype", location);
    }
  });
}

}  // namespace ttl::internal
