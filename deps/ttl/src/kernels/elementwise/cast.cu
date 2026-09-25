#include "ttl/internal/ops/elementwise_launch.hpp"

#include <cstdint>
#include <source_location>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ttl/runtime/cuda_dtype.hpp>
#include "ttl/common/error.hpp"
#include "ttl/internal/kernels/elementwise/elementwise_kernel.cuh"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/runtime/execution/device_error.cuh"
#include "ttl/tensor/dtype.hpp"

namespace ttl::internal {
namespace {

template <typename T>
struct CastResult final {
  T value_;
  bool valid_;
};

template <CudaStorageType T>
__device__ auto ToFloat(T value) -> float {
  if constexpr (CUDA_DTYPE_OF<T> == DType::FLOAT16) {
    return __half2float(value);
  } else if constexpr (CUDA_DTYPE_OF<T> == DType::BFLOAT16) {
    return __bfloat162float(value);
  } else {
    static_assert(CUDA_DTYPE_OF<T> == DType::FLOAT32);
    return value;
  }
}

template <CudaStorageType T>
__device__ auto ToInt64(T value) -> int64_t {
  static_assert(CUDA_DTYPE_OF<T> == DType::BOOL || CUDA_DTYPE_OF<T> == DType::UINT8 ||
                CUDA_DTYPE_OF<T> == DType::INT32 || CUDA_DTYPE_OF<T> == DType::INT64);
  return static_cast<int64_t>(value);
}

template <CudaStorageType T>
__device__ auto GetValueBits(T value) -> uint64_t {
  if constexpr (CUDA_DTYPE_OF<T> == DType::BOOL) {
    return value ? uint64_t{1} : uint64_t{0};
  } else if constexpr (CUDA_DTYPE_OF<T> == DType::UINT8) {
    return value;
  } else if constexpr (CUDA_DTYPE_OF<T> == DType::INT32) {
    return static_cast<uint32_t>(value);
  } else if constexpr (CUDA_DTYPE_OF<T> == DType::INT64) {
    return static_cast<uint64_t>(value);
  } else if constexpr (CUDA_DTYPE_OF<T> == DType::FLOAT16) {
    return __half_as_ushort(value);
  } else if constexpr (CUDA_DTYPE_OF<T> == DType::BFLOAT16) {
    return __bfloat16_as_ushort(value);
  } else {
    static_assert(CUDA_DTYPE_OF<T> == DType::FLOAT32);
    return __float_as_uint(value);
  }
}

template <CudaStorageType Target>
__device__ auto ConvertInteger(int64_t value) -> CastResult<Target> {
  if constexpr (CUDA_DTYPE_OF<Target> == DType::UINT8) {
    const auto valid = value >= 0 && value <= UINT8_MAX;
    return {.value_ = valid ? static_cast<uint8_t>(value) : uint8_t{0}, .valid_ = valid};
  } else if constexpr (CUDA_DTYPE_OF<Target> == DType::INT32) {
    const auto valid = value >= INT32_MIN && value <= INT32_MAX;
    return {.value_ = valid ? static_cast<int32_t>(value) : int32_t{0}, .valid_ = valid};
  } else {
    static_assert(CUDA_DTYPE_OF<Target> == DType::INT64);
    return {.value_ = value, .valid_ = true};
  }
}

template <CudaStorageType Target>
__device__ auto ConvertFloatingToInteger(float value) -> CastResult<Target> {
  if (!isfinite(value)) {
    return {.value_ = Target{}, .valid_ = false};
  }

  if constexpr (CUDA_DTYPE_OF<Target> == DType::UINT8) {
    const auto valid = value > -1.0F && value < 256.0F;
    return {.value_ = valid ? static_cast<uint8_t>(__float2int_rz(value)) : uint8_t{0}, .valid_ = valid};
  } else if constexpr (CUDA_DTYPE_OF<Target> == DType::INT32) {
    constexpr auto minimum = -0x1p31F;
    constexpr auto exclusive_maximum = 0x1p31F;
    const auto valid = value >= minimum && value < exclusive_maximum;
    return {.value_ = valid ? __float2int_rz(value) : int32_t{0}, .valid_ = valid};
  } else {
    static_assert(CUDA_DTYPE_OF<Target> == DType::INT64);
    constexpr auto minimum = -0x1p63F;
    constexpr auto exclusive_maximum = 0x1p63F;
    const auto valid = value >= minimum && value < exclusive_maximum;
    return {.value_ = valid ? __float2ll_rz(value) : int64_t{0}, .valid_ = valid};
  }
}

template <CudaStorageType Target>
__device__ auto ConvertIntegerToFloating(int64_t value) -> Target {
  if constexpr (CUDA_DTYPE_OF<Target> == DType::FLOAT16) {
    return __ll2half_rn(value);
  } else if constexpr (CUDA_DTYPE_OF<Target> == DType::BFLOAT16) {
    return __ll2bfloat16_rn(value);
  } else {
    static_assert(CUDA_DTYPE_OF<Target> == DType::FLOAT32);
    return __ll2float_rn(value);
  }
}

template <CudaStorageType Target>
__device__ auto ConvertFloating(float value) -> Target {
  if constexpr (CUDA_DTYPE_OF<Target> == DType::FLOAT16) {
    return __float2half_rn(value);
  } else if constexpr (CUDA_DTYPE_OF<Target> == DType::BFLOAT16) {
    return __float2bfloat16_rn(value);
  } else {
    static_assert(CUDA_DTYPE_OF<Target> == DType::FLOAT32);
    return value;
  }
}

template <CudaStorageType Target, CudaStorageType Source>
__device__ auto ConvertValue(Source value) -> CastResult<Target> {
  if constexpr (CUDA_DTYPE_OF<Target> == DType::BOOL) {
    if constexpr (CUDA_DTYPE_OF<Source> == DType::FLOAT16 || CUDA_DTYPE_OF<Source> == DType::BFLOAT16 ||
                  CUDA_DTYPE_OF<Source> == DType::FLOAT32) {
      return {.value_ = ToFloat(value) != 0.0F, .valid_ = true};
    } else {
      return {.value_ = ToInt64(value) != 0, .valid_ = true};
    }
  } else if constexpr (CUDA_DTYPE_OF<Target> == DType::UINT8 || CUDA_DTYPE_OF<Target> == DType::INT32 ||
                       CUDA_DTYPE_OF<Target> == DType::INT64) {
    if constexpr (CUDA_DTYPE_OF<Source> == DType::FLOAT16 || CUDA_DTYPE_OF<Source> == DType::BFLOAT16 ||
                  CUDA_DTYPE_OF<Source> == DType::FLOAT32) {
      return ConvertFloatingToInteger<Target>(ToFloat(value));
    } else {
      return ConvertInteger<Target>(ToInt64(value));
    }
  } else {
    static_assert(CUDA_DTYPE_OF<Target> == DType::FLOAT16 || CUDA_DTYPE_OF<Target> == DType::BFLOAT16 ||
                  CUDA_DTYPE_OF<Target> == DType::FLOAT32);
    if constexpr (CUDA_DTYPE_OF<Source> == DType::FLOAT16 || CUDA_DTYPE_OF<Source> == DType::BFLOAT16 ||
                  CUDA_DTYPE_OF<Source> == DType::FLOAT32) {
      return {.value_ = ConvertFloating<Target>(ToFloat(value)), .valid_ = true};
    } else {
      return {.value_ = ConvertIntegerToFloating<Target>(ToInt64(value)), .valid_ = true};
    }
  }
}

template <CudaStorageType Target, CudaStorageType Source>
__device__ void CastOne(Target *output, const Source *input, int64_t index,
                        const DeviceErrorLaunchContext &error_context) {
  const auto source = *input;
  const auto converted = ConvertValue<Target>(source);
  if (!converted.valid_) {
    ReportDeviceError(error_context, DeviceErrorCode::CAST_OUT_OF_RANGE, index, GetValueBits(source));
    return;
  }
  *output = converted.value_;
}

template <CudaStorageType Target, CudaStorageType Source>
struct CastOperation final {
  DeviceErrorLaunchContext error_context_;

  template <typename Parameters>
  __device__ void operator()(const Parameters &parameters, ElementwiseIndexType<Parameters> index) const {
    CastOne(reinterpret_cast<Target *>(GetElementwisePointer(parameters, 0, index)),
            reinterpret_cast<const Source *>(GetElementwisePointer(parameters, 1, index)), static_cast<int64_t>(index),
            error_context_);
  }
};

template <CudaStorageType Target, CudaStorageType Source, typename Index>
__global__ void ContiguousCastKernel(Target *output, const Source *input, Index num_elements,
                                     DeviceErrorLaunchContext error_context) {
  auto index = (static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x)) + static_cast<Index>(threadIdx.x);
  const auto grid_stride = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  while (index < num_elements) {
    CastOne(output + index, input + index, static_cast<int64_t>(index), error_context);
    index = static_cast<Index>(index + grid_stride);
  }
}

template <CudaStorageType Target, CudaStorageType Source, typename Parameters>
void LaunchCastWithParameters(cudaStream_t stream, const Parameters &parameters, IteratorPath path,
                              const DeviceErrorLaunchContext &error_context) {
  const auto block_count = GetElementwiseBlockCount(parameters.num_elements_);
  if (path == IteratorPath::CONTIGUOUS) {
    ContiguousCastKernel<<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<Target *>(parameters.pointers_[0]), reinterpret_cast<const Source *>(parameters.pointers_[1]),
        parameters.num_elements_, error_context);
    return;
  }
  ElementwiseKernel<<<block_count, ELEMENTWISE_THREADS_PER_BLOCK, 0, stream>>>(
      parameters, CastOperation<Target, Source>{error_context});
}

template <CudaStorageType Target, CudaStorageType Source>
void LaunchTypedCast(cudaStream_t stream, const ElementwiseIterator &iterator,
                     const DeviceErrorLaunchContext &error_context, std::source_location location) {
  if (iterator.GetIndexWidth() == IndexWidth::UINT32) {
    LaunchCastWithParameters<Target, Source>(stream, iterator.MakeParameters32(location), iterator.GetPath(),
                                             error_context);
    return;
  }
  LaunchCastWithParameters<Target, Source>(stream, iterator.MakeParameters64(), iterator.GetPath(), error_context);
}

}  // namespace

void LaunchCast(cudaStream_t stream, DType source_dtype, DType target_dtype, const ElementwiseIterator &iterator,
                const DeviceErrorLaunchContext &error_context, std::source_location location) {
  if (stream == nullptr || iterator.GetOperandCount() != 2 || iterator.GetNumElements() <= 0 ||
      error_context.record_ == nullptr) {
    throw InternalError("invalid elementwise cast launch plan", location);
  }

  DispatchCudaDType(source_dtype, "CastOut", [&]<CudaStorageType Source>(std::type_identity<Source>) {
    DispatchCudaDType(target_dtype, "CastOut", [&]<CudaStorageType Target>(std::type_identity<Target>) {
      LaunchTypedCast<Target, Source>(stream, iterator, error_context, location);
    });
  });
}

}  // namespace ttl::internal
