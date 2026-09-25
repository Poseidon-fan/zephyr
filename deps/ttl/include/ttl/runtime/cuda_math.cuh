#pragma once

#include <bit>
#include <source_location>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <ttl/runtime/cuda_dtype.hpp>
#include <ttl/tensor/scalar.hpp>

namespace ttl {

template <CudaStorageType T>
[[nodiscard]] __host__ __device__ constexpr auto IsCudaFloatingType() noexcept -> bool {
  return CUDA_DTYPE_OF<T> == DType::FLOAT16 || CUDA_DTYPE_OF<T> == DType::BFLOAT16 ||
         CUDA_DTYPE_OF<T> == DType::FLOAT32;
}

template <CudaStorageType T>
[[nodiscard]] __host__ __device__ constexpr auto IsCudaSignedIntegerType() noexcept -> bool {
  return CUDA_DTYPE_OF<T> == DType::INT32 || CUDA_DTYPE_OF<T> == DType::INT64;
}

template <CudaStorageType T>
__host__ __device__ auto ToElementwiseFloat(T value) -> float {
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
__host__ __device__ auto FromElementwiseFloat(float value) -> T {
  if constexpr (CUDA_DTYPE_OF<T> == DType::FLOAT16) {
    return __float2half_rn(value);
  } else if constexpr (CUDA_DTYPE_OF<T> == DType::BFLOAT16) {
    return __float2bfloat16_rn(value);
  } else {
    static_assert(CUDA_DTYPE_OF<T> == DType::FLOAT32);
    return value;
  }
}

template <CudaStorageType T>
[[nodiscard]] auto ConvertElementwiseScalar(const Scalar &value, std::source_location location) -> T {
  using HostType = StorageTypeForT<CUDA_DTYPE_OF<T>>;
  static_assert(sizeof(HostType) == sizeof(T));
  return std::bit_cast<T>(value.Cast<HostType>(location));
}

}  // namespace ttl
