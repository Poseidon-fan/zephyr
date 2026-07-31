#pragma once

#include <concepts>
#include <cstdint>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"

namespace ttl::internal {

template <DType dtype>
struct CudaTypeFor;

template <>
struct CudaTypeFor<DType::BOOL> {
  using Type = bool;
};

template <>
struct CudaTypeFor<DType::UINT8> {
  using Type = uint8_t;
};

template <>
struct CudaTypeFor<DType::INT32> {
  using Type = int32_t;
};

template <>
struct CudaTypeFor<DType::INT64> {
  using Type = int64_t;
};

template <>
struct CudaTypeFor<DType::FLOAT16> {
  using Type = __half;
};

template <>
struct CudaTypeFor<DType::BFLOAT16> {
  using Type = __nv_bfloat16;
};

template <>
struct CudaTypeFor<DType::FLOAT32> {
  using Type = float;
};

template <DType dtype>
using CudaTypeForT = typename CudaTypeFor<dtype>::Type;

template <typename T>
struct CudaDTypeOf;

template <>
struct CudaDTypeOf<bool> {
  static constexpr DType VALUE = DType::BOOL;
};

template <>
struct CudaDTypeOf<uint8_t> {
  static constexpr DType VALUE = DType::UINT8;
};

template <>
struct CudaDTypeOf<int32_t> {
  static constexpr DType VALUE = DType::INT32;
};

template <>
struct CudaDTypeOf<int64_t> {
  static constexpr DType VALUE = DType::INT64;
};

template <>
struct CudaDTypeOf<__half> {
  static constexpr DType VALUE = DType::FLOAT16;
};

template <>
struct CudaDTypeOf<__nv_bfloat16> {
  static constexpr DType VALUE = DType::BFLOAT16;
};

template <>
struct CudaDTypeOf<float> {
  static constexpr DType VALUE = DType::FLOAT32;
};

template <typename T>
concept CudaStorageType = !std::is_volatile_v<T> && requires { CudaDTypeOf<std::remove_cv_t<T>>::VALUE; };

template <CudaStorageType T>
// CUDA-mode clang-tidy misclassifies this dependent constexpr variable template as dynamically initialized.
// NOLINTNEXTLINE(bugprone-dynamic-static-initializers)
inline constexpr DType CUDA_DTYPE_OF = CudaDTypeOf<std::remove_cv_t<T>>::VALUE;

static_assert(sizeof(CudaTypeForT<DType::BOOL>) == sizeof(StorageTypeForT<DType::BOOL>));
static_assert(alignof(CudaTypeForT<DType::BOOL>) == alignof(StorageTypeForT<DType::BOOL>));
static_assert(std::is_trivially_copyable_v<CudaTypeForT<DType::BOOL>>);
static_assert(std::is_standard_layout_v<CudaTypeForT<DType::BOOL>>);
static_assert(sizeof(CudaTypeForT<DType::FLOAT16>) == sizeof(StorageTypeForT<DType::FLOAT16>));
static_assert(alignof(CudaTypeForT<DType::FLOAT16>) == alignof(StorageTypeForT<DType::FLOAT16>));
static_assert(sizeof(CudaTypeForT<DType::BFLOAT16>) == sizeof(StorageTypeForT<DType::BFLOAT16>));
static_assert(alignof(CudaTypeForT<DType::BFLOAT16>) == alignof(StorageTypeForT<DType::BFLOAT16>));

template <typename Function, typename Type>
concept CudaDTypeVisitorForOne =
    requires(Function &&function) { std::forward<Function>(function)(std::type_identity<Type>{}); };

template <typename Function, typename... Types>
concept CudaDTypeVisitorFor = (CudaDTypeVisitorForOne<Function, Types> && ...);

[[noreturn]] inline void ThrowInvalidCudaDType(std::string_view operation, DType dtype, std::source_location location) {
  std::string message;
  message.reserve(operation.size() + 32);
  message.append(operation);
  message.append(": invalid dtype value ");
  message.append(std::to_string(static_cast<uint8_t>(dtype)));
  throw InvalidArgumentError(std::move(message), location);
}

[[noreturn]] inline void ThrowUnsupportedCudaDType(std::string_view operation, DType dtype,
                                                   std::string_view expected_category, std::source_location location) {
  if (!IsValidDType(dtype)) {
    ThrowInvalidCudaDType(operation, dtype, location);
  }

  const auto dtype_name = GetDTypeName(dtype);
  std::string message;
  message.reserve(operation.size() + dtype_name.size() + expected_category.size() + 40);
  message.append(operation);
  message.append(" does not support dtype ");
  message.append(dtype_name);
  message.append("; expected ");
  message.append(expected_category);
  throw NotSupportedError(std::move(message), location);
}

/**
 * Invoke a templated callable with the CUDA computation type for any supported dtype.
 *
 * The callable must accept `std::type_identity<T>` for every CUDA storage type T.
 * Every specialization must return one common type.
 */
template <typename Function>
  requires CudaDTypeVisitorFor<Function &&, CudaTypeForT<DType::BOOL>, CudaTypeForT<DType::UINT8>,
                               CudaTypeForT<DType::INT32>, CudaTypeForT<DType::INT64>, CudaTypeForT<DType::FLOAT16>,
                               CudaTypeForT<DType::BFLOAT16>, CudaTypeForT<DType::FLOAT32>>
auto DispatchCudaDType(DType dtype, std::string_view operation, Function &&function,
                       std::source_location location = std::source_location::current()) -> decltype(auto) {
  switch (dtype) {
    case DType::BOOL:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::BOOL>>{});
    case DType::UINT8:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::UINT8>>{});
    case DType::INT32:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::INT32>>{});
    case DType::INT64:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::INT64>>{});
    case DType::FLOAT16:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::FLOAT16>>{});
    case DType::BFLOAT16:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::BFLOAT16>>{});
    case DType::FLOAT32:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::FLOAT32>>{});
  }
  ThrowInvalidCudaDType(operation, dtype, location);
}

/** Dispatch UINT8, INT32, INT64, FLOAT16, BFLOAT16, or FLOAT32; BOOL is excluded. */
template <typename Function>
  requires CudaDTypeVisitorFor<Function &&, CudaTypeForT<DType::UINT8>, CudaTypeForT<DType::INT32>,
                               CudaTypeForT<DType::INT64>, CudaTypeForT<DType::FLOAT16>, CudaTypeForT<DType::BFLOAT16>,
                               CudaTypeForT<DType::FLOAT32>>
auto DispatchCudaNumericDType(DType dtype, std::string_view operation, Function &&function,
                              std::source_location location = std::source_location::current()) -> decltype(auto) {
  switch (dtype) {
    case DType::UINT8:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::UINT8>>{});
    case DType::INT32:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::INT32>>{});
    case DType::INT64:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::INT64>>{});
    case DType::FLOAT16:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::FLOAT16>>{});
    case DType::BFLOAT16:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::BFLOAT16>>{});
    case DType::FLOAT32:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::FLOAT32>>{});
    case DType::BOOL:
      break;
  }
  ThrowUnsupportedCudaDType(operation, dtype, "numeric dtype", location);
}

/** Dispatch UINT8, INT32, or INT64; BOOL is excluded. */
template <typename Function>
  requires CudaDTypeVisitorFor<Function &&, CudaTypeForT<DType::UINT8>, CudaTypeForT<DType::INT32>,
                               CudaTypeForT<DType::INT64>>
auto DispatchCudaIntegralDType(DType dtype, std::string_view operation, Function &&function,
                               std::source_location location = std::source_location::current()) -> decltype(auto) {
  switch (dtype) {
    case DType::UINT8:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::UINT8>>{});
    case DType::INT32:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::INT32>>{});
    case DType::INT64:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::INT64>>{});
    case DType::BOOL:
    case DType::FLOAT16:
    case DType::BFLOAT16:
    case DType::FLOAT32:
      break;
  }
  ThrowUnsupportedCudaDType(operation, dtype, "integral dtype", location);
}

/** Dispatch FLOAT16, BFLOAT16, or FLOAT32. */
template <typename Function>
  requires CudaDTypeVisitorFor<Function &&, CudaTypeForT<DType::FLOAT16>, CudaTypeForT<DType::BFLOAT16>,
                               CudaTypeForT<DType::FLOAT32>>
auto DispatchCudaFloatingDType(DType dtype, std::string_view operation, Function &&function,
                               std::source_location location = std::source_location::current()) -> decltype(auto) {
  switch (dtype) {
    case DType::FLOAT16:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::FLOAT16>>{});
    case DType::BFLOAT16:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::BFLOAT16>>{});
    case DType::FLOAT32:
      return std::forward<Function>(function)(std::type_identity<CudaTypeForT<DType::FLOAT32>>{});
    case DType::BOOL:
    case DType::UINT8:
    case DType::INT32:
    case DType::INT64:
      break;
  }
  ThrowUnsupportedCudaDType(operation, dtype, "floating dtype", location);
}

}  // namespace ttl::internal
