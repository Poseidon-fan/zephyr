#pragma once

#include <concepts>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda/std/type_traits>

#include "ttl/internal/kernels/elementwise/elementwise_math.cuh"
#include "ttl/internal/ops/reduction.hpp"
#include "ttl/internal/runtime/library/cuda_dtype.hpp"

namespace ttl::internal {

template <CudaStorageType T>
using SummationAccumulator = cuda::std::conditional_t<IsCudaFloatingType<T>(), float, uint64_t>;

template <CudaStorageType T>
struct SumReductionOperation final {
  using Accumulator = SummationAccumulator<T>;

  __device__ auto Identity() const -> Accumulator { return Accumulator{0}; }

  __device__ auto Lift(T value, int64_t /*index*/) const -> Accumulator {
    if constexpr (IsCudaFloatingType<T>()) {
      return ToElementwiseFloat(value);
    } else if constexpr (std::same_as<T, int32_t>) {
      return static_cast<uint32_t>(value);
    } else {
      static_assert(std::same_as<T, int64_t>);
      return __builtin_bit_cast(uint64_t, value);
    }
  }

  __device__ auto Combine(Accumulator lhs, Accumulator rhs) const -> Accumulator {
    return static_cast<Accumulator>(lhs + rhs);
  }

  __device__ auto operator()(Accumulator lhs, Accumulator rhs) const -> Accumulator { return Combine(lhs, rhs); }

  __device__ auto Project(Accumulator accumulator, uint64_t /*reduction_count*/) const -> T {
    if constexpr (IsCudaFloatingType<T>()) {
      return FromElementwiseFloat<T>(accumulator);
    } else if constexpr (std::same_as<T, int32_t>) {
      return __builtin_bit_cast(int32_t, static_cast<uint32_t>(accumulator));
    } else {
      static_assert(std::same_as<T, int64_t>);
      return __builtin_bit_cast(int64_t, accumulator);
    }
  }
};

template <CudaStorageType T>
struct MeanReductionOperation final {
  using Accumulator = float;

  __device__ auto Identity() const -> Accumulator { return 0.0F; }
  __device__ auto Lift(T value, int64_t /*index*/) const -> Accumulator { return ToElementwiseFloat(value); }
  __device__ auto Combine(Accumulator lhs, Accumulator rhs) const -> Accumulator { return lhs + rhs; }
  __device__ auto operator()(Accumulator lhs, Accumulator rhs) const -> Accumulator { return Combine(lhs, rhs); }
  __device__ auto Project(Accumulator accumulator, uint64_t reduction_count) const -> T {
    return FromElementwiseFloat<T>(accumulator / static_cast<float>(reduction_count));
  }
};

template <CudaStorageType T>
struct alignas(8) IndexedReductionValue final {
  T value_;
  int64_t index_;
};

template <CudaStorageType T>
inline constexpr bool VALID_INDEXED_REDUCTION_VALUE = sizeof(IndexedReductionValue<T>) == 16 &&
                                                      alignof(IndexedReductionValue<T>) == 8;

template <CudaStorageType T, ReductionOp operation>
struct ExtremaReductionOperation final {
  static_assert(operation == ReductionOp::MINIMUM || operation == ReductionOp::MAXIMUM ||
                operation == ReductionOp::ARG_MIN || operation == ReductionOp::ARG_MAX);
  static_assert(VALID_INDEXED_REDUCTION_VALUE<T>);

  using Accumulator = IndexedReductionValue<T>;

  __device__ auto Identity() const -> Accumulator { return {.value_ = T{}, .index_ = -1}; }
  __device__ auto Lift(T value, int64_t index) const -> Accumulator { return {.value_ = value, .index_ = index}; }

  __device__ auto Combine(Accumulator lhs, Accumulator rhs) const -> Accumulator {
    if (lhs.index_ < 0) {
      return rhs;
    }
    if (rhs.index_ < 0) {
      return lhs;
    }

    if constexpr (IsCudaFloatingType<T>()) {
      const auto lhs_value = ToElementwiseFloat(lhs.value_);
      const auto rhs_value = ToElementwiseFloat(rhs.value_);
      const auto lhs_nan = isnan(lhs_value);
      const auto rhs_nan = isnan(rhs_value);
      if (lhs_nan || rhs_nan) {
        if (lhs_nan != rhs_nan) {
          return lhs_nan ? lhs : rhs;
        }
        return lhs.index_ <= rhs.index_ ? lhs : rhs;
      }
      if constexpr (operation == ReductionOp::MINIMUM || operation == ReductionOp::ARG_MIN) {
        if (lhs_value != rhs_value) {
          return lhs_value < rhs_value ? lhs : rhs;
        }
      } else {
        if (lhs_value != rhs_value) {
          return lhs_value > rhs_value ? lhs : rhs;
        }
      }
    } else {
      if constexpr (operation == ReductionOp::MINIMUM || operation == ReductionOp::ARG_MIN) {
        if (lhs.value_ != rhs.value_) {
          return lhs.value_ < rhs.value_ ? lhs : rhs;
        }
      } else {
        if (lhs.value_ != rhs.value_) {
          return lhs.value_ > rhs.value_ ? lhs : rhs;
        }
      }
    }
    return lhs.index_ <= rhs.index_ ? lhs : rhs;
  }

  __device__ auto operator()(Accumulator lhs, Accumulator rhs) const -> Accumulator { return Combine(lhs, rhs); }

  __device__ auto Project(Accumulator accumulator, uint64_t /*reduction_count*/) const
      -> cuda::std::conditional_t<operation == ReductionOp::ARG_MIN || operation == ReductionOp::ARG_MAX, int64_t, T> {
    if constexpr (operation == ReductionOp::ARG_MIN || operation == ReductionOp::ARG_MAX) {
      return accumulator.index_;
    } else {
      return accumulator.value_;
    }
  }
};

template <ReductionOp operation>
struct LogicalReductionOperation final {
  static_assert(operation == ReductionOp::ANY || operation == ReductionOp::ALL);

  using Accumulator = uint32_t;

  __device__ auto Identity() const -> Accumulator { return operation == ReductionOp::ALL ? 1U : 0U; }
  __device__ auto Lift(bool value, int64_t /*index*/) const -> Accumulator { return value ? 1U : 0U; }
  __device__ auto Combine(Accumulator lhs, Accumulator rhs) const -> Accumulator {
    if constexpr (operation == ReductionOp::ANY) {
      return lhs | rhs;
    } else {
      return lhs & rhs;
    }
  }
  __device__ auto operator()(Accumulator lhs, Accumulator rhs) const -> Accumulator { return Combine(lhs, rhs); }
  __device__ auto Project(Accumulator accumulator, uint64_t /*reduction_count*/) const -> bool {
    return accumulator != 0;
  }
};

}  // namespace ttl::internal
