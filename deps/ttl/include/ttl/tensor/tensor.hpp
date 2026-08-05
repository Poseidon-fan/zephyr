#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <type_traits>

#include "ttl/common/device.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"

namespace ttl::internal {

class TensorAccess;
class TensorFactory;
class TensorImpl;

}  // namespace ttl::internal

namespace ttl {

class Stream;

class ExecutionContext;

/** Conservative relationship between the storage regions reachable by two tensors. */
enum class AliasKind : uint8_t {
  DISJOINT,
  EXACT,
  MAY_OVERLAP,
};

/**
 * @brief Immutable tensor metadata and shared device-storage handle.
 *
 * Copying a Tensor shares both its immutable metadata and its Storage. Creating a view produces a new metadata object
 * that shares the same Storage. Data pointers returned by Tensor are CUDA device pointers and must not be dereferenced
 * by host code.
 */
class Tensor final {
 public:
  Tensor() = delete;
  Tensor(const Tensor &) noexcept = default;
  auto operator=(const Tensor &) noexcept -> Tensor & = default;
  Tensor(Tensor &&) noexcept = default;
  auto operator=(Tensor &&) noexcept -> Tensor & = default;
  ~Tensor() = default;

  [[nodiscard]] auto GetDevice(std::source_location location = std::source_location::current()) const -> Device;
  [[nodiscard]] auto GetDType(std::source_location location = std::source_location::current()) const -> DType;
  [[nodiscard]] auto GetShape(std::source_location location = std::source_location::current()) const -> const Shape &;
  [[nodiscard]] auto GetStrides(std::source_location location = std::source_location::current()) const
      -> const Strides &;
  [[nodiscard]] auto GetRank(std::source_location location = std::source_location::current()) const -> size_t;
  [[nodiscard]] auto GetNumElements(std::source_location location = std::source_location::current()) const -> int64_t;
  [[nodiscard]] auto GetStorageOffset(std::source_location location = std::source_location::current()) const -> int64_t;
  [[nodiscard]] auto IsContiguous(std::source_location location = std::source_location::current()) const -> bool;
  [[nodiscard]] auto IsNonOverlappingDense(std::source_location location = std::source_location::current()) const
      -> bool;
  [[nodiscard]] auto HasZeroStride(std::source_location location = std::source_location::current()) const -> bool;

  template <TensorStorageType T>
  [[nodiscard]] auto GetData(std::source_location location = std::source_location::current()) const
      -> const std::remove_cv_t<T> * {
    return static_cast<const std::remove_cv_t<T> *>(GetDataPointer(DTYPE_OF<T>, location));
  }

  /**
   * @brief Record a direct asynchronous read on a stream before releasing the final Tensor owner.
   *
   * This protects allocation lifetime only; it does not establish an execution dependency or retain storage for CUDA
   * Graph capture. Capture and writable access must use SubmitCudaKernel.
   */
  void RecordUsage(const Stream &stream, std::source_location location = std::source_location::current()) const;

 private:
  friend class internal::TensorAccess;
  friend class internal::TensorFactory;

  explicit Tensor(std::shared_ptr<const internal::TensorImpl> impl) noexcept;

  [[nodiscard]] auto GetImpl(std::source_location location) const -> const internal::TensorImpl &;
  [[nodiscard]] auto GetDataPointer(DType expected_dtype, std::source_location location) const -> const void *;

  std::shared_ptr<const internal::TensorImpl> impl_;
};

/**
 * @brief Classify aliasing without attempting an expensive exact analysis of arbitrary strided holes.
 *
 * Different Storage owners are disjoint. Identical metadata over the same Storage is exact. Other non-empty tensors
 * whose conservative reachable byte intervals intersect may overlap.
 */
[[nodiscard]] auto ClassifyAlias(const Tensor &lhs, const Tensor &rhs,
                                 std::source_location location = std::source_location::current()) -> AliasKind;

/** Allocate an uninitialized contiguous tensor on the execution context's device and stream. */
[[nodiscard]] auto Empty(ExecutionContext &context, const Shape &shape, DType dtype,
                         std::source_location location = std::source_location::current()) -> Tensor;

/** Allocate an uninitialized non-overlapping dense tensor with explicit element strides. */
[[nodiscard]] auto EmptyStrided(ExecutionContext &context, const Shape &shape, const Strides &strides, DType dtype,
                                std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
