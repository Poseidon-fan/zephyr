#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <source_location>
#include <type_traits>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/shape.hpp"

namespace ttl::internal {

class TensorAccess;
class TensorFactory;
class TensorImpl;

}  // namespace ttl::internal

namespace ttl {

class ExecutionContext;

/** Conservative relationship between the storage regions reachable by two tensors. */
enum class AliasKind : uint8_t {
  DISJOINT,
  EXACT,
  MAY_OVERLAP,
};

/**
 * Immutable tensor metadata and shared device-storage handle.
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

  [[nodiscard]] auto GetDevice() const noexcept -> Device;
  [[nodiscard]] auto GetDType() const noexcept -> DType;
  [[nodiscard]] auto GetShape() const noexcept -> const Shape &;
  [[nodiscard]] auto GetStrides() const noexcept -> const Strides &;
  [[nodiscard]] auto GetRank() const noexcept -> size_t;
  [[nodiscard]] auto GetNumElements() const noexcept -> int64_t;
  [[nodiscard]] auto GetStorageOffset() const noexcept -> int64_t;
  [[nodiscard]] auto IsContiguous() const noexcept -> bool;
  [[nodiscard]] auto IsNonOverlappingDense() const noexcept -> bool;
  [[nodiscard]] auto HasZeroStride() const noexcept -> bool;

  template <TensorStorageType T>
  [[nodiscard]] auto GetData(std::source_location location = std::source_location::current()) const
      -> const std::remove_cv_t<T> * {
    return static_cast<const std::remove_cv_t<T> *>(GetDataPointer(DTYPE_OF<T>, location));
  }

 private:
  friend class internal::TensorAccess;
  friend class internal::TensorFactory;

  explicit Tensor(std::shared_ptr<const internal::TensorImpl> impl) noexcept;

  [[nodiscard]] auto GetImpl() const noexcept -> const internal::TensorImpl &;
  [[nodiscard]] auto GetDataPointer(DType expected_dtype, std::source_location location) const -> const void *;

  std::shared_ptr<const internal::TensorImpl> impl_;
};

/**
 * Classify aliasing without attempting an expensive exact analysis of arbitrary strided holes.
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
