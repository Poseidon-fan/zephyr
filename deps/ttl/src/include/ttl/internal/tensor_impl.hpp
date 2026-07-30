#pragma once

#include <cstdint>
#include <memory>
#include <source_location>
#include <type_traits>

#include "ttl/dtype.hpp"
#include "ttl/shape.hpp"
#include "ttl/stream.hpp"
#include "ttl/tensor.hpp"

namespace ttl::internal {

class Storage;

enum class TensorFlag : uint8_t {
  NONE = 0,
  CONTIGUOUS = 1U << 0U,
  HAS_ZERO_STRIDE = 1U << 1U,
  NON_OVERLAPPING_DENSE = 1U << 2U,
};

/** Type-safe, one-byte set of TensorFlag values. */
class TensorFlags final {
 public:
  constexpr TensorFlags() noexcept = default;

  constexpr void Set(TensorFlag flag) noexcept { bits_ = static_cast<uint8_t>(bits_ | GetBits(flag)); }

  [[nodiscard]] constexpr auto Has(TensorFlag flag) const noexcept -> bool { return (bits_ & GetBits(flag)) != 0; }

 private:
  [[nodiscard]] static constexpr auto GetBits(TensorFlag flag) noexcept -> uint8_t {
    return static_cast<uint8_t>(flag);
  }

  uint8_t bits_{0};
};

static_assert(sizeof(TensorFlags) == sizeof(uint8_t));
static_assert(std::is_trivially_copyable_v<TensorFlags>);
static_assert(std::is_standard_layout_v<TensorFlags>);

/** Immutable metadata and Storage owner shared by Tensor value handles. */
class TensorImpl final {
 public:
  TensorImpl(const TensorImpl &) = delete;
  auto operator=(const TensorImpl &) -> TensorImpl & = delete;
  TensorImpl(TensorImpl &&) = delete;
  auto operator=(TensorImpl &&) -> TensorImpl & = delete;

  [[nodiscard]] auto GetStorage() const noexcept -> const std::shared_ptr<Storage> &;
  [[nodiscard]] auto GetDType() const noexcept -> DType;
  [[nodiscard]] auto GetShape() const noexcept -> const Shape &;
  [[nodiscard]] auto GetStrides() const noexcept -> const Strides &;
  [[nodiscard]] auto GetStorageOffset() const noexcept -> int64_t;
  [[nodiscard]] auto GetNumElements() const noexcept -> int64_t;
  [[nodiscard]] auto HasFlag(TensorFlag flag) const noexcept -> bool;

 private:
  friend class TensorFactory;

  TensorImpl(std::shared_ptr<Storage> storage, DType dtype, Shape shape, Strides strides, int64_t storage_offset,
             TensorFlags flags) noexcept;

  std::shared_ptr<Storage> storage_;
  Shape shape_;
  Strides strides_;
  int64_t storage_offset_;
  DType dtype_;
  TensorFlags flags_;
};

/** The only construction path for TensorImpl; validates every metadata and storage invariant. */
class TensorFactory final {
 public:
  [[nodiscard]] static auto Create(std::shared_ptr<Storage> storage, DType dtype, Shape shape, Strides strides,
                                   int64_t storage_offset,
                                   std::source_location location = std::source_location::current()) -> Tensor;
};

/** Private gateway used by operators and view factories without exposing mutable pointers publicly. */
class TensorAccess final {
 public:
  [[nodiscard]] static auto GetImpl(const Tensor &tensor,
                                    std::source_location location = std::source_location::current())
      -> const TensorImpl &;
  [[nodiscard]] static auto GetStorage(const Tensor &tensor,
                                       std::source_location location = std::source_location::current())
      -> const std::shared_ptr<Storage> &;

  template <TensorStorageType T>
  [[nodiscard]] static auto GetMutableData(Tensor &tensor,
                                           std::source_location location = std::source_location::current())
      -> std::remove_cv_t<T> * {
    return const_cast<std::remove_cv_t<T> *>(tensor.GetData<T>(location));
  }

  static void RecordUsage(const Tensor &tensor, const Stream &stream,
                          std::source_location location = std::source_location::current());
};

/** Return whether a layout maps every logical element to one unique element in one dense storage interval. */
[[nodiscard]] auto IsNonOverlappingDenseLayout(const Shape &shape, const Strides &strides) noexcept -> bool;

}  // namespace ttl::internal
