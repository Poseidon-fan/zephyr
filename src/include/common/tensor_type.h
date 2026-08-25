#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <ttl/tensor/dtype.hpp>

namespace zephyr {

/** Canonical packed-token dimension used by model runners. */
static constexpr const char *TOKEN_DIMENSION = "tokens";

/** A named shape dimension whose concrete extent is supplied for each model invocation. */
struct DynamicDimension final {
  /** Symbol name shared by all occurrences of this dimension. */
  std::string name_;

  [[nodiscard]] auto operator==(const DynamicDimension &) const -> bool = default;

  /** Returns the canonical symbol spelling. */
  [[nodiscard]] auto ToString() const -> std::string { return name_; }
};

/** One static extent or one invocation-bound dynamic extent. */
using Dimension = std::variant<int64_t, DynamicDimension>;

/** Logical tensor dimensions in row-major order; an empty shape represents a scalar. */
using Shape = std::vector<Dimension>;

/** Binds one dynamic dimension name to a concrete extent. */
struct DynamicDimensionBinding final {
  /** Dynamic dimension name. */
  std::string_view name_;

  /** Concrete extent used for this resolution. */
  int64_t extent_;
};

/** The element type and logical shape of an IR value. */
struct TensorType final {
  /** Tensor element type supplied by TTL. */
  ttl::DType dtype_;

  /** Logical tensor shape. */
  Shape shape_;

  [[nodiscard]] auto operator==(const TensorType &) const -> bool = default;

  /** Returns the canonical textual tensor type. */
  [[nodiscard]] auto ToString() const -> std::string;
};

/** One half-open interval used to select a tensor dimension. */
struct TensorRange final {
  /** Inclusive start offset. */
  int64_t begin_;

  /** Exclusive end offset. */
  int64_t end_;

  [[nodiscard]] auto operator==(const TensorRange &) const -> bool = default;
};

/** One half-open range per selected tensor dimension. */
using TensorSlice = std::vector<TensorRange>;

/** Returns a positive static extent or throws when the dimension is dynamic/invalid. */
[[nodiscard]] auto GetStaticExtent(const Dimension &dimension, std::string_view name) -> int64_t;

/** Resolves all dynamic dimensions using the supplied bindings. */
[[nodiscard]] auto ResolveShape(const Shape &shape, std::span<const DynamicDimensionBinding> bindings)
    -> std::vector<int64_t>;

/** Creates a full row-major slice for a statically shaped tensor. */
[[nodiscard]] auto MakeFullSlice(const Shape &shape, std::string_view name = "tensor dimension") -> TensorSlice;

}  // namespace zephyr
