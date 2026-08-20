#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <ttl/tensor/dtype.hpp>

namespace zephyr {

/** A named shape dimension whose concrete extent is supplied for each model invocation. */
struct DynamicDimension final {
  /** Symbol name shared by all occurrences of this dimension. */
  std::string name_;

  [[nodiscard]] auto operator==(const DynamicDimension &) const -> bool = default;
};

/** One static extent or one invocation-bound dynamic extent. */
using Dimension = std::variant<int64_t, DynamicDimension>;

/** Logical tensor dimensions in row-major order; an empty shape represents a scalar. */
using Shape = std::vector<Dimension>;

/** The element type and logical shape of an IR value. */
struct TensorType final {
  /** Tensor element type supplied by TTL. */
  ttl::DType dtype_;

  /** Logical tensor shape. */
  Shape shape_;

  [[nodiscard]] auto operator==(const TensorType &) const -> bool = default;
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

}  // namespace zephyr
