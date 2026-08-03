#pragma once

#include <optional>
#include <source_location>

#include "ttl/tensor/shape.hpp"

namespace ttl::internal {

/**
 * Compute strides for a metadata-only reshape.
 *
 * A missing result means the target shape crosses a physically discontinuous chunk and therefore requires a copy.
 */
[[nodiscard]] auto ComputeViewStrides(const Shape &old_shape, const Strides &old_strides, const Shape &new_shape,
                                      std::source_location location = std::source_location::current())
    -> std::optional<Strides>;

}  // namespace ttl::internal
