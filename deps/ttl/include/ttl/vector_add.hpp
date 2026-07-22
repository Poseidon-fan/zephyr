#pragma once

#include <span>
#include <vector>

namespace ttl {

[[nodiscard]] auto VectorAdd(std::span<const float> left, std::span<const float> right) -> std::vector<float>;

}  // namespace ttl
