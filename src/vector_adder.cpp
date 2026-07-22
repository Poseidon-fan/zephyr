#include <zephyr/vector_adder.hpp>

#include <ttl/vector_add.hpp>

#include <span>
#include <vector>

namespace zephyr {

auto VectorAdder::Add(const std::vector<float> &left, const std::vector<float> &right) const -> std::vector<float> {
  return ttl::VectorAdd(std::span<const float>(left), std::span<const float>(right));
}

}  // namespace zephyr
