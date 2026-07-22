#pragma once

#include <vector>

namespace zephyr {

class VectorAdder final {
 public:
  [[nodiscard]] auto Add(const std::vector<float> &left, const std::vector<float> &right) const -> std::vector<float>;
};

}  // namespace zephyr
