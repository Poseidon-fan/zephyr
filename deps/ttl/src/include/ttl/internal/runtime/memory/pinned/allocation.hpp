#pragma once

#include <cstddef>

namespace ttl::internal {

struct PinnedAllocation final {
  void *pointer_;
  size_t capacity_bytes_;
};

}  // namespace ttl::internal
