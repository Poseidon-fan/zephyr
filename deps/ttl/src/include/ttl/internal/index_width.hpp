#pragma once

#include <cstdint>

namespace ttl::internal {

/** Width of the arithmetic used to lower logical tensor indices to byte offsets. */
enum class IndexWidth : uint8_t {
  UINT32,
  UINT64,
};

}  // namespace ttl::internal
