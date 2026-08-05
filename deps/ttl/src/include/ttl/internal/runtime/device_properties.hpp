#pragma once

#include <source_location>

#include "ttl/common/device.hpp"
#include "ttl/runtime/device_properties.hpp"

namespace ttl::internal {

/**
 * @brief Query and validate one CUDA device without changing the calling thread's current device.
 *
 * The returned device satisfies TTL's SM80, warp-size, memory-pool, and launch-resource requirements.
 */
[[nodiscard]] auto QueryDeviceProperties(Device device, std::source_location location = std::source_location::current())
    -> DeviceProperties;

}  // namespace ttl::internal
