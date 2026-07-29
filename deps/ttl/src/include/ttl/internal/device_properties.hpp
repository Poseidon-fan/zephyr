#pragma once

#include <source_location>

#include "ttl/device.hpp"
#include "ttl/device_properties.hpp"

namespace ttl::internal {

/**
 * Query and validate one CUDA device without changing the calling thread's current device.
 *
 * The returned device satisfies TTL's SM80, warp-size, memory-pool, and launch-resource requirements.
 */
[[nodiscard]] auto QueryDeviceProperties(Device device, std::source_location location = std::source_location::current())
    -> DeviceProperties;

}  // namespace ttl::internal
