#include "ttl/internal/device_properties.hpp"

#include <algorithm>
#include <source_location>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>

#include "ttl/error.hpp"
#include "ttl/internal/cuda_check.hpp"

namespace ttl::internal {
namespace {

constexpr ComputeCapability MINIMUM_COMPUTE_CAPABILITY{
    .major_ = 8,
    .minor_ = 0,
};

[[nodiscard]] auto FormatDeviceCountError(Device device, int device_count) -> std::string {
  std::string message{"device ordinal "};
  message.append(std::to_string(device.GetOrdinal()));
  message.append(" is out of range for ");
  message.append(std::to_string(device_count));
  message.append(" visible CUDA device");
  if (device_count != 1) {
    message.push_back('s');
  }
  return message;
}

[[nodiscard]] auto FormatUnsupportedDevice(Device device, std::string_view reason) -> std::string {
  std::string message{device.ToString()};
  message.append(" is not supported: ");
  message.append(reason);
  return message;
}

void ValidateNativeProperties(Device device, const cudaDeviceProp &properties, std::source_location location) {
  if (properties.major < 0 || properties.minor < 0 || properties.minor > 9) {
    throw InternalError(FormatUnsupportedDevice(device, "CUDA returned an invalid compute capability"), location);
  }

  const ComputeCapability compute_capability{
      .major_ = properties.major,
      .minor_ = properties.minor,
  };
  if (compute_capability < MINIMUM_COMPUTE_CAPABILITY) {
    std::string reason{"compute capability "};
    reason.append(std::to_string(properties.major));
    reason.push_back('.');
    reason.append(std::to_string(properties.minor));
    reason.append(" is below the required 8.0");
    throw NotSupportedError(FormatUnsupportedDevice(device, reason), location);
  }

  if (properties.warpSize != 32) {
    throw NotSupportedError(FormatUnsupportedDevice(device, "warp size must be 32"), location);
  }
  if (properties.memoryPoolsSupported == 0) {
    throw NotSupportedError(FormatUnsupportedDevice(device, "CUDA stream-ordered memory pools are required"), location);
  }
  if (properties.computeMode == cudaComputeModeProhibited) {
    throw NotSupportedError(FormatUnsupportedDevice(device, "CUDA compute mode prohibits execution"), location);
  }

  if (properties.multiProcessorCount <= 0) {
    throw InternalError(FormatUnsupportedDevice(device, "CUDA returned an invalid multiprocessor count"), location);
  }
  if (properties.sharedMemPerBlockOptin == 0) {
    throw InternalError(FormatUnsupportedDevice(device, "CUDA returned no dynamic shared-memory capacity"), location);
  }
}

[[nodiscard]] auto GetDeviceName(const cudaDeviceProp &properties) -> std::string {
  const auto name_end = std::ranges::find(properties.name, '\0');
  return {properties.name, name_end};
}

}  // namespace

auto QueryDeviceProperties(Device device, std::source_location location) -> DeviceProperties {
  return QueryDeviceProperties(device, GetCudaApi(), location);
}

auto QueryDeviceProperties(Device device, const CudaApi &cuda_api, std::source_location location) -> DeviceProperties {
  int device_count = 0;
  CheckCuda(cuda_api.get_device_count_(&device_count), "cudaGetDeviceCount", location);
  if (device_count < 0) {
    throw InternalError("cudaGetDeviceCount returned a negative device count", location);
  }
  if (device.GetOrdinal() >= device_count) {
    throw InvalidArgumentError(FormatDeviceCountError(device, device_count), location);
  }

  cudaDeviceProp native_properties{};
  CheckCuda(cuda_api.get_device_properties_(&native_properties, device.GetOrdinal()), "cudaGetDeviceProperties",
            location);
  ValidateNativeProperties(device, native_properties, location);

  return DeviceProperties{
      .device_ = device,
      .name_ = GetDeviceName(native_properties),
      .compute_capability_ =
          {
              .major_ = native_properties.major,
              .minor_ = native_properties.minor,
          },
      .multiprocessor_count_ = native_properties.multiProcessorCount,
      .max_dynamic_shared_memory_per_block_bytes_ = native_properties.sharedMemPerBlockOptin,
      .supports_cluster_launch_ = native_properties.clusterLaunch != 0,
  };
}

}  // namespace ttl::internal
