#include <ttl/vector_add.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ttl {
namespace {

constexpr std::size_t THREADS_PER_BLOCK = 256;

auto CheckCuda(cudaError_t status, std::string_view operation) -> void {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t element_count) {
    void *allocation = nullptr;
    CheckCuda(cudaMalloc(&allocation, element_count * sizeof(float)), "cudaMalloc");
    data_ = static_cast<float *>(allocation);
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  auto operator=(const DeviceBuffer &) -> DeviceBuffer & = delete;
  DeviceBuffer(DeviceBuffer &&) = delete;
  auto operator=(DeviceBuffer &&) -> DeviceBuffer & = delete;

  [[nodiscard]] auto Data() noexcept -> float * { return data_; }

 private:
  float *data_ = nullptr;
};

__global__ void VectorAddKernel(const float *left, const float *right, float *result, std::size_t element_count) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < element_count) {
    result[index] = left[index] + right[index];
  }
}

}  // namespace

auto VectorAdd(std::span<const float> left, std::span<const float> right) -> std::vector<float> {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector sizes must match");
  }

  std::vector<float> result(left.size());
  if (left.empty()) {
    return result;
  }

  const auto byte_count = left.size_bytes();
  DeviceBuffer device_left(left.size());
  DeviceBuffer device_right(right.size());
  DeviceBuffer device_result(result.size());

  CheckCuda(cudaMemcpy(device_left.Data(), left.data(), byte_count, cudaMemcpyHostToDevice), "copy left to device");
  CheckCuda(cudaMemcpy(device_right.Data(), right.data(), byte_count, cudaMemcpyHostToDevice), "copy right to device");

  const auto block_count = static_cast<unsigned int>((left.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  VectorAddKernel<<<block_count, static_cast<unsigned int>(THREADS_PER_BLOCK)>>>(
      device_left.Data(), device_right.Data(), device_result.Data(), left.size());
  CheckCuda(cudaGetLastError(), "launch vector add kernel");

  CheckCuda(cudaMemcpy(result.data(), device_result.Data(), byte_count, cudaMemcpyDeviceToHost), "copy result to host");
  return result;
}

}  // namespace ttl
