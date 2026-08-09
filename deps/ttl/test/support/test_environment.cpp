#include "support/test_environment.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

namespace ttl::test {

auto GetCudaDeviceCount() -> size_t {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
    static_cast<void>(cudaGetLastError());
    return 0;
  }
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{"cudaGetDeviceCount failed: "} + cudaGetErrorString(status));
  }
  return static_cast<size_t>(count);
}

auto GetTestDevices(size_t count) -> std::vector<Device> {
  const size_t available = GetCudaDeviceCount();
  const size_t selected = std::min(count, available);
  std::vector<Device> devices;
  devices.reserve(selected);
  for (size_t index = 0; index < selected; ++index) {
    devices.emplace_back(static_cast<int32_t>(index));
  }
  return devices;
}

void RecordingErrorSink::Report(ErrorRecord error) noexcept {
  try {
    const std::scoped_lock lock{mutex_};
    records_.push_back(std::move(error));
  } catch (...) {
    // ErrorSink is a noexcept last-resort reporting boundary.
    return;
  }
}

auto RecordingErrorSink::GetRecords() const -> std::vector<ErrorRecord> {
  const std::scoped_lock lock{mutex_};
  return records_;
}

void SingleDeviceTest::SetUp() {
  auto devices = GetTestDevices(1);
  if (devices.empty()) {
    GTEST_SKIP() << "TTL CUDA tests require one SM80+ NVIDIA GPU";
  }

  device_ = devices.front();
  error_sink_ = std::make_shared<RecordingErrorSink>();
  RuntimeOptions options;
  options.devices_ = std::move(devices);
  options.error_sink_ = error_sink_;
  runtime_ = std::make_unique<Runtime>(std::move(options));
  context_.emplace(runtime_->CreateExecutionContext(*device_));
}

void SingleDeviceTest::TearDown() {
  if (context_.has_value()) {
    try {
      context_->Synchronize();
    } catch (const std::exception &error) {
      ADD_FAILURE() << "context synchronization failed during teardown: " << error.what();
    }
    context_.reset();
  }
  if (runtime_ != nullptr) {
    try {
      runtime_->Shutdown();
    } catch (const std::exception &error) {
      ADD_FAILURE() << "runtime shutdown failed during teardown: " << error.what();
    }
    runtime_.reset();
  }
}

void MultiDeviceTest::SetUp() {
  devices_ = GetTestDevices(2);
  if (devices_.size() < 2) {
    GTEST_SKIP() << "TTL multi-GPU tests require at least two SM80+ NVIDIA GPUs";
  }

  error_sink_ = std::make_shared<RecordingErrorSink>();
  RuntimeOptions options;
  options.devices_ = devices_;
  options.error_sink_ = error_sink_;
  runtime_ = std::make_unique<Runtime>(std::move(options));
  contexts_.reserve(devices_.size());
  for (Device device : devices_) {
    contexts_.push_back(runtime_->CreateExecutionContext(device));
  }
}

void MultiDeviceTest::TearDown() {
  for (ExecutionContext &context : contexts_) {
    try {
      context.Synchronize();
    } catch (const std::exception &error) {
      ADD_FAILURE() << "context synchronization failed during teardown: " << error.what();
    }
  }
  contexts_.clear();
  if (runtime_ != nullptr) {
    try {
      runtime_->Shutdown();
    } catch (const std::exception &error) {
      ADD_FAILURE() << "runtime shutdown failed during teardown: " << error.what();
    }
    runtime_.reset();
  }
}

}  // namespace ttl::test
