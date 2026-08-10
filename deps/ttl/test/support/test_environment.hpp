#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ttl/common/error_sink.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/runtime.hpp"

namespace ttl::test {

[[nodiscard]] auto GetCudaDeviceCount() -> size_t;
[[nodiscard]] auto GetTestDevices(size_t count) -> std::vector<Device>;
/** Return the index-th supported test device in discovery order. */
[[nodiscard]] auto GetTestDevice(size_t index) -> Device;

/** Build runtime options for tests without embedding a machine-specific device inventory. */
[[nodiscard]] inline auto MakeRuntimeOptions(std::vector<Device> devices, std::shared_ptr<ErrorSink> error_sink)
    -> RuntimeOptions {
  RuntimeOptions options;
  options.devices_ = std::move(devices);
  options.error_sink_ = std::move(error_sink);
  return options;
}

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override;
  [[nodiscard]] auto GetRecords() const -> std::vector<ErrorRecord>;

 private:
  mutable std::mutex mutex_;
  std::vector<ErrorRecord> records_;
};

/** Selects one supported device without creating a Runtime. Use for runtime-construction tests. */
class CudaDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override;
  [[nodiscard]] auto GetDevice() const -> Device { return device_.value(); }

 private:
  std::optional<Device> device_;
};

class SingleDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  [[nodiscard]] auto GetRuntime() -> Runtime & { return *runtime_; }
  [[nodiscard]] auto GetContext() -> ExecutionContext & { return *context_; }
  [[nodiscard]] auto GetDevice() const -> Device { return device_.value(); }
  void ShutdownRuntime();

  std::shared_ptr<RecordingErrorSink> error_sink_;
  std::unique_ptr<Runtime> runtime_;
  std::optional<ExecutionContext> context_;
  std::optional<Device> device_;
};

class MultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  [[nodiscard]] auto GetRuntime() -> Runtime & { return *runtime_; }
  [[nodiscard]] auto GetContexts() -> std::vector<ExecutionContext> & { return contexts_; }
  [[nodiscard]] auto GetDevices() const -> std::span<const Device> { return devices_; }
  void ShutdownRuntime();

  std::shared_ptr<RecordingErrorSink> error_sink_;
  std::unique_ptr<Runtime> runtime_;
  std::vector<ExecutionContext> contexts_;
  std::vector<Device> devices_;
};

}  // namespace ttl::test
