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

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override;
  [[nodiscard]] auto GetRecords() const -> std::vector<ErrorRecord>;

 private:
  mutable std::mutex mutex_;
  std::vector<ErrorRecord> records_;
};

class SingleDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  [[nodiscard]] auto GetRuntime() -> Runtime & { return *runtime_; }
  [[nodiscard]] auto GetContext() -> ExecutionContext & { return *context_; }
  [[nodiscard]] auto GetDevice() const -> Device { return device_.value(); }

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

  std::shared_ptr<RecordingErrorSink> error_sink_;
  std::unique_ptr<Runtime> runtime_;
  std::vector<ExecutionContext> contexts_;
  std::vector<Device> devices_;
};

}  // namespace ttl::test
