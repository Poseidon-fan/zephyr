#pragma once

#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"

namespace ttl::test {

/** Owns one runtime/context pair for tests that need deterministic explicit shutdown. */
class RuntimeSession final {
 public:
  RuntimeSession() : RuntimeSession(SelectDevice()) {}

  explicit RuntimeSession(Device device)
      : error_sink_(std::make_shared<RecordingErrorSink>()),
        runtime_(std::make_unique<Runtime>(MakeRuntimeOptions({device}, error_sink_))),
        context_(runtime_->CreateExecutionContext(device)) {}

  RuntimeSession(const RuntimeSession &) = delete;
  auto operator=(const RuntimeSession &) -> RuntimeSession & = delete;

  ~RuntimeSession() noexcept {
    context_.reset();
    if (runtime_ != nullptr) {
      try {
        runtime_->Shutdown();
      } catch (const std::exception &error) {
        ADD_FAILURE() << "runtime shutdown failed in RuntimeSession teardown: " << error.what();
      } catch (...) {
        ADD_FAILURE() << "runtime shutdown failed in RuntimeSession teardown with an unknown exception";
      }
    }
  }

  [[nodiscard]] auto GetRuntime() noexcept -> Runtime & { return *runtime_; }
  [[nodiscard]] auto GetContext() noexcept -> ExecutionContext & { return *context_; }
  [[nodiscard]] auto GetDevice() const noexcept -> Device { return context_->GetDevice(); }
  [[nodiscard]] auto GetErrorSink() const noexcept -> const std::shared_ptr<RecordingErrorSink> & {
    return error_sink_;
  }

  void Close() {
    context_.reset();
    runtime_->Shutdown();
  }

 private:
  [[nodiscard]] static auto SelectDevice() -> Device {
    const auto devices = GetTestDevices(1);
    if (devices.empty()) {
      throw std::runtime_error("TTL tests require an SM80-or-newer CUDA device");
    }
    return devices.front();
  }

  std::shared_ptr<RecordingErrorSink> error_sink_;
  std::unique_ptr<Runtime> runtime_;
  std::optional<ExecutionContext> context_;
};

}  // namespace ttl::test
