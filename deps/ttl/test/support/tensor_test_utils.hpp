#pragma once

#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "support/test_environment.hpp"
#include "support/test_tensor.hpp"

namespace ttl::test {

[[nodiscard]] inline auto MakeRuntimeOptions(std::vector<Device> devices, std::shared_ptr<ErrorSink> error_sink)
    -> RuntimeOptions {
  RuntimeOptions options;
  options.devices_ = std::move(devices);
  options.error_sink_ = std::move(error_sink);
  return options;
}

/** Owns one runtime/context pair for tests that need deterministic explicit shutdown. */
class RuntimeSession final {
 public:
  explicit RuntimeSession(Device device = Device{0})
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
      } catch (...) {
        return;
      }
    }
  }

  [[nodiscard]] auto GetRuntime() noexcept -> Runtime & { return *runtime_; }
  [[nodiscard]] auto GetContext() noexcept -> ExecutionContext & { return *context_; }
  [[nodiscard]] auto GetErrorSink() const noexcept -> const std::shared_ptr<RecordingErrorSink> & {
    return error_sink_;
  }

  void Close() {
    context_.reset();
    runtime_->Shutdown();
  }

 private:
  std::shared_ptr<RecordingErrorSink> error_sink_;
  std::unique_ptr<Runtime> runtime_;
  std::optional<ExecutionContext> context_;
};

template <TensorStorageType T>
[[nodiscard]] auto Upload(ExecutionContext &context, const Shape &shape, std::span<const T> values) -> Tensor {
  return TensorFromValues<T>(context, shape, values);
}

template <TensorStorageType T>
[[nodiscard]] auto Upload(ExecutionContext &context, const Shape &shape, const std::vector<T> &values) -> Tensor {
  return Upload(context, shape, std::span<const T>{values});
}

template <TensorStorageType T>
[[nodiscard]] auto Download(ExecutionContext &context, const Tensor &tensor) -> std::vector<T> {
  if constexpr (std::same_as<T, uint8_t>) {
    if (tensor.GetDType() == DType::BOOL) {
      return BoolTensorToValues(context, tensor);
    }
  }
  return TensorToValues<T>(context, tensor);
}

}  // namespace ttl::test
