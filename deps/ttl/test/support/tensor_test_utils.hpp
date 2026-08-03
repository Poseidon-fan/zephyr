#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "ttl/common/error.hpp"
#include "ttl/common/error_sink.hpp"
#include "ttl/ops/copy.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::test {

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override {
    try {
      const std::scoped_lock lock{latch_};
      records_.push_back(std::move(error));
    } catch (...) {
      return;
    }
  }

  [[nodiscard]] auto GetRecords() const -> std::vector<ErrorRecord> {
    const std::scoped_lock lock{latch_};
    return records_;
  }

 private:
  mutable std::mutex latch_;
  std::vector<ErrorRecord> records_;
};

[[nodiscard]] inline auto MakeRuntimeOptions(std::vector<Device> devices, std::shared_ptr<ErrorSink> error_sink)
    -> RuntimeOptions {
  auto options = RuntimeOptions{};
  options.devices_ = std::move(devices);
  options.error_sink_ = std::move(error_sink);
  return options;
}

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
  if (static_cast<size_t>(shape.GetNumElements()) != values.size()) {
    throw InvalidArgumentError("test upload shape does not match the host value count");
  }
  auto tensor = Empty(context, shape, DTYPE_OF<T>);
  CopyFromHostBlocking(context, tensor, std::as_bytes(values));
  return tensor;
}

template <TensorStorageType T>
[[nodiscard]] auto Upload(ExecutionContext &context, const Shape &shape, const std::vector<T> &values) -> Tensor {
  return Upload(context, shape, std::span<const T>{values});
}

template <TensorStorageType T>
[[nodiscard]] auto Download(ExecutionContext &context, const Tensor &tensor) -> std::vector<T> {
  std::vector<T> values(static_cast<size_t>(tensor.GetNumElements()));
  CopyToHostBlocking(context, std::as_writable_bytes(std::span<T>{values}), tensor);
  return values;
}

}  // namespace ttl::test
