#include <zephyr/vector_adder.hpp>

#include <ttl/common/device.hpp>
#include <ttl/common/error_sink.hpp>
#include <ttl/ops/copy.hpp>
#include <ttl/ops/elementwise.hpp>
#include <ttl/runtime/runtime.hpp>
#include <ttl/tensor/dtype.hpp>
#include <ttl/tensor/shape.hpp>
#include <ttl/tensor/tensor.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace zephyr {
namespace {

class VectorAdderErrorSink final : public ttl::ErrorSink {
 public:
  void Report(ttl::ErrorRecord error) noexcept override {
    static_cast<void>(error);
    failed_.store(true, std::memory_order_relaxed);
  }

  void ThrowIfFailed() const {
    if (failed_.load(std::memory_order_relaxed)) {
      throw std::runtime_error("TTL reported an asynchronous cleanup failure");
    }
  }

 private:
  std::atomic<bool> failed_{false};
};

}  // namespace

auto VectorAdder::Add(const std::vector<float> &left, const std::vector<float> &right) const -> std::vector<float> {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector sizes must match");
  }
  if (left.empty()) {
    return {};
  }

  auto error_sink = std::make_shared<VectorAdderErrorSink>();
  auto options = ttl::RuntimeOptions{};
  options.devices_ = {ttl::Device{0}};
  options.error_sink_ = error_sink;
  ttl::Runtime runtime{std::move(options)};

  std::vector<float> result(left.size());
  {
    auto context = runtime.CreateExecutionContext(ttl::Device{0});
    const auto shape = ttl::Shape{static_cast<int64_t>(left.size())};
    auto left_tensor = ttl::Empty(context, shape, ttl::DType::FLOAT32);
    auto right_tensor = ttl::Empty(context, shape, ttl::DType::FLOAT32);
    ttl::CopyFromHostBlocking(context, left_tensor, std::as_bytes(std::span{left}));
    ttl::CopyFromHostBlocking(context, right_tensor, std::as_bytes(std::span{right}));
    auto result_tensor = ttl::Add(context, left_tensor, right_tensor);
    ttl::CopyToHostBlocking(context, std::as_writable_bytes(std::span{result}), result_tensor);
  }
  runtime.Shutdown();
  error_sink->ThrowIfFailed();
  return result;
}

}  // namespace zephyr
