#include "ttl/ops/copy.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/pinned_buffer.hpp"
#include "ttl/runtime.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord error) noexcept override {
    try {
      std::scoped_lock lock{latch_};
      errors_.push_back(std::move(error));
    } catch (...) {
      return;
    }
  }

  [[nodiscard]] auto IsEmpty() const noexcept -> bool {
    std::scoped_lock lock{latch_};
    return errors_.empty();
  }

 private:
  mutable std::mutex latch_;
  std::vector<ErrorRecord> errors_;
};

class HostTransferTest : public testing::Test {
 protected:
  void SetUp() override {
    error_sink_ = std::make_shared<RecordingErrorSink>();
    runtime_ = std::make_unique<Runtime>(RuntimeOptions{
        .devices_ = {Device{0}},
        .device_memory_ =
            {
                .enable_maintenance_thread_ = false,
            },
        .error_sink_ = error_sink_,
        .pinned_memory_ = {},
    });
  }

  void TearDown() override {
    runtime_->Shutdown();
    EXPECT_TRUE(error_sink_->IsEmpty());
  }

  std::shared_ptr<RecordingErrorSink> error_sink_;
  std::unique_ptr<Runtime> runtime_;
};

TEST_F(HostTransferTest, CopiesPageableHostMemoryThroughExplicitBlockingBoundaries) {
  auto context = runtime_->CreateExecutionContext(Device{0});
  auto tensor = Empty(context, Shape{6}, DType::INT32);
  constexpr std::array<int32_t, 6> input{1, 1, 2, 3, 5, 8};
  std::array<int32_t, 6> output{};

  CopyFromHostBlocking(context, tensor, std::as_bytes(std::span{input}));
  CopyToHostBlocking(context, std::as_writable_bytes(std::span{output}), tensor);

  EXPECT_EQ(output, input);
}

TEST_F(HostTransferTest, PreservesPinnedBuffersUntilAsynchronousTransfersComplete) {
  auto context = runtime_->CreateExecutionContext(Device{0});
  auto tensor = Empty(context, Shape{8}, DType::INT64);
  constexpr std::array<int64_t, 8> input{3, 1, 4, 1, 5, 9, 2, 6};

  {
    auto pinned_input = runtime_->AllocatePinned(sizeof(input));
    std::memcpy(pinned_input.GetData(), input.data(), sizeof(input));
    CopyFromPinnedAsync(context, tensor, pinned_input);
  }

  auto pinned_output = runtime_->AllocatePinned(sizeof(input));
  CopyToPinnedAsync(context, pinned_output, tensor);
  context.Synchronize();

  std::array<int64_t, 8> output{};
  std::memcpy(output.data(), pinned_output.GetData(), sizeof(output));
  EXPECT_EQ(output, input);
}

TEST_F(HostTransferTest, RepresentsZeroBytePinnedMemoryWithoutCallingCudaCopy) {
  auto context = runtime_->CreateExecutionContext(Device{0});
  auto tensor = Empty(context, Shape{0}, DType::FLOAT32);
  auto pinned = runtime_->AllocatePinned(0);

  EXPECT_EQ(pinned.GetData(), nullptr);
  EXPECT_EQ(pinned.GetSizeBytes(), 0);
  EXPECT_TRUE(pinned.AsBytes().empty());
  EXPECT_NO_THROW(CopyFromPinnedAsync(context, tensor, pinned));
  EXPECT_NO_THROW(CopyToPinnedAsync(context, pinned, tensor));
}

}  // namespace
}  // namespace ttl
