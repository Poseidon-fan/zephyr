#include "parallel/tp.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <ttl/common/error.hpp>
#include <ttl/common/error_sink.hpp>
#include <ttl/ops/copy.hpp>
#include <ttl/runtime/runtime.hpp>
#include <ttl/tensor/tensor.hpp>

namespace zephyr::parallel {
namespace {

class TestErrorSink final : public ttl::ErrorSink {
 public:
  void Report(ttl::ErrorRecord error) noexcept override {
    try {
      const std::scoped_lock lock{mutex_};
      records_.push_back(std::move(error));
    } catch (...) {
    }
  }

 private:
  std::mutex mutex_;
  std::vector<ttl::ErrorRecord> records_;
};

auto SupportedDevices() -> std::vector<ttl::Device> {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
    static_cast<void>(cudaGetLastError());
    return {};
  }
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{"cudaGetDeviceCount failed: "} + cudaGetErrorString(status));
  }

  std::vector<ttl::Device> devices;
  for (int index = 0; index < count; ++index) {
    cudaDeviceProp properties{};
    const cudaError_t property_status = cudaGetDeviceProperties(&properties, index);
    if (property_status != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      throw std::runtime_error(std::string{"cudaGetDeviceProperties failed: "} + cudaGetErrorString(property_status));
    }
    if ((properties.major * 10) + properties.minor >= 80) {
      devices.emplace_back(index);
    }
  }
  return devices;
}

auto MakeRuntime(const std::vector<ttl::Device> &devices) -> std::unique_ptr<ttl::Runtime> {
  ttl::RuntimeOptions options;
  options.devices_ = devices;
  options.error_sink_ = std::make_shared<TestErrorSink>();
  return std::make_unique<ttl::Runtime>(std::move(options));
}

auto Upload(ttl::ExecutionContext &context, std::span<const float> values) -> ttl::Tensor {
  auto tensor = ttl::Empty(context, ttl::Shape{static_cast<int64_t>(values.size())}, ttl::DType::FLOAT32);
  const auto bytes =
      std::span<const std::byte>{reinterpret_cast<const std::byte *>(values.data()), values.size_bytes()};
  ttl::CopyFromHostBlocking(context, tensor, bytes);
  return tensor;
}

auto Download(ttl::ExecutionContext &context, const ttl::Tensor &tensor) -> std::vector<float> {
  std::vector<float> values(static_cast<size_t>(tensor.GetNumElements()));
  const auto bytes = std::span<std::byte>{reinterpret_cast<std::byte *>(values.data()), values.size() * sizeof(float)};
  ttl::CopyToHostBlocking(context, bytes, tensor);
  return values;
}

TEST(TpContextTest, PreservesExternalDeviceOrderAndSupportsSubset) {
  const auto devices = SupportedDevices();
  if (devices.empty()) {
    GTEST_SKIP() << "requires one SM80+ CUDA device";
  }

  auto runtime = MakeRuntime(devices);
  std::vector<ttl::Device> selected{devices.front()};
  if (devices.size() >= 2) {
    selected = {devices[1], devices[0]};
  }
  auto context = TpContext::Create(*runtime, selected);
  EXPECT_EQ(context->GetStatus(), ttl::CommunicatorStatus::READY);
  ASSERT_EQ(context->WorldSize(), selected.size());
  for (size_t rank = 0; rank < selected.size(); ++rank) {
    const auto rank_context = context->GetRank(rank);
    EXPECT_EQ(rank_context.Rank(), rank);
    EXPECT_EQ(rank_context.WorldSize(), selected.size());
    EXPECT_EQ(rank_context.Device(), selected[rank]);
  }
  context->Close();
  EXPECT_EQ(context->GetStatus(), ttl::CommunicatorStatus::CLOSED);
  context.reset();
  runtime->Shutdown();
}

TEST(TpContextTest, RejectsInvalidDeviceListsAndRanks) {
  const auto devices = SupportedDevices();
  if (devices.empty()) {
    GTEST_SKIP() << "requires one SM80+ CUDA device";
  }

  auto runtime = MakeRuntime(devices);
  EXPECT_THROW(static_cast<void>(TpContext::Create(*runtime, std::span<const ttl::Device>{})),
               ttl::InvalidArgumentError);

  const std::array duplicate{devices[0], devices[0]};
  EXPECT_THROW(static_cast<void>(TpContext::Create(*runtime, duplicate)), ttl::InvalidArgumentError);

  const auto unregistered_ordinal = devices.back().GetOrdinal() + 1;
  const std::array unregistered{ttl::Device{unregistered_ordinal}};
  EXPECT_THROW(static_cast<void>(TpContext::Create(*runtime, unregistered)), ttl::InvalidArgumentError);

  const std::array valid{devices[0]};
  auto context = TpContext::Create(*runtime, valid);
  EXPECT_THROW(static_cast<void>(context->GetRank(1)), ttl::InvalidArgumentError);
  context->Abort();
  context.reset();
  runtime->Shutdown();
}

TEST(TpContextTest, RankViewCanBeCopiedIntoWorker) {
  const auto devices = SupportedDevices();
  if (devices.empty()) {
    GTEST_SKIP() << "requires one SM80+ CUDA device";
  }

  const std::array selected{devices.front()};
  auto runtime = MakeRuntime(std::vector<ttl::Device>{selected.front()});
  auto context = TpContext::Create(*runtime, selected);
  const auto rank = context->GetRank(0);
  size_t observed_rank = 99;
  std::thread worker([rank, &observed_rank] { observed_rank = rank.Rank(); });
  worker.join();
  EXPECT_EQ(observed_rank, 0U);
  context->Abort();
  context.reset();
  runtime->Shutdown();
}

TEST(TpContextTest, AllReduceSumSupportsOutOfPlaceAndInPlace) {
  const auto devices = SupportedDevices();
  if (devices.size() < 2) {
    GTEST_SKIP() << "requires two SM80+ CUDA devices";
  }

  const std::array selected{devices[0], devices[1]};
  auto runtime = MakeRuntime(std::vector<ttl::Device>{selected.begin(), selected.end()});
  auto context = TpContext::Create(*runtime, selected);
  {
    std::array execution_contexts{runtime->CreateExecutionContext(selected[0]),
                                  runtime->CreateExecutionContext(selected[1])};
    {
      std::array inputs{Upload(execution_contexts[0], std::array{1.0F, 2.0F}),
                        Upload(execution_contexts[1], std::array{10.0F, 20.0F})};
      std::array outputs{ttl::Empty(execution_contexts[0], ttl::Shape{2}, ttl::DType::FLOAT32),
                         ttl::Empty(execution_contexts[1], ttl::Shape{2}, ttl::DType::FLOAT32)};
      std::array failures{std::exception_ptr{}, std::exception_ptr{}};
      std::array workers{std::thread{}, std::thread{}};
      for (size_t rank = 0; rank < 2; ++rank) {
        workers[rank] = std::thread([&, rank] {
          try {
            context->GetRank(rank).AllReduce(execution_contexts[rank], outputs[rank], inputs[rank], ttl::ReduceOp::SUM);
          } catch (...) {
            failures[rank] = std::current_exception();
          }
        });
      }
      for (auto &worker : workers) {
        worker.join();
      }
      for (const auto &failure : failures) {
        if (failure != nullptr) {
          std::rethrow_exception(failure);
        }
      }
      for (auto &execution_context : execution_contexts) {
        execution_context.Synchronize();
      }
      EXPECT_EQ(Download(execution_contexts[0], outputs[0]), (std::vector<float>{11.0F, 22.0F}));
      EXPECT_EQ(Download(execution_contexts[1], outputs[1]), (std::vector<float>{11.0F, 22.0F}));

      inputs = {Upload(execution_contexts[0], std::array{3.0F, 4.0F}),
                Upload(execution_contexts[1], std::array{30.0F, 40.0F})};
      failures = {};
      for (size_t rank = 0; rank < 2; ++rank) {
        workers[rank] = std::thread([&, rank] {
          try {
            context->GetRank(rank).AllReduce(execution_contexts[rank], inputs[rank], inputs[rank], ttl::ReduceOp::SUM);
          } catch (...) {
            failures[rank] = std::current_exception();
          }
        });
      }
      for (auto &worker : workers) {
        worker.join();
      }
      for (const auto &failure : failures) {
        if (failure != nullptr) {
          std::rethrow_exception(failure);
        }
      }
      for (auto &execution_context : execution_contexts) {
        execution_context.Synchronize();
      }
      EXPECT_EQ(Download(execution_contexts[0], inputs[0]), (std::vector<float>{33.0F, 44.0F}));
      EXPECT_EQ(Download(execution_contexts[1], inputs[1]), (std::vector<float>{33.0F, 44.0F}));
    }
  }
  context->Close();
  context.reset();
  runtime->Shutdown();
}

TEST(TpContextTest, CloseAbortAndDestructorReleaseRuntimeResources) {
  const auto devices = SupportedDevices();
  if (devices.empty()) {
    GTEST_SKIP() << "requires one SM80+ CUDA device";
  }

  const std::array selected{devices.front()};
  for (int mode = 0; mode < 3; ++mode) {
    auto runtime = MakeRuntime(std::vector<ttl::Device>{selected.front()});
    {
      auto context = TpContext::Create(*runtime, selected);
      if (mode == 0) {
        context->Close();
      } else if (mode == 1) {
        context->Abort();
      }
    }
    EXPECT_NO_THROW(runtime->Shutdown());
  }
}

}  // namespace
}  // namespace zephyr::parallel
