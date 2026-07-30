#include "ttl/tensor.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <utility>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "ttl/device.hpp"
#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/error_sink.hpp"
#include "ttl/internal/allocation.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/device_allocator.hpp"
#include "ttl/internal/event_pool.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/shape.hpp"
#include "ttl/stream.hpp"

namespace ttl::internal {
namespace {

constexpr size_t FAKE_RESOURCE_COUNT = 64;
constexpr size_t FAKE_ALLOCATION_BYTES = 4096;

struct alignas(256) FakeAllocation final {
  std::array<std::byte, FAKE_ALLOCATION_BYTES> bytes_;
};

std::array<FakeAllocation, FAKE_RESOURCE_COUNT> fake_allocations;
std::array<int, FAKE_RESOURCE_COUNT> fake_streams;
std::array<int, FAKE_RESOURCE_COUNT> fake_events;
int fake_pool;
size_t fake_allocation_index = 0;
size_t fake_stream_index = 0;
size_t fake_event_index = 0;
int fake_current_device = 0;

void ResetFakeCuda() {
  fake_allocation_index = 0;
  fake_stream_index = 0;
  fake_event_index = 0;
  fake_current_device = 0;
}

auto FakeGetDevice(int *device) -> cudaError_t {
  *device = fake_current_device;
  return cudaSuccess;
}

auto FakeSetDevice(int device) -> cudaError_t {
  fake_current_device = device;
  return cudaSuccess;
}

auto FakeGetStreamPriorityRange(int *least_priority, int *greatest_priority) -> cudaError_t {
  *least_priority = 0;
  *greatest_priority = 0;
  return cudaSuccess;
}

auto FakeCreateStream(cudaStream_t *stream, unsigned int flags, int priority) -> cudaError_t {
  if (flags != cudaStreamNonBlocking || priority != 0 || fake_stream_index == fake_streams.size()) {
    return cudaErrorInvalidValue;
  }
  *stream = reinterpret_cast<cudaStream_t>(&fake_streams[fake_stream_index]);
  fake_stream_index++;
  return cudaSuccess;
}

auto FakeDestroyStream(cudaStream_t /*stream*/) -> cudaError_t { return cudaSuccess; }

auto FakeCreateEvent(cudaEvent_t *event, unsigned int flags) -> cudaError_t {
  if (flags != cudaEventDisableTiming || fake_event_index == fake_events.size()) {
    return cudaErrorInvalidValue;
  }
  *event = reinterpret_cast<cudaEvent_t>(&fake_events[fake_event_index]);
  fake_event_index++;
  return cudaSuccess;
}

auto FakeRecordEvent(cudaEvent_t /*event*/, cudaStream_t /*stream*/) -> cudaError_t { return cudaSuccess; }

auto FakeQueryEvent(cudaEvent_t /*event*/) -> cudaError_t { return cudaSuccess; }

auto FakeSynchronizeEvent(cudaEvent_t /*event*/) -> cudaError_t { return cudaSuccess; }

auto FakeDestroyEvent(cudaEvent_t /*event*/) -> cudaError_t { return cudaSuccess; }

auto FakeSynchronizeStream(cudaStream_t /*stream*/) -> cudaError_t { return cudaSuccess; }

auto FakeCreatePool(cudaMemPool_t *pool, const cudaMemPoolProps *properties) -> cudaError_t {
  if (properties->location.type != cudaMemLocationTypeDevice || properties->location.id != 0) {
    return cudaErrorInvalidValue;
  }
  *pool = reinterpret_cast<cudaMemPool_t>(&fake_pool);
  return cudaSuccess;
}

auto FakeDestroyPool(cudaMemPool_t /*pool*/) -> cudaError_t { return cudaSuccess; }

auto FakeSetPoolAttribute(cudaMemPool_t /*pool*/, cudaMemPoolAttr /*attribute*/, void * /*value*/) -> cudaError_t {
  return cudaSuccess;
}

auto FakeGetPoolAttribute(cudaMemPool_t /*pool*/, cudaMemPoolAttr /*attribute*/, void *value) -> cudaError_t {
  *static_cast<uint64_t *>(value) = 0;
  return cudaSuccess;
}

auto FakeTrimPool(cudaMemPool_t /*pool*/, size_t /*minimum_bytes*/) -> cudaError_t { return cudaSuccess; }

auto FakeMallocFromPool(void **pointer, size_t bytes, cudaMemPool_t /*pool*/, cudaStream_t /*stream*/) -> cudaError_t {
  if (bytes > FAKE_ALLOCATION_BYTES || fake_allocation_index == fake_allocations.size()) {
    return cudaErrorMemoryAllocation;
  }
  *pointer = fake_allocations[fake_allocation_index].bytes_.data();
  fake_allocation_index++;
  return cudaSuccess;
}

auto FakeFreeAsync(void * /*pointer*/, cudaStream_t /*stream*/) -> cudaError_t { return cudaSuccess; }

auto FakeGetPointerAttributes(cudaPointerAttributes *attributes, const void * /*pointer*/) -> cudaError_t {
  attributes->type = cudaMemoryTypeDevice;
  attributes->device = 0;
  return cudaSuccess;
}

[[nodiscard]] auto MakeFakeCudaApi() -> CudaApi {
  auto cuda_api = GetCudaApi();
  cuda_api.get_device_ = FakeGetDevice;
  cuda_api.set_device_ = FakeSetDevice;
  cuda_api.get_stream_priority_range_ = FakeGetStreamPriorityRange;
  cuda_api.create_stream_with_priority_ = FakeCreateStream;
  cuda_api.destroy_stream_ = FakeDestroyStream;
  cuda_api.create_event_with_flags_ = FakeCreateEvent;
  cuda_api.record_event_ = FakeRecordEvent;
  cuda_api.query_event_ = FakeQueryEvent;
  cuda_api.synchronize_event_ = FakeSynchronizeEvent;
  cuda_api.destroy_event_ = FakeDestroyEvent;
  cuda_api.synchronize_stream_ = FakeSynchronizeStream;
  cuda_api.create_memory_pool_ = FakeCreatePool;
  cuda_api.destroy_memory_pool_ = FakeDestroyPool;
  cuda_api.set_memory_pool_attribute_ = FakeSetPoolAttribute;
  cuda_api.get_memory_pool_attribute_ = FakeGetPoolAttribute;
  cuda_api.trim_memory_pool_ = FakeTrimPool;
  cuda_api.malloc_from_pool_async_ = FakeMallocFromPool;
  cuda_api.free_async_ = FakeFreeAsync;
  cuda_api.get_pointer_attributes_ = FakeGetPointerAttributes;
  return cuda_api;
}

const CudaApi FAKE_CUDA_API = MakeFakeCudaApi();

class RecordingErrorSink final : public ErrorSink {
 public:
  void Report(ErrorRecord /*error*/) noexcept override { report_count_.fetch_add(1, std::memory_order_relaxed); }

  [[nodiscard]] auto GetReportCount() const noexcept -> size_t { return report_count_.load(std::memory_order_relaxed); }

 private:
  std::atomic<size_t> report_count_{0};
};

class TensorTest : public testing::Test {
 protected:
  void SetUp() override {
    ResetFakeCuda();
    error_sink_ = std::make_shared<RecordingErrorSink>();
    event_pool_ = std::make_shared<EventPool>(Device{0}, error_sink_, FAKE_RESOURCE_COUNT);
    allocator_ = DeviceAllocator::Create(Device{0}, error_sink_, event_pool_,
                                         DeviceAllocatorOptions{.enable_maintenance_thread_ = false});
    allocation_stream_ = std::make_unique<Stream>(StreamAccess::WrapExternal(
        Device{0}, reinterpret_cast<cudaStream_t>(&fake_streams.back()), nullptr, error_sink_));
  }

  void TearDown() override {
    allocation_stream_.reset();
    allocator_->Poll();
    allocator_->Shutdown();
    event_pool_->Close();
    EXPECT_EQ(error_sink_->GetReportCount(), 0);
  }

  [[nodiscard]] auto MakeStorage(size_t bytes) -> std::shared_ptr<Storage> {
    return allocator_->Allocate(*allocation_stream_, bytes, 256,
                                AllocationContext{
                                    .operation_ = "tensor test",
                                    .output_shape_ = std::nullopt,
                                    .dtype_ = std::nullopt,
                                    .location_ = std::source_location::current(),
                                });
  }

  [[nodiscard]] auto WrapStorage(void *pointer, size_t bytes) -> std::shared_ptr<Storage> {
    return allocator_->WrapExternal(*allocation_stream_, pointer, bytes, ExternalOwnership::BORROWED, nullptr);
  }

  std::shared_ptr<RecordingErrorSink> error_sink_;
  std::shared_ptr<EventPool> event_pool_;
  std::shared_ptr<DeviceAllocator> allocator_;
  std::unique_ptr<Stream> allocation_stream_;

 private:
  ScopedCudaApiOverride cuda_api_override_{FAKE_CUDA_API};
};

TEST(TensorFlagsTest, StoresIndependentFlags) {
  TensorFlags flags;
  EXPECT_FALSE(flags.Has(TensorFlag::CONTIGUOUS));
  EXPECT_FALSE(flags.Has(TensorFlag::HAS_ZERO_STRIDE));
  EXPECT_FALSE(flags.Has(TensorFlag::NON_OVERLAPPING_DENSE));

  flags.Set(TensorFlag::CONTIGUOUS);
  flags.Set(TensorFlag::NON_OVERLAPPING_DENSE);

  EXPECT_TRUE(flags.Has(TensorFlag::CONTIGUOUS));
  EXPECT_FALSE(flags.Has(TensorFlag::HAS_ZERO_STRIDE));
  EXPECT_TRUE(flags.Has(TensorFlag::NON_OVERLAPPING_DENSE));
}

TEST_F(TensorTest, ExposesImmutableMetadataAndTypedDevicePointer) {
  auto storage = MakeStorage(64);
  const auto *base_pointer = static_cast<float *>(storage->GetBasePointer());
  auto tensor = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 3}, Strides{3, 1}, 1);

  EXPECT_EQ(tensor.GetDevice(), Device{0});
  EXPECT_EQ(tensor.GetDType(), DType::FLOAT32);
  EXPECT_EQ(tensor.GetShape(), Shape({2, 3}));
  EXPECT_EQ(tensor.GetStrides(), Strides({3, 1}));
  EXPECT_EQ(tensor.GetRank(), 2);
  EXPECT_EQ(tensor.GetNumElements(), 6);
  EXPECT_EQ(tensor.GetStorageOffset(), 1);
  EXPECT_EQ(tensor.GetData<float>(), base_pointer + 1);
  EXPECT_EQ(TensorAccess::GetMutableData<float>(tensor), base_pointer + 1);
  EXPECT_THROW([[maybe_unused]] const auto *data = tensor.GetData<int32_t>(), InvalidArgumentError);
}

TEST_F(TensorTest, RepresentsScalarAndZeroByteEmptyTensor) {
  {
    auto storage = MakeStorage(sizeof(float));
    const auto scalar = TensorFactory::Create(storage, DType::FLOAT32, Shape{}, Strides{}, 0);

    EXPECT_TRUE(scalar.GetShape().IsScalar());
    EXPECT_EQ(scalar.GetNumElements(), 1);
    EXPECT_TRUE(scalar.IsContiguous());
    EXPECT_TRUE(scalar.IsNonOverlappingDense());
    EXPECT_FALSE(scalar.HasZeroStride());
  }

  {
    auto storage = MakeStorage(0);
    const auto empty = TensorFactory::Create(storage, DType::FLOAT32, Shape{0, 3}, Strides{3, 1}, 0);

    EXPECT_TRUE(empty.GetShape().IsEmpty());
    EXPECT_EQ(empty.GetData<float>(), nullptr);
    EXPECT_TRUE(empty.IsContiguous());
    EXPECT_TRUE(empty.IsNonOverlappingDense());
  }
}

TEST_F(TensorTest, AllowsEmptyTensorAtOnePastStorageEnd) {
  auto storage = MakeStorage(16);
  const auto *base_pointer = static_cast<std::byte *>(storage->GetBasePointer());
  const auto empty = TensorFactory::Create(storage, DType::FLOAT32, Shape{0}, Strides{1}, 4);

  EXPECT_EQ(static_cast<const void *>(empty.GetData<float>()), base_pointer + 16);
  EXPECT_THROW(
      [[maybe_unused]] const auto tensor = TensorFactory::Create(storage, DType::FLOAT32, Shape{0}, Strides{1}, 5),
      InvalidArgumentError);
}

TEST_F(TensorTest, RejectsInvalidMetadataAndPreservesCallSite) {
  auto storage = MakeStorage(64);

  EXPECT_THROW(
      [[maybe_unused]] const auto tensor = TensorFactory::Create(nullptr, DType::FLOAT32, Shape{1}, Strides{1}, 0),
      InvalidArgumentError);
  EXPECT_THROW(
      [[maybe_unused]] const auto tensor = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 2}, Strides{1}, 0),
      InvalidArgumentError);
  EXPECT_THROW(
      [[maybe_unused]] const auto tensor = TensorFactory::Create(storage, DType::FLOAT32, Shape{1}, Strides{1}, -1),
      InvalidArgumentError);
  EXPECT_THROW(
      [[maybe_unused]] const auto tensor = TensorFactory::Create(storage, DType::FLOAT32, Shape{17}, Strides{1}, 0),
      InvalidArgumentError);

  const auto invalid_dtype = std::bit_cast<DType>(uint8_t{255});
  EXPECT_THROW(
      [[maybe_unused]] const auto tensor = TensorFactory::Create(storage, invalid_dtype, Shape{1}, Strides{1}, 0),
      InvalidArgumentError);

  const auto location = std::source_location::current();
  try {
    [[maybe_unused]] const auto tensor =
        TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 2}, Strides{1}, 0, location);
    FAIL() << "expected an out-of-bounds Tensor to be rejected";
  } catch (const InvalidArgumentError &error) {
    EXPECT_EQ(error.GetLocation().line(), location.line());
  }
}

TEST_F(TensorTest, ChecksReachableRangeAndArithmeticOverflow) {
  auto storage = MakeStorage(32);

  EXPECT_NO_THROW([[maybe_unused]] const auto tensor =
                      TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 2}, Strides{2, 1}, 4));
  EXPECT_THROW([[maybe_unused]] const auto tensor =
                   TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 2}, Strides{2, 1}, 5),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto tensor = TensorFactory::Create(
                   storage, DType::UINT8, Shape{2, 2}, Strides{std::numeric_limits<int64_t>::max(), 1}, 0),
               OverflowError);
  EXPECT_THROW([[maybe_unused]] const auto tensor = TensorFactory::Create(storage, DType::INT64, Shape{}, Strides{},
                                                                          std::numeric_limits<int64_t>::max()),
               OverflowError);
}

TEST_F(TensorTest, RejectsMisalignedActualDataPointer) {
  auto *misaligned_pointer = fake_allocations.back().bytes_.data() + 1;
  auto storage = WrapStorage(misaligned_pointer, 32);

  EXPECT_THROW(
      [[maybe_unused]] const auto tensor = TensorFactory::Create(storage, DType::FLOAT32, Shape{1}, Strides{1}, 0),
      InvalidArgumentError);
  EXPECT_NO_THROW([[maybe_unused]] const auto tensor =
                      TensorFactory::Create(storage, DType::UINT8, Shape{1}, Strides{1}, 0));
}

TEST_F(TensorTest, ComputesContiguousAndDenseFlagsForStridedLayouts) {
  auto storage = MakeStorage(1024);
  const auto contiguous = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 3}, Strides{3, 1}, 0);
  const auto transpose = TensorFactory::Create(storage, DType::FLOAT32, Shape{3, 2}, Strides{1, 3}, 0);
  const auto slice = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 2}, Strides{3, 1}, 0);
  const auto broadcast = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 3}, Strides{0, 1}, 0);
  const auto size_one = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 1, 3}, Strides{3, 999, 1}, 0);
  const auto empty = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 0, 3}, Strides{0, 0, 0}, 0);

  EXPECT_TRUE(contiguous.IsContiguous());
  EXPECT_TRUE(contiguous.IsNonOverlappingDense());
  EXPECT_FALSE(contiguous.HasZeroStride());

  EXPECT_FALSE(transpose.IsContiguous());
  EXPECT_TRUE(transpose.IsNonOverlappingDense());
  EXPECT_FALSE(transpose.HasZeroStride());

  EXPECT_FALSE(slice.IsContiguous());
  EXPECT_FALSE(slice.IsNonOverlappingDense());

  EXPECT_FALSE(broadcast.IsContiguous());
  EXPECT_FALSE(broadcast.IsNonOverlappingDense());
  EXPECT_TRUE(broadcast.HasZeroStride());

  EXPECT_TRUE(size_one.IsContiguous());
  EXPECT_TRUE(size_one.IsNonOverlappingDense());
  EXPECT_FALSE(size_one.HasZeroStride());

  EXPECT_TRUE(empty.IsContiguous());
  EXPECT_TRUE(empty.IsNonOverlappingDense());
  EXPECT_TRUE(empty.HasZeroStride());
}

TEST_F(TensorTest, ClassifiesExactDisjointAndPotentiallyOverlappingAliases) {
  auto storage = MakeStorage(128);
  const auto original = TensorFactory::Create(storage, DType::FLOAT32, Shape{3}, Strides{1}, 0);
  const auto exact = TensorFactory::Create(storage, DType::FLOAT32, Shape{3}, Strides{1}, 0);
  const auto overlapping = TensorFactory::Create(storage, DType::FLOAT32, Shape{3}, Strides{1}, 1);
  const auto disjoint = TensorFactory::Create(storage, DType::FLOAT32, Shape{3}, Strides{1}, 4);
  const auto sparse_interval = TensorFactory::Create(storage, DType::FLOAT32, Shape{2}, Strides{2}, 0);
  auto other_storage = MakeStorage(128);
  const auto other = TensorFactory::Create(other_storage, DType::FLOAT32, Shape{3}, Strides{1}, 0);

  EXPECT_EQ(ClassifyAlias(original, exact), AliasKind::EXACT);
  EXPECT_EQ(ClassifyAlias(original, overlapping), AliasKind::MAY_OVERLAP);
  EXPECT_EQ(ClassifyAlias(original, disjoint), AliasKind::DISJOINT);
  EXPECT_EQ(ClassifyAlias(overlapping, sparse_interval), AliasKind::MAY_OVERLAP);
  EXPECT_EQ(ClassifyAlias(original, other), AliasKind::DISJOINT);
}

TEST_F(TensorTest, HandlesEmptyAliasSemantics) {
  auto storage = MakeStorage(32);
  const auto empty = TensorFactory::Create(storage, DType::FLOAT32, Shape{0}, Strides{1}, 0);
  const auto exact_empty = TensorFactory::Create(storage, DType::FLOAT32, Shape{0}, Strides{1}, 0);
  const auto other_empty = TensorFactory::Create(storage, DType::FLOAT32, Shape{0}, Strides{1}, 1);

  EXPECT_EQ(ClassifyAlias(empty, exact_empty), AliasKind::EXACT);
  EXPECT_EQ(ClassifyAlias(empty, other_empty), AliasKind::DISJOINT);
}

TEST_F(TensorTest, DetectsAliasingAcrossDistinctExternalStorageWrappers) {
  auto *pointer = fake_allocations.back().bytes_.data();
  auto lhs_storage = WrapStorage(pointer, 32);
  auto rhs_storage = WrapStorage(pointer, 32);
  const auto lhs = TensorFactory::Create(lhs_storage, DType::FLOAT32, Shape{4}, Strides{1}, 0);
  const auto rhs = TensorFactory::Create(rhs_storage, DType::FLOAT32, Shape{4}, Strides{1}, 0);

  EXPECT_EQ(ClassifyAlias(lhs, rhs), AliasKind::MAY_OVERLAP);
}

TEST_F(TensorTest, SharedTensorHandlesKeepStorageAlive) {
  auto storage = MakeStorage(64);
  EXPECT_EQ(allocator_->GetStats().outstanding_storage_count_, 1);

  {
    const auto tensor = TensorFactory::Create(storage, DType::FLOAT32, Shape{4}, Strides{1}, 0);
    const auto tensor_copy = tensor;  // NOLINT(performance-unnecessary-copy-initialization)
    storage.reset();

    EXPECT_EQ(tensor_copy.GetData<float>(), tensor.GetData<float>());
    EXPECT_EQ(allocator_->GetStats().outstanding_storage_count_, 1);
  }

  allocator_->Poll();
  EXPECT_EQ(allocator_->GetStats().outstanding_storage_count_, 0);
}

TEST_F(TensorTest, RejectsMovedFromTensorAtCheckedOperationBoundaries) {
  auto storage = MakeStorage(64);
  auto tensor = TensorFactory::Create(storage, DType::FLOAT32, Shape{4}, Strides{1}, 0);
  const auto moved_tensor = std::move(tensor);

  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW([[maybe_unused]] const auto *data = tensor.GetData<float>(), InvalidArgumentError);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW([[maybe_unused]] const auto alias = ClassifyAlias(tensor, moved_tensor), InvalidArgumentError);
}

}  // namespace
}  // namespace ttl::internal
