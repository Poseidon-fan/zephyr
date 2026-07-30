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
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/internal/event_pool.hpp"
#include "ttl/internal/storage.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/layout.hpp"
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

TEST_F(TensorTest, CreatesCompatibleViewsAndInfersReshapeDimensions) {
  auto storage = MakeStorage(1024);
  const auto input = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 3, 4}, Strides{12, 4, 1}, 2);

  const auto matrix = View(input, Shape{6, 4});
  EXPECT_EQ(matrix.GetShape(), Shape({6, 4}));
  EXPECT_EQ(matrix.GetStrides(), Strides({4, 1}));
  EXPECT_EQ(matrix.GetStorageOffset(), 2);
  EXPECT_EQ(matrix.GetData<float>(), input.GetData<float>());

  constexpr std::array<int64_t, 3> requested{2, -1, 2};
  EXPECT_EQ(InferReshape(input, requested), Shape({2, 6, 2}));
  const auto inferred = View(input, requested);
  EXPECT_EQ(inferred.GetShape(), Shape({2, 6, 2}));
  EXPECT_EQ(inferred.GetStrides(), Strides({12, 2, 1}));

  const auto transposed = Transpose(input, 0, 1);
  EXPECT_THROW([[maybe_unused]] const auto flattened = View(transposed, Shape{24}), InvalidArgumentError);

  const auto size_one = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 1, 3}, Strides{3, 99, 1}, 0);
  const auto removed_size_one = View(size_one, Shape{2, 3});
  EXPECT_EQ(removed_size_one.GetStrides(), Strides({3, 1}));

  auto empty_storage = MakeStorage(0);
  const auto empty = TensorFactory::Create(empty_storage, DType::FLOAT32, Shape{2, 0, 3}, Strides{0, 0, 0}, 0);
  const auto reshaped_empty = View(empty, Shape{0, 6});
  EXPECT_EQ(reshaped_empty.GetStrides(), Strides({6, 1}));
  EXPECT_EQ(reshaped_empty.GetData<float>(), nullptr);
}

TEST_F(TensorTest, RejectsInvalidReshapeRequests) {
  auto storage = MakeStorage(64);
  const auto input = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 3}, Strides{3, 1}, 0);
  const auto empty = TensorFactory::Create(MakeStorage(0), DType::FLOAT32, Shape{0, 3}, Strides{3, 1}, 0);

  constexpr std::array<int64_t, 2> multiple_inferred{-1, -1};
  constexpr std::array<int64_t, 1> negative{-2};
  constexpr std::array<int64_t, 2> wrong_count{4, 2};
  constexpr std::array<int64_t, 2> nonintegral{-1, 4};
  constexpr std::array<int64_t, 2> empty_inferred{-1, 3};

  EXPECT_THROW([[maybe_unused]] const auto shape = InferReshape(input, multiple_inferred), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto shape = InferReshape(input, negative), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto shape = InferReshape(input, wrong_count), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto shape = InferReshape(input, nonintegral), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto shape = InferReshape(empty, empty_inferred), InvalidArgumentError);
}

TEST_F(TensorTest, AppliesMetadataOnlyDimensionTransforms) {
  auto storage = MakeStorage(1024);
  const auto input = TensorFactory::Create(storage, DType::FLOAT32, Shape{2, 1, 3, 4}, Strides{12, 12, 4, 1}, 0);

  constexpr std::array<int64_t, 4> axes{-1, 0, 2, 1};
  const auto permuted = Permute(input, axes);
  EXPECT_EQ(permuted.GetShape(), Shape({4, 2, 3, 1}));
  EXPECT_EQ(permuted.GetStrides(), Strides({1, 12, 4, 12}));

  const auto transposed = Transpose(input, 0, -1);
  EXPECT_EQ(transposed.GetShape(), Shape({4, 1, 3, 2}));
  EXPECT_EQ(transposed.GetStrides(), Strides({1, 12, 4, 12}));

  const auto squeezed = Squeeze(input);
  EXPECT_EQ(squeezed.GetShape(), Shape({2, 3, 4}));
  EXPECT_EQ(squeezed.GetStrides(), Strides({12, 4, 1}));
  EXPECT_EQ(Squeeze(input, 1).GetShape(), squeezed.GetShape());
  EXPECT_THROW([[maybe_unused]] const auto invalid = Squeeze(input, 0), InvalidArgumentError);

  const auto inserted = Unsqueeze(squeezed, 1);
  EXPECT_EQ(inserted.GetShape(), input.GetShape());
  EXPECT_EQ(inserted.GetStrides(), input.GetStrides());
  const auto appended = Unsqueeze(squeezed, -1);
  EXPECT_EQ(appended.GetShape(), Shape({2, 3, 4, 1}));
  EXPECT_EQ(appended.GetStrides(), Strides({12, 4, 1, 1}));

  constexpr std::array<int64_t, 4> identity_axes{0, 1, 2, 3};
  EXPECT_EQ(ClassifyAlias(input, Permute(input, identity_axes)), AliasKind::EXACT);
  constexpr std::array<int64_t, 4> duplicate_axes{0, 1, 1, 3};
  EXPECT_THROW([[maybe_unused]] const auto invalid = Permute(input, duplicate_axes), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto invalid = Unsqueeze(input, 5), InvalidArgumentError);
}

TEST_F(TensorTest, AppliesRangesSelectionsAndBroadcastViews) {
  auto storage = MakeStorage(1024);
  const auto input = TensorFactory::Create(storage, DType::FLOAT32, Shape{4, 5}, Strides{5, 1}, 0);

  const auto narrowed = Narrow(input, 0, -3, 2);
  EXPECT_EQ(narrowed.GetShape(), Shape({2, 5}));
  EXPECT_EQ(narrowed.GetStrides(), input.GetStrides());
  EXPECT_EQ(narrowed.GetStorageOffset(), 5);

  const auto sliced = Slice(input, 1, 1, std::nullopt, 2);
  EXPECT_EQ(sliced.GetShape(), Shape({4, 2}));
  EXPECT_EQ(sliced.GetStrides(), Strides({5, 2}));
  EXPECT_EQ(sliced.GetStorageOffset(), 1);
  EXPECT_FALSE(sliced.IsNonOverlappingDense());

  const auto empty_slice = Slice(input, 1, 5, 2);
  EXPECT_EQ(empty_slice.GetShape(), Shape({4, 0}));
  EXPECT_EQ(empty_slice.GetStorageOffset(), 5);

  const auto selected = Select(input, 0, -1);
  EXPECT_EQ(selected.GetShape(), Shape({5}));
  EXPECT_EQ(selected.GetStrides(), Strides({1}));
  EXPECT_EQ(selected.GetStorageOffset(), 15);

  const auto row = TensorFactory::Create(storage, DType::FLOAT32, Shape{1, 3}, Strides{3, 1}, 0);
  const auto expanded = Expand(row, Shape{2, 4, 3});
  EXPECT_EQ(expanded.GetStrides(), Strides({0, 0, 1}));
  EXPECT_TRUE(expanded.HasZeroStride());
  EXPECT_FALSE(expanded.IsNonOverlappingDense());

  const std::array shapes{Shape{2, 1, 0}, Shape{1, 3, 1}};
  EXPECT_EQ(BroadcastShapes(shapes), Shape({2, 3, 0}));
  EXPECT_EQ(BroadcastShapes(std::span<const Shape>{}), Shape{});

  EXPECT_THROW([[maybe_unused]] const auto invalid = Narrow(input, 0, 3, 2), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto invalid = Slice(input, 0, std::nullopt, std::nullopt, 0),
               InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto invalid = Select(input, 1, 5), InvalidArgumentError);
  EXPECT_THROW([[maybe_unused]] const auto invalid = Expand(input, Shape{4}), InvalidArgumentError);
  const std::array incompatible_shapes{Shape{2}, Shape{3}};
  EXPECT_THROW([[maybe_unused]] const auto invalid = BroadcastShapes(incompatible_shapes), InvalidArgumentError);
}

TEST_F(TensorTest, BuildsBroadcastAndScalarElementwisePlans) {
  auto output = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{2, 3}, Strides{3, 1}, 0);
  const auto lhs = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{2, 3}, Strides{3, 1}, 0);
  const auto rhs = TensorFactory::Create(MakeStorage(32), DType::FLOAT32, Shape{3}, Strides{1}, 0);

  const auto broadcast = ElementwiseIterator::Builder{}
                             .AddOutput(output)
                             .AddInput(lhs)
                             .AddInput(rhs)
                             .SetRequireSameDType(true)
                             .SetAliasPolicy(AliasPolicy::EXACT_ONE_BINARY_INPUT)
                             .Build("AddOut");
  EXPECT_EQ(broadcast.GetShape(), Shape({2, 3}));
  EXPECT_EQ(broadcast.GetNumElements(), 6);
  EXPECT_EQ(broadcast.GetRank(), 2);
  EXPECT_EQ(broadcast.GetOperandCount(), 3);
  EXPECT_EQ(broadcast.GetIndexWidth(), IndexWidth::UINT32);
  EXPECT_EQ(broadcast.GetPath(), IteratorPath::SINGLE_INNER_STRIDE);
  EXPECT_EQ(broadcast.GetVectorWidthElements(), 1);

  const auto parameters = broadcast.MakeParameters32();
  EXPECT_EQ(parameters.shape_[0], 2);
  EXPECT_EQ(parameters.shape_[1], 3);
  EXPECT_EQ(parameters.strides_bytes_[0][0], 12);
  EXPECT_EQ(parameters.strides_bytes_[0][1], 4);
  EXPECT_EQ(parameters.strides_bytes_[2][0], 0);
  EXPECT_EQ(parameters.strides_bytes_[2][1], 4);

  auto vector_output = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{8}, Strides{1}, 0);
  const auto vector_input = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{8}, Strides{1}, 0);
  const auto scalar = TensorFactory::Create(MakeStorage(sizeof(float)), DType::FLOAT32, Shape{}, Strides{}, 0);
  const auto scalar_plan = ElementwiseIterator::Builder{}
                               .AddOutput(vector_output)
                               .AddInput(vector_input)
                               .AddInput(scalar)
                               .SetRequireSameDType(true)
                               .SetAliasPolicy(AliasPolicy::EXACT_ONE_BINARY_INPUT)
                               .Build("AddScalarOut");
  EXPECT_EQ(scalar_plan.GetRank(), 1);
  EXPECT_EQ(scalar_plan.GetPath(), IteratorPath::CONTIGUOUS_WITH_SCALAR_INPUTS);
  EXPECT_EQ(scalar_plan.GetVectorWidthElements(), 4);
  EXPECT_EQ(scalar_plan.MakeParameters32().strides_bytes_[2][0], 0);
}

TEST_F(TensorTest, ClassifiesDenseGenericAndWideIndexIteratorPaths) {
  auto transposed_output = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{3, 2}, Strides{1, 3}, 0);
  const auto transposed_input = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{3, 2}, Strides{1, 3}, 0);
  const auto dense = ElementwiseIterator::Builder{}
                         .AddOutput(transposed_output)
                         .AddInput(transposed_input)
                         .SetAliasPolicy(AliasPolicy::EXACT_UNARY)
                         .Build("UnaryOut");
  EXPECT_EQ(dense.GetRank(), 1);
  EXPECT_EQ(dense.GetPath(), IteratorPath::CONTIGUOUS);
  EXPECT_EQ(dense.MakeParameters32().shape_[0], 6);

  auto generic_output = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{2, 2}, Strides{2, 1}, 0);
  const auto generic_input = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{2, 2}, Strides{3, 2}, 0);
  const auto generic = ElementwiseIterator::Builder{}
                           .AddOutput(generic_output)
                           .AddInput(generic_input)
                           .SetAliasPolicy(AliasPolicy::EXACT_UNARY)
                           .Build("UnaryOut");
  EXPECT_EQ(generic.GetRank(), 2);
  EXPECT_EQ(generic.GetPath(), IteratorPath::GENERIC_STRIDED);

  constexpr int64_t huge_stride = int64_t{1} << 32;
  auto *external_pointer = fake_allocations.back().bytes_.data();
  auto wide_storage = WrapStorage(external_pointer, size_t{1} << 35);
  const auto wide_input = TensorFactory::Create(wide_storage, DType::FLOAT32, Shape{2, 1}, Strides{huge_stride, 1}, 0);
  auto wide_output = TensorFactory::Create(MakeStorage(32), DType::FLOAT32, Shape{2, 1}, Strides{1, 1}, 0);
  const auto wide = ElementwiseIterator::Builder{}
                        .AddOutput(wide_output)
                        .AddInput(wide_input)
                        .SetAliasPolicy(AliasPolicy::EXACT_UNARY)
                        .Build("UnaryOut");
  EXPECT_EQ(wide.GetIndexWidth(), IndexWidth::UINT64);
  EXPECT_THROW([[maybe_unused]] const auto invalid = wide.MakeParameters32(), InternalError);
  EXPECT_EQ(wide.MakeParameters64().strides_bytes_[1][0], static_cast<uint64_t>(huge_stride) * sizeof(float));
}

TEST_F(TensorTest, EnforcesElementwiseOutputDTypeShapeAndAliasContracts) {
  auto storage = MakeStorage(128);
  auto output = TensorFactory::Create(storage, DType::FLOAT32, Shape{4}, Strides{1}, 0);
  const auto exact_input = output;

  EXPECT_NO_THROW([[maybe_unused]] const auto iterator = ElementwiseIterator::Builder{}
                                                             .AddOutput(output)
                                                             .AddInput(exact_input)
                                                             .SetAliasPolicy(AliasPolicy::EXACT_UNARY)
                                                             .Build("UnaryOut"));
  EXPECT_THROW([[maybe_unused]] const auto iterator = ElementwiseIterator::Builder{}
                                                          .AddOutput(output)
                                                          .AddInput(exact_input)
                                                          .SetAliasPolicy(AliasPolicy::NO_ALIAS)
                                                          .Build("UnaryOut"),
               InvalidArgumentError);

  auto partial_output = TensorFactory::Create(storage, DType::FLOAT32, Shape{4}, Strides{1}, 0);
  const auto partial_input = TensorFactory::Create(storage, DType::FLOAT32, Shape{4}, Strides{1}, 1);
  EXPECT_THROW([[maybe_unused]] const auto iterator = ElementwiseIterator::Builder{}
                                                          .AddOutput(partial_output)
                                                          .AddInput(partial_input)
                                                          .SetAliasPolicy(AliasPolicy::COPY)
                                                          .Build("CopyOut"),
               InvalidArgumentError);

  const auto broadcast_source =
      TensorFactory::Create(MakeStorage(sizeof(float)), DType::FLOAT32, Shape{1}, Strides{1}, 0);
  auto broadcast_output = Expand(broadcast_source, Shape{4});
  const auto separate_input = TensorFactory::Create(MakeStorage(32), DType::FLOAT32, Shape{4}, Strides{1}, 0);
  EXPECT_THROW([[maybe_unused]] const auto iterator = ElementwiseIterator::Builder{}
                                                          .AddOutput(broadcast_output)
                                                          .AddInput(separate_input)
                                                          .SetAliasPolicy(AliasPolicy::EXACT_UNARY)
                                                          .Build("UnaryOut"),
               InvalidArgumentError);

  auto dtype_output = TensorFactory::Create(MakeStorage(32), DType::FLOAT32, Shape{4}, Strides{1}, 0);
  const auto integer_input = TensorFactory::Create(MakeStorage(32), DType::INT32, Shape{4}, Strides{1}, 0);
  EXPECT_THROW([[maybe_unused]] const auto iterator = ElementwiseIterator::Builder{}
                                                          .AddOutput(dtype_output)
                                                          .AddInput(integer_input)
                                                          .SetRequireSameDType(true)
                                                          .SetAliasPolicy(AliasPolicy::EXACT_UNARY)
                                                          .Build("UnaryOut"),
               InvalidArgumentError);

  auto shape_output = TensorFactory::Create(MakeStorage(32), DType::FLOAT32, Shape{2}, Strides{1}, 0);
  const auto shape_input = TensorFactory::Create(MakeStorage(32), DType::FLOAT32, Shape{3}, Strides{1}, 0);
  EXPECT_THROW([[maybe_unused]] const auto iterator = ElementwiseIterator::Builder{}
                                                          .AddOutput(shape_output)
                                                          .AddInput(shape_input)
                                                          .SetAliasPolicy(AliasPolicy::EXACT_UNARY)
                                                          .Build("UnaryOut"),
               InvalidArgumentError);

  EXPECT_THROW([[maybe_unused]] const auto iterator = ElementwiseIterator::Builder{}.Build("MissingOutput"),
               InvalidArgumentError);
  auto builder = ElementwiseIterator::Builder{}.AddOutput(shape_output);
  builder.AddInput(shape_input).AddInput(shape_input).AddInput(shape_input);
  EXPECT_THROW(builder.AddInput(shape_input), InvalidArgumentError);
}

TEST_F(TensorTest, SelectsScalarVectorWidthForMisalignedViewsAndHandlesEmptyPlans) {
  auto output = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{4}, Strides{1}, 1);
  const auto input = TensorFactory::Create(MakeStorage(64), DType::FLOAT32, Shape{4}, Strides{1}, 1);
  const auto misaligned = ElementwiseIterator::Builder{}
                              .AddOutput(output)
                              .AddInput(input)
                              .SetAliasPolicy(AliasPolicy::EXACT_UNARY)
                              .Build("UnaryOut");
  EXPECT_EQ(misaligned.GetPath(), IteratorPath::CONTIGUOUS);
  EXPECT_EQ(misaligned.GetVectorWidthElements(), 1);

  auto empty_output = TensorFactory::Create(MakeStorage(0), DType::FLOAT32, Shape{0, 3}, Strides{3, 1}, 0);
  const auto broadcast_input = TensorFactory::Create(MakeStorage(0), DType::FLOAT32, Shape{0, 3}, Strides{3, 1}, 0);
  const auto empty = ElementwiseIterator::Builder{}
                         .AddOutput(empty_output)
                         .AddInput(broadcast_input)
                         .SetAliasPolicy(AliasPolicy::EXACT_UNARY)
                         .Build("UnaryOut");
  EXPECT_EQ(empty.GetNumElements(), 0);
  EXPECT_EQ(empty.GetVectorWidthElements(), 1);
  EXPECT_EQ(empty.GetIndexWidth(), IndexWidth::UINT32);
  EXPECT_EQ(empty.MakeParameters32().num_elements_, 0);
}

}  // namespace
}  // namespace ttl::internal
