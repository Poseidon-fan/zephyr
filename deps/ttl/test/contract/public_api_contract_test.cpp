#include <concepts>
#include <type_traits>

#include <gtest/gtest.h>

#include "ttl/common/device.hpp"
#include "ttl/distributed/communicator.hpp"
#include "ttl/runtime/event.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/generator.hpp"
#include "ttl/runtime/graph.hpp"
#include "ttl/runtime/kernel_launch.hpp"
#include "ttl/runtime/pinned_buffer.hpp"
#include "ttl/runtime/runtime.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::test {
namespace {

static_assert(!std::default_initializable<Device>);
static_assert(std::is_trivially_copyable_v<Device>);
static_assert(std::is_standard_layout_v<Device>);
static_assert(sizeof(Device) == sizeof(int32_t));

static_assert(std::is_trivially_copyable_v<Shape>);
static_assert(std::is_trivially_copyable_v<Strides>);
static_assert(std::is_standard_layout_v<Shape>);
static_assert(std::is_standard_layout_v<Strides>);

static_assert(!std::copy_constructible<Runtime>);
static_assert(!std::move_constructible<Runtime>);
static_assert(!std::copy_constructible<ExecutionContext>);
static_assert(std::is_nothrow_move_constructible_v<ExecutionContext>);
static_assert(std::is_nothrow_move_assignable_v<ExecutionContext>);

static_assert(std::copy_constructible<Stream>);
static_assert(std::is_nothrow_move_constructible_v<Stream>);
static_assert(std::copy_constructible<Event>);
static_assert(std::is_nothrow_move_constructible_v<Event>);
static_assert(std::copy_constructible<Tensor>);
static_assert(std::is_nothrow_move_constructible_v<Tensor>);
static_assert(std::copy_constructible<PinnedBuffer>);
static_assert(std::is_nothrow_move_constructible_v<PinnedBuffer>);

static_assert(!std::copy_constructible<Generator>);
static_assert(std::is_nothrow_move_constructible_v<Generator>);
static_assert(!std::copy_constructible<CaptureSession>);
static_assert(std::is_nothrow_move_constructible_v<CaptureSession>);
static_assert(!std::copy_constructible<CapturedGraph>);
static_assert(std::is_nothrow_move_constructible_v<CapturedGraph>);
static_assert(!std::copy_constructible<CapturedGraphGroup>);
static_assert(std::is_nothrow_move_constructible_v<CapturedGraphGroup>);
static_assert(!std::copy_constructible<NcclCommunicator>);
static_assert(std::is_nothrow_move_constructible_v<NcclCommunicator>);
static_assert(!std::copy_constructible<LocalCommunicatorGroup>);
static_assert(std::is_nothrow_move_constructible_v<LocalCommunicatorGroup>);
static_assert(!std::copy_constructible<CudaKernelLaunch>);
static_assert(!std::move_constructible<CudaKernelLaunch>);

TEST(PublicApiContractTest, SupportedStorageTypesHaveBidirectionalMappings) {
  static_assert(DTYPE_OF<bool> == DType::BOOL);
  static_assert(DTYPE_OF<uint8_t> == DType::UINT8);
  static_assert(DTYPE_OF<int32_t> == DType::INT32);
  static_assert(DTYPE_OF<int64_t> == DType::INT64);
  static_assert(DTYPE_OF<Float16> == DType::FLOAT16);
  static_assert(DTYPE_OF<BFloat16> == DType::BFLOAT16);
  static_assert(DTYPE_OF<float> == DType::FLOAT32);
  static_assert(std::same_as<StorageTypeForT<DType::BOOL>, bool>);
  static_assert(std::same_as<StorageTypeForT<DType::UINT8>, uint8_t>);
  static_assert(std::same_as<StorageTypeForT<DType::INT32>, int32_t>);
  static_assert(std::same_as<StorageTypeForT<DType::INT64>, int64_t>);
  static_assert(std::same_as<StorageTypeForT<DType::FLOAT16>, Float16>);
  static_assert(std::same_as<StorageTypeForT<DType::BFLOAT16>, BFloat16>);
  static_assert(std::same_as<StorageTypeForT<DType::FLOAT32>, float>);
  SUCCEED();
}

}  // namespace
}  // namespace ttl::test
