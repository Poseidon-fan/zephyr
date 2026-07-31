#include "ttl/ops/copy.hpp"

#include <cstddef>
#include <cstring>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <driver_types.h>

#include "ttl/dtype.hpp"
#include "ttl/error.hpp"
#include "ttl/event.hpp"
#include "ttl/execution_context.hpp"
#include "ttl/internal/checked_math.hpp"
#include "ttl/internal/cuda_api.hpp"
#include "ttl/internal/cuda_check.hpp"
#include "ttl/internal/elementwise_iterator.hpp"
#include "ttl/internal/elementwise_launch.hpp"
#include "ttl/internal/event.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/internal/op_guard.hpp"
#include "ttl/internal/pinned_allocator.hpp"
#include "ttl/internal/stream.hpp"
#include "ttl/internal/tensor_impl.hpp"
#include "ttl/stream.hpp"
#include "ttl/tensor.hpp"

namespace ttl {
namespace {

void CopyOutImpl(internal::OpGuard &guard, Tensor &output, const Tensor &input, bool require_contiguous_output,
                 std::source_location location) {
  guard.ValidateTensor(output);
  guard.ValidateTensor(input);
  if (output.GetShape() != input.GetShape()) {
    throw InvalidArgumentError("CopyOut requires equal input and output shapes", location);
  }
  if (output.GetDType() != input.GetDType()) {
    throw InvalidArgumentError("CopyOut requires equal input and output dtypes", location);
  }
  if (require_contiguous_output && !output.IsContiguous()) {
    throw InvalidArgumentError("ContiguousOut requires a canonical contiguous output", location);
  }

  auto iterator = internal::ElementwiseIterator::Builder{}
                      .AddOutput(output)
                      .AddInput(input)
                      .SetAliasPolicy(internal::AliasPolicy::COPY)
                      .SetRequireSameDType(true)
                      .Build(require_contiguous_output ? "ContiguousOut" : "CopyOut", location);
  if (ClassifyAlias(output, input, location) == AliasKind::EXACT || output.GetNumElements() == 0) {
    return;
  }

  guard.RecordTensor(output);
  guard.RecordTensor(input);
  if (output.IsContiguous() && input.IsContiguous()) {
    const auto bytes =
        internal::CheckedBytes(output.GetNumElements(), GetDTypeSize(output.GetDType(), location), location);
    internal::CheckCuda(internal::GetCudaApi().memcpy_async_(internal::TensorAccess::GetMutableData(output, location),
                                                             internal::TensorAccess::GetData(input, location), bytes,
                                                             cudaMemcpyDeviceToDevice, guard.GetNativeStream()),
                        "cudaMemcpyAsync (CopyOut)", location);
  } else {
    internal::LaunchCopy(guard.GetNativeStream(), output.GetDType(), iterator, location);
  }
  guard.CheckLaunch();
}

auto ValidateHostTransfer(internal::OpGuard &guard, const Tensor &tensor, size_t host_bytes, std::string_view operation,
                          std::source_location location) -> size_t {
  guard.ValidateTensor(tensor);
  if (!tensor.IsContiguous()) {
    std::string message{operation};
    message.append(" requires a contiguous tensor");
    throw InvalidArgumentError(std::move(message), location);
  }
  const auto expected_bytes =
      internal::CheckedBytes(tensor.GetNumElements(), GetDTypeSize(tensor.GetDType(), location), location);
  if (host_bytes != expected_bytes) {
    std::string message{operation};
    message.append(" requires host and tensor byte counts to match");
    throw InvalidArgumentError(std::move(message), location);
  }
  return expected_bytes;
}

void CopyFromPinnedImpl(internal::OpGuard &guard, Tensor &output, const PinnedBuffer &source,
                        std::source_location location) {
  auto &source_block = internal::PinnedBufferAccess::GetBlock(source, location);
  const auto bytes = ValidateHostTransfer(guard, output, source_block.GetSizeBytes(), "CopyFromPinnedAsync", location);
  if (bytes == 0) {
    return;
  }

  guard.RecordTensor(output);
  source_block.RecordUsage(guard.GetStream(), location);
  internal::CheckCuda(internal::GetCudaApi().memcpy_async_(internal::TensorAccess::GetMutableData(output, location),
                                                           source_block.GetData(), bytes, cudaMemcpyHostToDevice,
                                                           guard.GetNativeStream()),
                      "cudaMemcpyAsync (CopyFromPinnedAsync)", location);
}

void CopyToPinnedImpl(internal::OpGuard &guard, PinnedBuffer &output, const Tensor &source,
                      std::source_location location) {
  auto &output_block = internal::PinnedBufferAccess::GetBlock(output, location);
  const auto bytes = ValidateHostTransfer(guard, source, output_block.GetSizeBytes(), "CopyToPinnedAsync", location);
  if (bytes == 0) {
    return;
  }

  guard.RecordTensor(source);
  output_block.RecordUsage(guard.GetStream(), location);
  internal::CheckCuda(
      internal::GetCudaApi().memcpy_async_(output_block.GetData(), internal::TensorAccess::GetData(source, location),
                                           bytes, cudaMemcpyDeviceToHost, guard.GetNativeStream()),
      "cudaMemcpyAsync (CopyToPinnedAsync)", location);
}

}  // namespace

void CopyOut(ExecutionContext &context, Tensor &output, const Tensor &input, std::source_location location) {
  internal::OpGuard guard{context, "CopyOut", location};
  CopyOutImpl(guard, output, input, false, location);
}

auto Clone(ExecutionContext &context, const Tensor &input, std::source_location location) -> Tensor {
  const auto &input_impl = internal::TensorAccess::GetImpl(input, location);
  {
    internal::OpGuard guard{context, "Clone", location};
    guard.ValidateTensor(input);
  }
  auto output = Empty(context, input_impl.GetShape(), input_impl.GetDType(), location);
  CopyOut(context, output, input, location);
  return output;
}

void ContiguousOut(ExecutionContext &context, Tensor &output, const Tensor &input, std::source_location location) {
  internal::OpGuard guard{context, "ContiguousOut", location};
  CopyOutImpl(guard, output, input, true, location);
}

auto Contiguous(ExecutionContext &context, const Tensor &input, std::source_location location) -> Tensor {
  const auto &input_impl = internal::TensorAccess::GetImpl(input, location);
  if (input_impl.HasFlag(internal::TensorFlag::CONTIGUOUS)) {
    internal::OpGuard guard{context, "Contiguous", location};
    guard.ValidateTensor(input);
    return input;
  }

  {
    internal::OpGuard guard{context, "Contiguous", location};
    guard.ValidateTensor(input);
  }
  auto output = Empty(context, input_impl.GetShape(), input_impl.GetDType(), location);
  ContiguousOut(context, output, input, location);
  return output;
}

void CopyPeerOut(ExecutionContext &destination_context, Tensor &destination, const Tensor &source,
                 const Event &source_ready, std::source_location location) {
  internal::ContextUseGuard use_guard{destination_context, internal::ContextUseMode::SUBMIT, location};
  const auto destination_device = destination_context.GetDevice();
  const auto source_device = source.GetDevice();
  if (destination.GetDevice() != destination_device) {
    throw InvalidArgumentError("CopyPeerOut destination must be on the destination context device", location);
  }
  if (source_device == destination_device) {
    throw InvalidArgumentError("CopyPeerOut requires source and destination on different devices", location);
  }
  if (source_ready.GetDevice() != source_device) {
    throw InvalidArgumentError("CopyPeerOut source event must be recorded on the source device", location);
  }
  if (destination.GetShape() != source.GetShape()) {
    throw InvalidArgumentError("CopyPeerOut requires equal source and destination shapes", location);
  }
  if (destination.GetDType() != source.GetDType()) {
    throw InvalidArgumentError("CopyPeerOut requires equal source and destination dtypes", location);
  }
  if (!destination.IsContiguous() || !source.IsContiguous()) {
    throw InvalidArgumentError("CopyPeerOut requires contiguous source and destination tensors", location);
  }

  if (!internal::ContextAccess::CanAccessPeer(destination_context, source_device, location)) {
    throw NotSupportedError("destination device cannot directly access the source device", location);
  }
  if (destination.GetNumElements() == 0) {
    return;
  }

  const auto &destination_stream = destination_context.GetStream();
  internal::EventAccess::Wait(destination_stream, source_ready, location);
  internal::TensorAccess::RecordUsage(destination, destination_stream, location);
  internal::TensorAccess::RecordUsage(source, destination_stream, location);
  const auto bytes =
      internal::CheckedBytes(destination.GetNumElements(), GetDTypeSize(destination.GetDType(), location), location);
  internal::CheckCuda(internal::GetCudaApi().memcpy_peer_async_(
                          internal::TensorAccess::GetMutableData(destination, location),
                          destination_device.GetOrdinal(), internal::TensorAccess::GetData(source, location),
                          source_device.GetOrdinal(), bytes, internal::StreamAccess::GetNative(destination_stream)),
                      "cudaMemcpyPeerAsync (CopyPeerOut)", location);
}

void CopyFromPinnedAsync(ExecutionContext &context, Tensor &output, const PinnedBuffer &source,
                         std::source_location location) {
  internal::OpGuard guard{context, "CopyFromPinnedAsync", location};
  CopyFromPinnedImpl(guard, output, source, location);
}

void CopyToPinnedAsync(ExecutionContext &context, PinnedBuffer &output, const Tensor &source,
                       std::source_location location) {
  internal::OpGuard guard{context, "CopyToPinnedAsync", location};
  CopyToPinnedImpl(guard, output, source, location);
}

void CopyFromHostBlocking(ExecutionContext &context, Tensor &output, std::span<const std::byte> source,
                          std::source_location location) {
  {
    internal::OpGuard guard{context, "CopyFromHostBlocking", location};
    ValidateHostTransfer(guard, output, source.size(), "CopyFromHostBlocking", location);
  }

  {
    auto staging = internal::ContextAccess::AllocatePinned(context, source.size(), location);
    if (!source.empty()) {
      std::memcpy(staging.GetData(), source.data(), source.size());
    }
    internal::OpGuard guard{context, "CopyFromHostBlocking", location};
    CopyFromPinnedImpl(guard, output, staging, location);
  }
  context.Synchronize(location);
}

void CopyToHostBlocking(ExecutionContext &context, std::span<std::byte> output, const Tensor &source,
                        std::source_location location) {
  {
    internal::OpGuard guard{context, "CopyToHostBlocking", location};
    ValidateHostTransfer(guard, source, output.size(), "CopyToHostBlocking", location);
  }

  auto staging = internal::ContextAccess::AllocatePinned(context, output.size(), location);
  {
    internal::OpGuard guard{context, "CopyToHostBlocking", location};
    CopyToPinnedImpl(guard, staging, source, location);
  }
  context.Synchronize(location);
  if (!output.empty()) {
    std::memcpy(output.data(), staging.GetData(), output.size());
  }
}

}  // namespace ttl
