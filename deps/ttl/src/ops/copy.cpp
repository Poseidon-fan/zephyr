#include "ttl/ops/copy.hpp"

#include <cstddef>
#include <cstring>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <driver_types.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/elementwise_launch.hpp"
#include "ttl/internal/runtime/cuda_api.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/execution/event.hpp"
#include "ttl/internal/runtime/execution/execution_context.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/execution/stream.hpp"
#include "ttl/internal/runtime/graph/graph.hpp"
#include "ttl/internal/runtime/memory/pinned/block.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/event.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/stream.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

class PeerTransferGuard final {
 public:
  PeerTransferGuard(ExecutionContext &destination_context, Tensor &destination, const Tensor &source,
                    const Event &source_ready, std::source_location location)
      : submission_(destination_context, "CopyPeerOut", location),
        destination_context_(destination_context),
        destination_(destination),
        source_(source),
        source_ready_(source_ready),
        location_(location),
        destination_device_(destination_context.GetDevice()),
        source_device_(source.GetDevice()) {
    submission_.ValidateTensor(destination_);
    if (source_device_ == destination_device_) {
      throw InvalidArgumentError("CopyPeerOut requires source and destination on different devices", location_);
    }
    if (source_ready_.GetDevice() != source_device_) {
      throw InvalidArgumentError("CopyPeerOut source event must be recorded on the source device", location_);
    }
    if (destination_.GetShape() != source_.GetShape()) {
      throw InvalidArgumentError("CopyPeerOut requires equal source and destination shapes", location_);
    }
    if (destination_.GetDType() != source_.GetDType()) {
      throw InvalidArgumentError("CopyPeerOut requires equal source and destination dtypes", location_);
    }
    if (!destination_.IsContiguous() || !source_.IsContiguous()) {
      throw InvalidArgumentError("CopyPeerOut requires contiguous source and destination tensors", location_);
    }
    if (!internal::ContextAccess::CanAccessPeer(destination_context_, source_device_, location_)) {
      throw NotSupportedError("destination device cannot directly access the source device", location_);
    }
  }

  void Submit() {
    if (destination_.GetNumElements() == 0) {
      return;
    }
    const auto &stream = submission_.GetStream();
    internal::EventAccess::Wait(stream, source_ready_, location_);
    submission_.RecordTensor(destination_, stream);
    submission_.RecordRemoteTensor(source_, stream);
    const auto bytes = internal::CheckedBytes(destination_.GetNumElements(),
                                              GetDTypeInfo(destination_.GetDType(), location_).size_bytes_, location_);
    internal::CheckCuda(internal::GetCudaApi().memcpy_peer_async_(
                            internal::TensorAccess::GetMutableData(destination_, location_),
                            destination_device_.GetOrdinal(), internal::TensorAccess::GetData(source_, location_),
                            source_device_.GetOrdinal(), bytes, submission_.GetNativeStream()),
                        "cudaMemcpyPeerAsync (CopyPeerOut)", location_);
  }

 private:
  internal::SubmissionScope submission_;
  ExecutionContext &destination_context_;
  Tensor &destination_;
  const Tensor &source_;
  const Event &source_ready_;
  std::source_location location_;
  Device destination_device_;
  Device source_device_;
};

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
    const auto bytes = internal::CheckedBytes(output.GetNumElements(),
                                              GetDTypeInfo(output.GetDType(), location).size_bytes_, location);
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
      internal::CheckedBytes(tensor.GetNumElements(), GetDTypeInfo(tensor.GetDType(), location).size_bytes_, location);
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
  internal::OpGuard guard{context, "CopyOut", location, internal::CapturePolicy::SAFE};
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
  internal::OpGuard guard{context, "ContiguousOut", location, internal::CapturePolicy::SAFE};
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
  PeerTransferGuard guard{destination_context, destination, source, source_ready, location};
  guard.Submit();
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
