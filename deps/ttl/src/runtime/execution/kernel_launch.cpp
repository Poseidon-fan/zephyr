#include "ttl/runtime/kernel_launch.hpp"

#include <algorithm>
#include <bit>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <driver_types.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/memory/scratch_arena.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

[[nodiscard]] auto ToInternalCapturePolicy(CudaCapturePolicy policy, std::source_location location)
    -> internal::CapturePolicy {
  switch (policy) {
    case CudaCapturePolicy::FORBIDDEN:
      return internal::CapturePolicy::FORBIDDEN;
    case CudaCapturePolicy::SAFE:
      return internal::CapturePolicy::SAFE;
  }
  throw InvalidArgumentError("invalid CUDA capture policy", location);
}

}  // namespace

class CudaKernelLaunch::Impl final {
 public:
  Impl(ExecutionContext &context, std::string_view operation, std::span<const Tensor> inputs,
       std::span<Tensor *const> outputs, const CudaKernelLaunchOptions &options, std::source_location location)
      : operation_(operation),
        guard_(context, operation_, location, ToInternalCapturePolicy(options.capture_policy_, location)) {
    input_impls_.reserve(inputs.size());
    for (const auto &input : inputs) {
      guard_.RecordTensor(input);
      input_impls_.push_back(&internal::TensorAccess::GetImpl(input, location));
    }

    output_impls_.reserve(outputs.size());
    for (auto *output : outputs) {
      if (output == nullptr) {
        throw InvalidArgumentError("CUDA kernel output tensor must not be null", location);
      }
      guard_.RecordTensor(*output);
      output_impls_.push_back(&internal::TensorAccess::GetImpl(*output, location));
    }

    if (!std::has_single_bit(options.workspace_alignment_) || options.workspace_alignment_ > 256) {
      throw InvalidArgumentError("CUDA kernel workspace alignment must be a power of two no greater than 256",
                                 location);
    }
    if (options.workspace_bytes_ != 0) {
      scratch_scope_.emplace(guard_.MakeScratchScope());
      const auto allocation =
          scratch_scope_->AllocateBytes(options.workspace_bytes_, options.workspace_alignment_, location);
      workspace_ = {.data_ = allocation.GetData(), .size_bytes_ = allocation.GetSizeBytes()};
    }
  }

  [[nodiscard]] auto HasInput(const Tensor &tensor, std::source_location location) const -> bool {
    const auto *impl = &internal::TensorAccess::GetImpl(tensor, location);
    return std::ranges::find(input_impls_, impl) != input_impls_.end();
  }

  [[nodiscard]] auto HasOutput(const Tensor &tensor, std::source_location location) const -> bool {
    const auto *impl = &internal::TensorAccess::GetImpl(tensor, location);
    return std::ranges::find(output_impls_, impl) != output_impls_.end();
  }

  std::string operation_;
  internal::OpGuard guard_;
  std::vector<const internal::TensorImpl *> input_impls_;
  std::vector<const internal::TensorImpl *> output_impls_;
  std::optional<internal::ScratchArena::Scope> scratch_scope_;
  CudaWorkspace workspace_{.data_ = nullptr, .size_bytes_ = 0};
};

CudaKernelLaunch::CudaKernelLaunch(ExecutionContext &context, std::string_view operation,
                                   std::span<const Tensor> inputs, std::span<Tensor *const> outputs,
                                   const CudaKernelLaunchOptions &options, std::source_location location)
    : impl_(std::make_unique<Impl>(context, operation, inputs, outputs, options, location)) {}

CudaKernelLaunch::~CudaKernelLaunch() noexcept = default;

auto CudaKernelLaunch::GetStream() const noexcept -> cudaStream_t { return impl_->guard_.GetNativeStream(); }

auto CudaKernelLaunch::GetWorkspace() const noexcept -> CudaWorkspace { return impl_->workspace_; }

auto CudaKernelLaunch::IsCapturing() const noexcept -> bool { return impl_->guard_.IsCapturing(); }

auto CudaKernelLaunch::GetInputData(const Tensor &tensor, std::source_location location) const -> const void * {
  if (!impl_->HasInput(tensor, location) && !impl_->HasOutput(tensor, location)) {
    throw InvalidArgumentError("tensor was not registered with this CUDA kernel submission", location);
  }
  return internal::TensorAccess::GetData(tensor, location);
}

auto CudaKernelLaunch::GetOutputData(Tensor &tensor, std::source_location location) const -> void * {
  if (!impl_->HasOutput(tensor, location)) {
    throw InvalidArgumentError("tensor was not registered as an output of this CUDA kernel submission", location);
  }
  return internal::TensorAccess::GetMutableData(tensor, location);
}

auto CudaKernelLaunch::GetInputDataAsDType(const Tensor &tensor, DType dtype, std::source_location location) const
    -> const void * {
  if (tensor.GetDType() != dtype) {
    throw InvalidArgumentError("CUDA kernel input pointer dtype does not match the tensor dtype", location);
  }
  return GetInputData(tensor, location);
}

auto CudaKernelLaunch::GetOutputDataAsDType(Tensor &tensor, DType dtype, std::source_location location) const
    -> void * {
  if (tensor.GetDType() != dtype) {
    throw InvalidArgumentError("CUDA kernel output pointer dtype does not match the tensor dtype", location);
  }
  return GetOutputData(tensor, location);
}

void CudaKernelLaunch::CheckLaunch() const { impl_->guard_.CheckLaunch(); }

void CudaKernelLaunch::FailAfterCallbackException() noexcept { impl_->guard_.FailExternalSubmissionNoexcept(); }

}  // namespace ttl
