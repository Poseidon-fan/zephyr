#include "ttl/runtime/kernel_launch.hpp"

#include <algorithm>
#include <bit>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <driver_types.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/execution/parallel_op_scope.hpp"
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

void ValidateWorkspaceRequest(const CudaWorkspaceRequest &request, std::string_view role,
                              std::source_location location) {
  if (!std::has_single_bit(request.alignment_) || request.alignment_ > 256) {
    throw InvalidArgumentError(std::string{"CUDA kernel "} + std::string{role} +
                                   " workspace alignment must be a power of two no greater than 256",
                               location);
  }
}

class LaunchWorkspace final {
 public:
  LaunchWorkspace() = default;
  LaunchWorkspace(const LaunchWorkspace &) = delete;
  auto operator=(const LaunchWorkspace &) -> LaunchWorkspace & = delete;
  LaunchWorkspace(LaunchWorkspace &&) noexcept = default;
  auto operator=(LaunchWorkspace &&) -> LaunchWorkspace & = delete;

  template <typename MakeScope>
    requires std::invocable<MakeScope>
  void Allocate(const CudaWorkspaceRequest &request, MakeScope &&make_scope, std::source_location location) {
    if (request.size_bytes_ == 0) {
      return;
    }
    scratch_scope_.emplace(std::invoke(std::forward<MakeScope>(make_scope)));
    const auto allocation = scratch_scope_->AllocateBytes(request.size_bytes_, request.alignment_, location);
    workspace_ = {.data_ = allocation.GetData(), .size_bytes_ = allocation.GetSizeBytes()};
  }

  [[nodiscard]] auto GetWorkspace() const noexcept -> CudaWorkspace { return workspace_; }

 private:
  std::optional<internal::ScratchArena::Scope> scratch_scope_;
  CudaWorkspace workspace_{.data_ = nullptr, .size_bytes_ = 0};
};

}  // namespace

class CudaKernelLaunch::Impl final {
 public:
  Impl(ExecutionContext &context, std::string_view operation, std::span<const Tensor> inputs,
       std::span<Tensor *const> outputs, const CudaKernelLaunchOptions &options, std::source_location location)
      : operation_(operation),
        guard_(context, operation_, location, ToInternalCapturePolicy(options.capture_policy_, location)) {
    ValidateWorkspaceRequest(options.workspace_, "primary", location);
    if (options.auxiliary_workspaces_.size() > options.auxiliary_stream_count_) {
      throw InvalidArgumentError("CUDA kernel auxiliary workspace requests exceed the requested stream count",
                                 location);
    }
    for (const auto &request : options.auxiliary_workspaces_) {
      ValidateWorkspaceRequest(request, "auxiliary", location);
    }
    if (options.auxiliary_stream_count_ > context.GetAuxiliaryStreamCount(location)) {
      throw InvalidArgumentError("CUDA kernel requested more auxiliary streams than the context owns", location);
    }

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

    primary_workspace_.Allocate(options.workspace_, [&] { return guard_.MakeScratchScope(); }, location);
    if (options.auxiliary_stream_count_ == 0) {
      return;
    }

    try {
      parallel_scope_.emplace(guard_, options.auxiliary_stream_count_, location);
      for (size_t index = 0; index < options.auxiliary_stream_count_; index++) {
        for (const auto &input : inputs) {
          parallel_scope_->RecordTensor(input, index);
        }
        for (auto *output : outputs) {
          parallel_scope_->RecordTensor(*output, index);
        }
      }

      auxiliary_workspaces_.reserve(options.auxiliary_stream_count_);
      for (size_t index = 0; index < options.auxiliary_stream_count_; index++) {
        auxiliary_workspaces_.emplace_back();
        if (index < options.auxiliary_workspaces_.size()) {
          auxiliary_workspaces_.back().Allocate(
              options.auxiliary_workspaces_[index], [&] { return parallel_scope_->MakeAuxiliaryScratchScope(index); },
              location);
        }
      }
    } catch (...) {
      FinishParallelScopeAfterConstructionFailure();
      throw;
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

  [[nodiscard]] auto GetAuxiliaryStream(size_t index, std::source_location location) const -> cudaStream_t {
    ValidateAuxiliaryIndex(index, location);
    return parallel_scope_->GetNativeAuxiliaryStream(index);
  }

  [[nodiscard]] auto GetAuxiliaryWorkspace(size_t index, std::source_location location) const -> CudaWorkspace {
    ValidateAuxiliaryIndex(index, location);
    return auxiliary_workspaces_[index].GetWorkspace();
  }

  void Finish() {
    auto launch_error = std::exception_ptr{};
    try {
      guard_.CheckLaunch();
    } catch (...) {
      launch_error = std::current_exception();
    }

    if (parallel_scope_.has_value()) {
      try {
        parallel_scope_->Finish();
      } catch (...) {
        if (launch_error != nullptr) {
          std::rethrow_exception(launch_error);
        }
        throw;
      }
    }
    if (launch_error != nullptr) {
      std::rethrow_exception(launch_error);
    }
  }

  void FailAfterCallbackException() noexcept {
    guard_.FailExternalSubmissionNoexcept();
    if (parallel_scope_.has_value()) {
      parallel_scope_->FailExternalSubmissionNoexcept();
    }
  }

  [[nodiscard]] auto GetAuxiliaryStreamCount() const noexcept -> size_t { return auxiliary_workspaces_.size(); }

  [[nodiscard]] auto GetPrimaryWorkspace() const noexcept -> CudaWorkspace { return primary_workspace_.GetWorkspace(); }

  [[nodiscard]] auto GetPrimaryStream() const noexcept -> cudaStream_t { return guard_.GetNativeStream(); }

  [[nodiscard]] auto IsCapturing() const noexcept -> bool { return guard_.IsCapturing(); }

 private:
  void ValidateAuxiliaryIndex(size_t index, std::source_location location) const {
    if (index >= auxiliary_workspaces_.size()) {
      throw InvalidArgumentError("CUDA kernel auxiliary stream index is out of range", location);
    }
  }

  void FinishParallelScopeAfterConstructionFailure() noexcept {
    if (!parallel_scope_.has_value()) {
      return;
    }
    try {
      parallel_scope_->Finish();
    } catch (...) {
      return;
    }
  }

  std::string operation_;
  internal::OpGuard guard_;
  std::vector<const internal::TensorImpl *> input_impls_;
  std::vector<const internal::TensorImpl *> output_impls_;
  LaunchWorkspace primary_workspace_;
  std::optional<internal::ParallelOpScope> parallel_scope_;
  std::vector<LaunchWorkspace> auxiliary_workspaces_;
};

CudaKernelLaunch::CudaKernelLaunch(ExecutionContext &context, std::string_view operation,
                                   std::span<const Tensor> inputs, std::span<Tensor *const> outputs,
                                   const CudaKernelLaunchOptions &options, std::source_location location)
    : impl_(std::make_unique<Impl>(context, operation, inputs, outputs, options, location)) {}

CudaKernelLaunch::~CudaKernelLaunch() noexcept = default;

auto CudaKernelLaunch::GetStream() const noexcept -> cudaStream_t { return impl_->GetPrimaryStream(); }

auto CudaKernelLaunch::GetWorkspace() const noexcept -> CudaWorkspace { return impl_->GetPrimaryWorkspace(); }

auto CudaKernelLaunch::GetAuxiliaryStreamCount() const noexcept -> size_t { return impl_->GetAuxiliaryStreamCount(); }

auto CudaKernelLaunch::GetAuxiliaryStream(size_t index, std::source_location location) const -> cudaStream_t {
  return impl_->GetAuxiliaryStream(index, location);
}

auto CudaKernelLaunch::GetAuxiliaryWorkspace(size_t index, std::source_location location) const -> CudaWorkspace {
  return impl_->GetAuxiliaryWorkspace(index, location);
}

auto CudaKernelLaunch::IsCapturing() const noexcept -> bool { return impl_->IsCapturing(); }

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

void CudaKernelLaunch::Finish() { impl_->Finish(); }

void CudaKernelLaunch::FailAfterCallbackException() noexcept { impl_->FailAfterCallbackException(); }

}  // namespace ttl
