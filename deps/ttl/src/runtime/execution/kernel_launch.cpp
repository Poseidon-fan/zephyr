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
#include "ttl/internal/distributed/communicator.hpp"
#include "ttl/internal/ops/random.hpp"
#include "ttl/internal/runtime/execution/generator.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/execution/parallel_op_scope.hpp"
#include "ttl/internal/runtime/memory/device/scratch_arena.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/generator.hpp"
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
    const internal::ScratchAllocation allocation =
        AllocateBytes(request.size_bytes_, request.alignment_, std::forward<MakeScope>(make_scope), location);
    workspace_ = {.data_ = allocation.GetData(), .size_bytes_ = allocation.GetSizeBytes()};
  }

  template <typename MakeScope>
    requires std::invocable<MakeScope>
  [[nodiscard]] auto AllocateBytes(size_t bytes, size_t alignment, MakeScope &&make_scope,
                                   std::source_location location) -> internal::ScratchAllocation {
    if (!scratch_scope_.has_value()) {
      scratch_scope_.emplace(std::invoke(std::forward<MakeScope>(make_scope)));
    }
    return scratch_scope_->AllocateBytes(bytes, alignment, location);
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
      : context_(context),
        operation_(operation),
        operation_location_(location),
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
    // Always join auxiliary lanes even when the primary launch check fails. If both operations fail, preserve the
    // earlier launch error because it is the root failure and the join error is a cleanup consequence.
    std::exception_ptr launch_error;
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
    // The callback exception remains the user-visible failure. Both scopes best-effort close their fork/join state and
    // report cleanup errors through ErrorSink without throwing across the original exception.
    guard_.FailExternalSubmissionNoexcept();
    if (parallel_scope_.has_value()) {
      parallel_scope_->FailExternalSubmissionNoexcept();
    }
  }

  [[nodiscard]] auto GetAuxiliaryStreamCount() const noexcept -> size_t { return auxiliary_workspaces_.size(); }

  [[nodiscard]] auto GetPrimaryWorkspace() const noexcept -> CudaWorkspace { return primary_workspace_.GetWorkspace(); }

  [[nodiscard]] auto GetPrimaryStream() const noexcept -> cudaStream_t { return guard_.GetNativeStream(); }

  [[nodiscard]] auto IsCapturing() const noexcept -> bool { return guard_.IsCapturing(); }

  void PublishPrimaryToAuxiliary() {
    if (!parallel_scope_.has_value()) {
      throw InvalidArgumentError("CUDA kernel launch has no auxiliary streams", operation_location_);
    }
    parallel_scope_->PublishPrimaryToAuxiliary();
  }

  void PublishAuxiliaryToPrimary() {
    if (!parallel_scope_.has_value()) {
      throw InvalidArgumentError("CUDA kernel launch has no auxiliary streams", operation_location_);
    }
    parallel_scope_->PublishAuxiliaryToPrimary();
  }

  [[nodiscard]] auto GetDeviceErrorContext(DType source_dtype, DType target_dtype) -> CudaDeviceErrorContext {
    if (!device_error_context_.has_value()) {
      device_error_context_.emplace(guard_.RegisterDeviceError(source_dtype, target_dtype));
    } else if (device_error_context_->source_dtype_ != source_dtype ||
               device_error_context_->target_dtype_ != target_dtype) {
      throw InvalidArgumentError("CUDA kernel device error context dtype metadata cannot change", operation_location_);
    }
    return *device_error_context_;
  }

  [[nodiscard]] auto ReservePhilox(Generator &generator, uint64_t block_count) -> CudaPhiloxReservation {
    if (block_count == 0) {
      throw InvalidArgumentError("Philox reservation requires at least one block", operation_location_);
    }
    if (generator_guard_ != nullptr) {
      throw InvalidArgumentError("a CUDA kernel launch may reserve from one Generator exactly once",
                                 operation_location_);
    }
    internal::GeneratorImpl &generator_impl = internal::GeneratorAccess::GetImpl(generator, operation_location_);
    if (context_.GetDevice() != generator_impl.device_ || context_.GetStream().GetId() != generator_impl.stream_id_) {
      throw InvalidArgumentError("Generator must be used with the execution context stream that created it",
                                 operation_location_);
    }
    generator_guard_ = std::make_unique<internal::GeneratorUseGuard>(generator_impl, operation_location_);
    const internal::ScratchAllocation base_counter_allocation = primary_workspace_.AllocateBytes(
        sizeof(uint64_t), alignof(uint64_t), [&] { return guard_.MakeScratchScope(); }, operation_location_);
    auto *base_counter = static_cast<uint64_t *>(base_counter_allocation.GetData());
    generator_impl.storage_->RecordUsage(context_.GetStream());
    guard_.RetainStorage(generator_impl.storage_);
    const auto error_context = guard_.RegisterDeviceError(DType::INT64, DType::INT64);
    internal::LaunchReservePhilox(guard_.GetNativeStream(),
                                  static_cast<internal::GeneratorState *>(generator_impl.storage_->GetBasePointer()),
                                  block_count, base_counter, error_context, operation_location_);
    if (parallel_scope_.has_value()) {
      parallel_scope_->PublishPrimaryToAuxiliary();
    }
    return CudaPhiloxReservation{
        .generator_state_ = static_cast<const CudaPhiloxGeneratorState *>(generator_impl.storage_->GetBasePointer()),
        .base_counter_ = base_counter,
    };
  }

  void RetainCommunicator(const std::shared_ptr<internal::CommunicatorGroupState> &communicator) {
    guard_.RetainCommunicator(communicator);
  }

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

  ExecutionContext &context_;
  std::string operation_;
  std::source_location operation_location_;
  internal::OpGuard guard_;
  std::vector<const internal::TensorImpl *> input_impls_;
  std::vector<const internal::TensorImpl *> output_impls_;
  LaunchWorkspace primary_workspace_;
  std::optional<internal::ParallelOpScope> parallel_scope_;
  std::vector<LaunchWorkspace> auxiliary_workspaces_;
  std::optional<CudaDeviceErrorContext> device_error_context_;
  std::unique_ptr<internal::GeneratorUseGuard> generator_guard_;
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

void CudaKernelLaunch::PublishPrimaryToAuxiliary() { impl_->PublishPrimaryToAuxiliary(); }

void CudaKernelLaunch::PublishAuxiliaryToPrimary() { impl_->PublishAuxiliaryToPrimary(); }

auto CudaKernelLaunch::GetDeviceErrorContext(DType source_dtype, DType target_dtype) -> CudaDeviceErrorContext {
  return impl_->GetDeviceErrorContext(source_dtype, target_dtype);
}

auto CudaKernelLaunch::ReservePhilox(Generator &generator, uint64_t block_count) -> CudaPhiloxReservation {
  return impl_->ReservePhilox(generator, block_count);
}

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

void CudaKernelLaunch::RetainCommunicator(const std::shared_ptr<internal::CommunicatorGroupState> &communicator) {
  impl_->RetainCommunicator(communicator);
}

}  // namespace ttl
