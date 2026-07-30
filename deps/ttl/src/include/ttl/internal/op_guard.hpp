#pragma once

#include <source_location>
#include <string_view>

#include "ttl/execution_context.hpp"
#include "ttl/internal/device_guard.hpp"
#include "ttl/internal/execution_context.hpp"
#include "ttl/tensor.hpp"

namespace ttl::internal {

class ParallelOpScope;

/** Common checked entry scope for CUDA operator wrappers. */
class OpGuard final {
 public:
  OpGuard(ExecutionContext &context, std::string_view operation,
          std::source_location location = std::source_location::current());

  OpGuard(const OpGuard &) = delete;
  auto operator=(const OpGuard &) -> OpGuard & = delete;
  OpGuard(OpGuard &&) = delete;
  auto operator=(OpGuard &&) -> OpGuard & = delete;

  void RecordTensor(const Tensor &tensor);
  void CheckLaunch() const;

  [[nodiscard]] auto GetStream() const noexcept -> const Stream &;
  [[nodiscard]] auto GetNativeStream() const noexcept -> cudaStream_t;

 private:
  friend class ParallelOpScope;

  void RecordTensorOnStream(const Tensor &tensor, const Stream &stream);

  ExecutionContext &context_;
  std::string_view operation_;
  std::source_location location_;
  ContextUseGuard use_guard_;
  DeviceGuard device_guard_;
  bool parallel_scope_active_{false};
};

}  // namespace ttl::internal
