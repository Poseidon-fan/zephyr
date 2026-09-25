#include "layer/activation.hpp"

#include <array>
#include <span>

#include <ttl/ops/creation.hpp>
#include <ttl/ops/elementwise.hpp>
#include <ttl/runtime/kernel_launch.hpp>

#include "common/exception.hpp"
#include "layer/activation.cuh"

namespace zephyr::layer {

auto SiluAndMul(ttl::ExecutionContext &context, const ttl::Tensor &gate, const ttl::Tensor &value) -> ttl::Tensor {
  if (!ttl::IsFloating(gate.GetDType()) || gate.GetShape() != value.GetShape() || gate.GetDType() != value.GetDType()) {
    throw InvalidArgumentException("gated activation inputs must have matching shapes and floating-point dtypes");
  }
  auto activated = ttl::Empty(context, gate.GetShape(), gate.GetDType());
  const std::array inputs{gate};
  const std::array outputs{&activated};
  ttl::SubmitCudaKernel(context, "Silu", std::span<const ttl::Tensor>{inputs}, std::span<ttl::Tensor *const>{outputs},
                        [&](ttl::CudaKernelLaunch &launch) {
                          LaunchSilu(launch.GetStream(), gate.GetDType(), launch.GetInputData(gate),
                                     launch.GetOutputData(activated), gate.GetShape(), gate.GetStrides());
                        });
  return ttl::Multiply(context, activated, value);
}

}  // namespace zephyr::layer
