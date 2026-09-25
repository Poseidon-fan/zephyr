#pragma once

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

namespace zephyr::layer {

/** Compute SiLU(gate) * value; round the FP32 activation to the input dtype before multiplying. */
[[nodiscard]] auto SiluAndMul(ttl::ExecutionContext &context, const ttl::Tensor &gate, const ttl::Tensor &value)
    -> ttl::Tensor;

}  // namespace zephyr::layer
