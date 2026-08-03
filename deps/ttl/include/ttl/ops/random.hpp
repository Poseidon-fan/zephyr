#pragma once

#include <source_location>

#include "ttl/runtime/execution_context.hpp"
#include "ttl/runtime/generator.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {

struct UniformOptions final {
  double low_{0.0};
  double high_{1.0};
};

struct NormalOptions final {
  double mean_{0.0};
  double standard_deviation_{1.0};
};

void UniformOut(ExecutionContext &context, Tensor &output, Generator &generator, const UniformOptions &options = {},
                std::source_location location = std::source_location::current());
[[nodiscard]] auto Uniform(ExecutionContext &context, const Shape &shape, DType dtype, Generator &generator,
                           const UniformOptions &options = {},
                           std::source_location location = std::source_location::current()) -> Tensor;

void NormalOut(ExecutionContext &context, Tensor &output, Generator &generator, const NormalOptions &options = {},
               std::source_location location = std::source_location::current());
[[nodiscard]] auto Normal(ExecutionContext &context, const Shape &shape, DType dtype, Generator &generator,
                          const NormalOptions &options = {},
                          std::source_location location = std::source_location::current()) -> Tensor;

}  // namespace ttl
