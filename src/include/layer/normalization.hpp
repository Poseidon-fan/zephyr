#pragma once

#include <cstdint>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/tensor/tensor.hpp>

#include "weight/weight_builder.hpp"

namespace zephyr::layer {

/** Weighted RMS normalization over the final input dimension. */
class RmsNorm final {
 public:
  RmsNorm(ttl::ExecutionContext &context, int64_t size, double epsilon, const weight::WeightBuilder &builder);
  RmsNorm(ttl::Tensor weight, double epsilon);

  [[nodiscard]] auto Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor;

 private:
  ttl::Tensor weight_;
  double epsilon_;
};

}  // namespace zephyr::layer
