#include "layer/normalization.hpp"

#include <utility>

#include <ttl/ops/normalization.hpp>
#include <ttl/tensor/shape.hpp>

namespace zephyr::layer {

RmsNorm::RmsNorm(ttl::ExecutionContext &context, int64_t size, double epsilon, const weight::WeightBuilder &builder)
    : RmsNorm(builder.Get(context, ttl::Shape{size}, "weight"), epsilon) {}

RmsNorm::RmsNorm(ttl::Tensor weight, double epsilon) : weight_(std::move(weight)), epsilon_(epsilon) {}

auto RmsNorm::Forward(ttl::ExecutionContext &context, const ttl::Tensor &input) const -> ttl::Tensor {
  return ttl::RmsNorm(context, input, weight_,
                      ttl::NormOptions{.normalized_rank_ = 1, .epsilon_ = static_cast<float>(epsilon_)});
}

}  // namespace zephyr::layer
