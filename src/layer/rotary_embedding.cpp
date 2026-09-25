#include "layer/rotary_embedding.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include <ttl/ops/cast.hpp>
#include <ttl/ops/copy.hpp>
#include <ttl/ops/creation.hpp>
#include <ttl/ops/elementwise.hpp>
#include <ttl/runtime/kernel_launch.hpp>
#include <ttl/tensor/layout.hpp>
#include <ttl/tensor/scalar.hpp>

#include "common/exception.hpp"
#include "layer/rotary_embedding.cuh"

namespace zephyr::layer {

RotaryEmbedding::RotaryEmbedding(ttl::ExecutionContext &context, float base, int64_t head_dim,
                                 int64_t max_position_embeddings, bool is_gpt_neox, ttl::DType dtype)
    : RotaryEmbedding(Create(context, base, head_dim, max_position_embeddings, is_gpt_neox, dtype)) {}

auto RotaryEmbedding::Create(ttl::ExecutionContext &context, float base, int64_t head_dim,
                             int64_t max_position_embeddings, bool is_gpt_neox, ttl::DType dtype) -> RotaryEmbedding {
  if (head_dim <= 0 || head_dim % 2 != 0 || !std::isfinite(base) || base <= 0.0F) {
    throw InvalidArgumentException("rotary head dimension must be positive and even, and base finite and positive");
  }
  if (max_position_embeddings <= 0 ||
      (dtype != ttl::DType::FLOAT32 && dtype != ttl::DType::FLOAT16 && dtype != ttl::DType::BFLOAT16)) {
    throw InvalidArgumentException("rotary cache length must be positive and dtype must be floating");
  }
  std::vector<float> frequencies(static_cast<size_t>(head_dim / 2));
  for (int64_t index = 0; index < head_dim / 2; ++index) {
    frequencies[static_cast<size_t>(index)] =
        1.0F / std::pow(base, static_cast<float>(2 * index) / static_cast<float>(head_dim));
  }
  auto inverse_frequencies = ttl::Empty(context, ttl::Shape{1, head_dim / 2}, ttl::DType::FLOAT32);
  ttl::CopyFromHostBlocking(context, inverse_frequencies, std::as_bytes(std::span{frequencies}));
  const auto positions = ttl::Cast(context,
                                   ttl::Arange(context, ttl::Scalar{int64_t{0}}, ttl::Scalar{max_position_embeddings},
                                               ttl::Scalar{int64_t{1}}, ttl::DType::INT64),
                                   ttl::DType::FLOAT32);
  const auto angles =
      ttl::Multiply(context, ttl::View(positions, ttl::Shape{max_position_embeddings, 1}), inverse_frequencies);
  auto cos = ttl::Cos(context, angles);
  auto sin = ttl::Sin(context, angles);
  if (dtype != ttl::DType::FLOAT32) {
    cos = ttl::Cast(context, cos, dtype);
    sin = ttl::Cast(context, sin, dtype);
  }
  return RotaryEmbedding{std::move(cos), std::move(sin), is_gpt_neox};
}

auto RotaryEmbedding::Forward(ttl::ExecutionContext &context, const ttl::Tensor &query, const ttl::Tensor &key,
                              const ttl::Tensor &positions) const -> std::pair<ttl::Tensor, ttl::Tensor> {
  if (query.GetRank() != 4 || key.GetRank() != 4 || positions.GetRank() != 1) {
    throw InvalidArgumentException("rotary query/key must be [B,H,S,D] and positions must be rank one");
  }
  const auto &q_shape = query.GetShape();
  const auto &k_shape = key.GetShape();
  const auto batch = q_shape.GetDimension(0);
  const auto sequence = q_shape.GetDimension(2);
  const auto head_dim = q_shape.GetDimension(3);
  const auto rotary_half_dim = cos_.GetShape().GetDimension(1);
  if (batch != k_shape.GetDimension(0) || sequence != k_shape.GetDimension(2) || head_dim != k_shape.GetDimension(3) ||
      head_dim < 2 * rotary_half_dim || query.GetDType() != cos_.GetDType() || key.GetDType() != query.GetDType() ||
      (positions.GetDType() != ttl::DType::INT32 && positions.GetDType() != ttl::DType::INT64)) {
    throw InvalidArgumentException(
        "rotary Q/K shapes and dtype must agree with the cache, and positions must be integer");
  }
  // One CUDA block owns one token; reject an unrepresentable launch before flattening dimensions.
  if (sequence != 0 && batch > std::numeric_limits<int32_t>::max() / sequence) {
    throw InvalidArgumentException("rotary token count exceeds the CUDA grid range");
  }
  const auto tokens = batch * sequence;
  if (positions.GetShape().GetDimension(0) != tokens) {
    throw InvalidArgumentException("rotary positions length must equal batch times sequence");
  }

  // Only a contiguous token-major transpose shares input storage; all other layouts are materialized before rotation.
  auto q = ttl::View(ttl::Contiguous(context, ttl::Transpose(query, 1, 2)),
                     ttl::Shape{tokens, q_shape.GetDimension(1), head_dim});
  auto k = ttl::View(ttl::Contiguous(context, ttl::Transpose(key, 1, 2)),
                     ttl::Shape{tokens, k_shape.GetDimension(1), head_dim});
  const auto indices = ttl::Contiguous(context, positions);
  const std::array inputs{cos_, sin_, indices};
  const std::array<ttl::Tensor *, 2> outputs{&q, &k};
  ttl::SubmitCudaKernel(context, "RotaryEmbedding", inputs, outputs, [&](ttl::CudaKernelLaunch &launch) {
    const auto error = launch.GetDeviceErrorContext(indices.GetDType());
    // Separate launches preserve ordering even when the two input views share storage.
    for (auto *tensor : outputs) {
      LaunchRotaryEmbedding(launch.GetStream(), tensor->GetDType(), launch.GetOutputData(*tensor),
                            launch.GetInputData(cos_), launch.GetInputData(sin_), launch.GetInputData(indices),
                            indices.GetDType(), tokens, tensor->GetShape().GetDimension(1), head_dim, rotary_half_dim,
                            cos_.GetShape().GetDimension(0), is_gpt_neox_, error);
    }
  });
  auto q_output = ttl::Transpose(ttl::View(q, ttl::Shape{batch, sequence, q_shape.GetDimension(1), head_dim}), 1, 2);
  auto k_output = ttl::Transpose(ttl::View(k, ttl::Shape{batch, sequence, k_shape.GetDimension(1), head_dim}), 1, 2);
  return {ttl::Contiguous(context, q_output), ttl::Contiguous(context, k_output)};
}

}  // namespace zephyr::layer
