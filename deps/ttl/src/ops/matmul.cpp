#include "ttl/ops/matmul.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <library_types.h>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/ops/composition.hpp"
#include "ttl/internal/ops/elementwise_iterator.hpp"
#include "ttl/internal/ops/matmul_plan.hpp"
#include "ttl/internal/runtime/cuda_check.hpp"
#include "ttl/internal/runtime/execution/op_guard.hpp"
#include "ttl/internal/runtime/library/cublas_api.hpp"
#include "ttl/internal/runtime/memory/device/scratch_arena.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/ops/creation.hpp"
#include "ttl/ops/elementwise.hpp"
#include "ttl/runtime/execution_context.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/layout.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl {
namespace {

constexpr int MATMUL_HEURISTIC_RESULT_COUNT = 8;

enum class MatmulKind : uint8_t {
  MATMUL,
  BATCHED_MATMUL,
  LINEAR,
};

struct MatmulShapeInfo final {
  Shape output_shape_;
  Shape batch_shape_;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  int64_t batch_count_;
};

struct MatrixInfo final {
  const void *data_;
  uint64_t rows_;
  uint64_t columns_;
  int64_t leading_dimension_;
  int64_t batch_stride_;
  cublasLtOrder_t order_;
};

struct MutableMatrixInfo final {
  void *data_;
  uint64_t rows_;
  uint64_t columns_;
  int64_t leading_dimension_;
  int64_t batch_stride_;
  cublasLtOrder_t order_;
};

struct MatmulExecutionResult final {
  bool bias_fused_;
  bool activation_fused_;
};

enum class LinearBiasMode : uint8_t {
  NONE,
  FUSED_CONTIGUOUS,
  POST_ADD,
};

struct MatmulProblem final {
  MatmulKind kind_;
  MatmulShapeInfo shape_;
  LinearOptions linear_options_;
  LinearBiasMode bias_mode_;
};

[[nodiscard]] auto GetName(MatmulKind kind) noexcept -> std::string_view {
  switch (kind) {
    case MatmulKind::MATMUL:
      return "MatmulOut";
    case MatmulKind::BATCHED_MATMUL:
      return "BatchedMatmulOut";
    case MatmulKind::LINEAR:
      return "LinearOut";
  }
  return "MatmulOut";
}

void ValidateFloatingDType(DType dtype, std::string_view operation, std::source_location location) {
  if (!IsFloating(dtype)) {
    throw InvalidArgumentError(std::string{operation} + " supports only floating-point tensors", location);
  }
}

void ValidateLinearOptions(const LinearOptions &options, std::source_location location) {
  switch (options.activation_) {
    case LinearActivation::NONE:
    case LinearActivation::RELU:
      return;
    case LinearActivation::GELU:
      switch (options.gelu_approximation_) {
        case GeluApproximation::NONE:
        case GeluApproximation::TANH:
          return;
      }
      break;
  }
  throw InvalidArgumentError("invalid Linear activation options", location);
}

[[nodiscard]] auto GetBatchShape(const Tensor &tensor, std::source_location location) -> Shape {
  return Shape{tensor.GetShape().GetDimensions().first(tensor.GetRank() - 2), location};
}

[[nodiscard]] auto InferMatmulShape(const Tensor &lhs, const Tensor &rhs, MatmulKind kind,
                                    std::source_location location) -> MatmulShapeInfo {
  if (kind == MatmulKind::LINEAR) {
    if (lhs.GetRank() < 1 || rhs.GetRank() != 2) {
      throw InvalidArgumentError("Linear requires input rank >= 1 and weight rank 2", location);
    }
    const auto k = lhs.GetShape().GetDimension(lhs.GetRank() - 1, location);
    const auto n = rhs.GetShape().GetDimension(0, location);
    if (rhs.GetShape().GetDimension(1, location) != k) {
      throw InvalidArgumentError("Linear input and weight reduction dimensions must match", location);
    }
    auto dimensions = std::array<int64_t, TTL_MAX_RANK>{};
    std::ranges::copy(lhs.GetShape().GetDimensions(), dimensions.begin());
    dimensions[lhs.GetRank() - 1] = n;
    const auto output_shape = Shape{std::span<const int64_t>{dimensions.data(), lhs.GetRank()}, location};
    auto m = int64_t{1};
    for (size_t axis = 0; axis + 1 < lhs.GetRank(); ++axis) {
      m = internal::CheckedMultiply(m, lhs.GetShape().GetDimension(axis, location), "Linear row count", location);
    }
    return {
        .output_shape_ = output_shape,
        .batch_shape_ = Shape{},
        .m_ = m,
        .n_ = n,
        .k_ = k,
        .batch_count_ = 1,
    };
  }

  const auto expected_rank = kind == MatmulKind::MATMUL ? size_t{2} : size_t{3};
  if ((kind == MatmulKind::MATMUL && (lhs.GetRank() != expected_rank || rhs.GetRank() != expected_rank)) ||
      (kind == MatmulKind::BATCHED_MATMUL && (lhs.GetRank() < expected_rank || rhs.GetRank() < expected_rank))) {
    throw InvalidArgumentError(
        kind == MatmulKind::MATMUL ? "Matmul requires two rank-2 tensors" : "BatchedMatmul requires tensor ranks >= 3",
        location);
  }
  const auto m = lhs.GetShape().GetDimension(lhs.GetRank() - 2, location);
  const auto k = lhs.GetShape().GetDimension(lhs.GetRank() - 1, location);
  const auto rhs_k = rhs.GetShape().GetDimension(rhs.GetRank() - 2, location);
  const auto n = rhs.GetShape().GetDimension(rhs.GetRank() - 1, location);
  if (k != rhs_k) {
    throw InvalidArgumentError("matmul reduction dimensions must match", location);
  }

  auto batch_shape = Shape{};
  if (kind == MatmulKind::BATCHED_MATMUL) {
    const std::array batch_shapes{GetBatchShape(lhs, location), GetBatchShape(rhs, location)};
    batch_shape = BroadcastShapes(batch_shapes, location);
  }
  auto dimensions = std::array<int64_t, TTL_MAX_RANK>{};
  std::ranges::copy(batch_shape.GetDimensions(), dimensions.begin());
  dimensions[batch_shape.GetRank()] = m;
  dimensions[batch_shape.GetRank() + 1] = n;
  const auto output_shape =
      Shape{std::span<const int64_t>{dimensions.data(), batch_shape.GetRank() + size_t{2}}, location};
  return {
      .output_shape_ = output_shape,
      .batch_shape_ = batch_shape,
      .m_ = m,
      .n_ = n,
      .k_ = k,
      .batch_count_ = batch_shape.GetNumElements(),
  };
}

[[nodiscard]] auto BuildMatmulProblem(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs, MatmulKind kind,
                                      const std::optional<Tensor> &bias, const LinearOptions &linear_options,
                                      std::source_location location) -> MatmulProblem {
  const auto name = GetName(kind);
  if (kind == MatmulKind::LINEAR) {
    ValidateLinearOptions(linear_options, location);
  }
  internal::OpGuard guard{context, name, location, internal::CapturePolicy::SAFE};
  guard.ValidateTensor(lhs);
  guard.ValidateTensor(rhs);
  ValidateFloatingDType(lhs.GetDType(), name, location);
  if (rhs.GetDType() != lhs.GetDType()) {
    throw InvalidArgumentError(kind == MatmulKind::LINEAR ? "Linear input and weight must have the same dtype"
                                                          : "matmul operands must have the same dtype",
                               location);
  }
  auto shape = InferMatmulShape(lhs, rhs, kind, location);
  auto bias_mode = LinearBiasMode::NONE;
  if (bias.has_value()) {
    if (kind != MatmulKind::LINEAR) {
      throw InternalError("non-Linear matmul problem received a bias", location);
    }
    guard.ValidateTensor(*bias);
    if (bias->GetDType() != lhs.GetDType() || bias->GetShape() != Shape{shape.n_}) {
      throw InvalidArgumentError("Linear bias must have shape [out_features] and match the input dtype", location);
    }
    bias_mode = bias->IsContiguous() ? LinearBiasMode::FUSED_CONTIGUOUS : LinearBiasMode::POST_ADD;
  }
  return MatmulProblem{
      .kind_ = kind,
      .shape_ = shape,
      .linear_options_ = linear_options,
      .bias_mode_ = bias_mode,
  };
}

void ValidateMatmulOutput(internal::OpGuard &guard, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                          const std::optional<Tensor> &bias, const MatmulProblem &problem,
                          std::source_location location) {
  const auto name = GetName(problem.kind_);
  guard.ValidateTensor(output);
  if (rhs.GetDType() != lhs.GetDType() || output.GetDType() != lhs.GetDType()) {
    throw InvalidArgumentError("matmul operands and output must have the same dtype", location);
  }
  if (output.GetShape() != problem.shape_.output_shape_) {
    throw InvalidArgumentError("matmul output shape does not match inference", location);
  }
  internal::ValidateWritableOutput(output, name, location);
  const std::array<const Tensor *, 2> inputs{&lhs, &rhs};
  internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, output, inputs, name, location);
  if (bias.has_value()) {
    const std::array<const Tensor *, 1> bias_input{&*bias};
    internal::ValidateAlias(internal::AliasPolicy::NO_ALIAS, output, bias_input, name, location);
  }
}

[[nodiscard]] auto GetCudaDataType(DType dtype, std::source_location location) -> cudaDataType_t {
  switch (dtype) {
    case DType::FLOAT16:
      return CUDA_R_16F;
    case DType::BFLOAT16:
      return CUDA_R_16BF;
    case DType::FLOAT32:
      return CUDA_R_32F;
    case DType::BOOL:
    case DType::UINT8:
    case DType::INT32:
    case DType::INT64:
      break;
  }
  throw InternalError("matmul received a non-floating dtype after validation", location);
}

[[nodiscard]] auto GetComputeType(DType dtype, const MatmulOptions &options) noexcept -> cublasComputeType_t {
  return dtype == DType::FLOAT32 && options.allow_tf32_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

[[nodiscard]] auto GetPointerAlignment(const void *pointer) noexcept -> uint32_t {
  constexpr auto maximum_alignment = uintptr_t{256};
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  if (address == 0) {
    return 1;
  }
  const auto alignment = uintptr_t{1} << std::min<unsigned int>(std::countr_zero(address), 8);
  return static_cast<uint32_t>(std::min(alignment, maximum_alignment));
}

[[nodiscard]] auto GetContiguousByteStrides(const Shape &shape, size_t element_size, std::source_location location)
    -> std::array<uint64_t, TTL_MAX_RANK> {
  auto strides = std::array<uint64_t, TTL_MAX_RANK>{};
  auto stride = uint64_t{element_size};
  for (size_t remaining_rank = shape.GetRank(); remaining_rank > 0; --remaining_rank) {
    const auto axis = remaining_rank - 1;
    strides[axis] = stride;
    stride = internal::CheckedMultiply(stride, static_cast<uint64_t>(shape.GetDimension(axis, location)),
                                       "matmul contiguous byte stride", location);
  }
  return strides;
}

[[nodiscard]] auto MakeBatchedMatrixShape(const Shape &batch_shape, int64_t rows, int64_t columns,
                                          std::source_location location) -> Shape {
  auto dimensions = std::array<int64_t, TTL_MAX_RANK>{};
  std::ranges::copy(batch_shape.GetDimensions(), dimensions.begin());
  dimensions[batch_shape.GetRank()] = rows;
  dimensions[batch_shape.GetRank() + 1] = columns;
  return Shape{std::span<const int64_t>{dimensions.data(), batch_shape.GetRank() + size_t{2}}, location};
}

[[nodiscard]] auto BuildMaterializationParameters(void *output, const Tensor &input, const Shape &logical_shape,
                                                  std::source_location location) -> internal::CompositionParameters64 {
  const auto element_size = GetDTypeInfo(input.GetDType(), location).size_bytes_;
  const auto contiguous_strides = GetContiguousByteStrides(logical_shape, element_size, location);
  auto parameters = internal::CompositionParameters64{
      .output_ = static_cast<std::byte *>(output),
      .input_ = static_cast<const std::byte *>(internal::TensorAccess::GetData(input, location)),
      .num_elements_ = static_cast<uint64_t>(logical_shape.GetNumElements()),
      .rank_ = static_cast<uint8_t>(logical_shape.GetRank()),
  };
  const auto rank_offset = logical_shape.GetRank() - input.GetRank();
  for (size_t axis = 0; axis < logical_shape.GetRank(); ++axis) {
    parameters.shape_[axis] = static_cast<uint64_t>(logical_shape.GetDimension(axis, location));
    parameters.output_strides_bytes_[axis] = contiguous_strides[axis];
    if (axis < rank_offset) {
      parameters.input_strides_bytes_[axis] = 0;
      continue;
    }
    const auto input_axis = axis - rank_offset;
    const auto input_extent = input.GetShape().GetDimension(input_axis, location);
    parameters.input_strides_bytes_[axis] =
        input_extent == 1 && logical_shape.GetDimension(axis, location) != 1
            ? 0
            : internal::CheckedBytes(input.GetStrides().GetStride(input_axis, location), element_size, location);
  }
  return parameters;
}

[[nodiscard]] auto BuildCopyBackParameters(Tensor &output, const void *input, std::source_location location)
    -> internal::CompositionParameters64 {
  const auto element_size = GetDTypeInfo(output.GetDType(), location).size_bytes_;
  const auto contiguous_strides = GetContiguousByteStrides(output.GetShape(), element_size, location);
  auto parameters = internal::CompositionParameters64{
      .output_ = static_cast<std::byte *>(internal::TensorAccess::GetMutableData(output, location)),
      .input_ = static_cast<const std::byte *>(input),
      .num_elements_ = static_cast<uint64_t>(output.GetNumElements()),
      .rank_ = static_cast<uint8_t>(output.GetRank()),
  };
  for (size_t axis = 0; axis < output.GetRank(); ++axis) {
    parameters.shape_[axis] = static_cast<uint64_t>(output.GetShape().GetDimension(axis, location));
    parameters.output_strides_bytes_[axis] =
        internal::CheckedBytes(output.GetStrides().GetStride(axis, location), element_size, location);
    parameters.input_strides_bytes_[axis] = contiguous_strides[axis];
  }
  return parameters;
}

[[nodiscard]] auto GetUniformBatchStride(const Tensor &tensor, const Shape &batch_shape, std::source_location location)
    -> std::optional<int64_t> {
  if (batch_shape.GetNumElements() <= 1) {
    return int64_t{0};
  }
  const auto tensor_batch_rank = tensor.GetRank() - 2;
  const auto rank_offset = batch_shape.GetRank() - tensor_batch_rank;
  auto aligned_strides = std::array<int64_t, TTL_MAX_RANK>{};
  auto has_nonzero_stride = false;
  for (size_t axis = 0; axis < batch_shape.GetRank(); ++axis) {
    if (axis < rank_offset) {
      continue;
    }
    const auto tensor_axis = axis - rank_offset;
    const auto extent = tensor.GetShape().GetDimension(tensor_axis, location);
    if (extent != 1) {
      aligned_strides[axis] = tensor.GetStrides().GetStride(tensor_axis, location);
      has_nonzero_stride = true;
    }
  }
  if (!has_nonzero_stride) {
    return int64_t{0};
  }

  auto candidate = int64_t{0};
  auto trailing_extent = int64_t{1};
  for (size_t remaining_rank = batch_shape.GetRank(); remaining_rank > 0; --remaining_rank) {
    const auto axis = remaining_rank - 1;
    const auto extent = batch_shape.GetDimension(axis, location);
    if (extent <= 1) {
      continue;
    }
    if (candidate == 0) {
      candidate = aligned_strides[axis];
      if (candidate == 0) {
        return std::nullopt;
      }
    }
    const auto expected =
        internal::CheckedMultiply(candidate, trailing_extent, "matmul flattened batch stride", location);
    if (aligned_strides[axis] != expected) {
      return std::nullopt;
    }
    trailing_extent = internal::CheckedMultiply(trailing_extent, extent, "matmul flattened batch extent", location);
  }
  return candidate;
}

[[nodiscard]] auto TryDirectMatrix(const Tensor &tensor, const Shape &batch_shape, uint64_t rows, uint64_t columns,
                                   bool flatten_prefix, std::source_location location) -> std::optional<MatrixInfo> {
  if (flatten_prefix) {
    if (!tensor.IsContiguous()) {
      return std::nullopt;
    }
    return MatrixInfo{
        .data_ = internal::TensorAccess::GetData(tensor, location),
        .rows_ = rows,
        .columns_ = columns,
        .leading_dimension_ = internal::CheckedNarrow<int64_t>(std::max<uint64_t>(columns, 1), "matrix ld", location),
        .batch_stride_ = 0,
        .order_ = CUBLASLT_ORDER_ROW,
    };
  }

  const auto row_stride = tensor.GetStrides().GetStride(tensor.GetRank() - 2, location);
  const auto column_stride = tensor.GetStrides().GetStride(tensor.GetRank() - 1, location);
  auto order = CUBLASLT_ORDER_ROW;
  auto leading_dimension = int64_t{0};
  const auto row_compatible =
      (columns <= 1 || column_stride == 1) &&
      (rows <= 1 || row_stride >= internal::CheckedNarrow<int64_t>(columns, "matrix columns", location));
  const auto column_compatible =
      (rows <= 1 || row_stride == 1) &&
      (columns <= 1 || column_stride >= internal::CheckedNarrow<int64_t>(rows, "matrix rows", location));
  if (row_compatible) {
    order = CUBLASLT_ORDER_ROW;
    leading_dimension = rows <= 1
                            ? internal::CheckedNarrow<int64_t>(std::max<uint64_t>(columns, 1), "row-major ld", location)
                            : row_stride;
  } else if (column_compatible) {
    order = CUBLASLT_ORDER_COL;
    leading_dimension = columns <= 1
                            ? internal::CheckedNarrow<int64_t>(std::max<uint64_t>(rows, 1), "column-major ld", location)
                            : column_stride;
  } else {
    return std::nullopt;
  }
  const auto batch_stride = GetUniformBatchStride(tensor, batch_shape, location);
  if (!batch_stride.has_value()) {
    return std::nullopt;
  }
  return MatrixInfo{
      .data_ = internal::TensorAccess::GetData(tensor, location),
      .rows_ = rows,
      .columns_ = columns,
      .leading_dimension_ = leading_dimension,
      .batch_stride_ = *batch_stride,
      .order_ = order,
  };
}

[[nodiscard]] auto TryDirectOutput(Tensor &output, const Shape &batch_shape, uint64_t rows, uint64_t columns,
                                   bool flatten_prefix, std::source_location location)
    -> std::optional<MutableMatrixInfo> {
  const auto matrix = TryDirectMatrix(output, batch_shape, rows, columns, flatten_prefix, location);
  if (!matrix.has_value()) {
    return std::nullopt;
  }
  return MutableMatrixInfo{
      .data_ = internal::TensorAccess::GetMutableData(output, location),
      .rows_ = matrix->rows_,
      .columns_ = matrix->columns_,
      .leading_dimension_ = matrix->leading_dimension_,
      .batch_stride_ = matrix->batch_stride_,
      .order_ = matrix->order_,
  };
}

[[nodiscard]] auto MakeContiguousMatrix(const void *data, uint64_t rows, uint64_t columns, int64_t batch_count,
                                        std::source_location location) -> MatrixInfo {
  return MatrixInfo{
      .data_ = data,
      .rows_ = rows,
      .columns_ = columns,
      .leading_dimension_ = internal::CheckedNarrow<int64_t>(std::max<uint64_t>(columns, 1), "matrix ld", location),
      .batch_stride_ = batch_count > 1 ? internal::CheckedNarrow<int64_t>(
                                             internal::CheckedMultiply(rows, columns, "matrix batch stride", location),
                                             "matrix batch stride", location)
                                       : 0,
      .order_ = CUBLASLT_ORDER_ROW,
  };
}

[[nodiscard]] auto MakeContiguousOutput(void *data, uint64_t rows, uint64_t columns, int64_t batch_count,
                                        std::source_location location) -> MutableMatrixInfo {
  const auto matrix = MakeContiguousMatrix(data, rows, columns, batch_count, location);
  return {
      .data_ = data,
      .rows_ = matrix.rows_,
      .columns_ = matrix.columns_,
      .leading_dimension_ = matrix.leading_dimension_,
      .batch_stride_ = matrix.batch_stride_,
      .order_ = matrix.order_,
  };
}

template <typename T>
void SetMatmulAttribute(cublasLtMatmulDesc_t descriptor, cublasLtMatmulDescAttributes_t attribute, const T &value,
                        std::source_location location) {
  internal::CheckCublas(internal::GetCublasApi().lt_matmul_desc_set_attribute_(
                            descriptor, attribute, static_cast<const void *>(&value), sizeof(value)),
                        "cublasLtMatmulDescSetAttribute", location);
}

template <typename T>
void SetLayoutAttribute(cublasLtMatrixLayout_t descriptor, cublasLtMatrixLayoutAttribute_t attribute, const T &value,
                        std::source_location location) {
  internal::CheckCublas(
      internal::GetCublasApi().lt_matrix_layout_set_attribute_(descriptor, attribute, &value, sizeof(value)),
      "cublasLtMatrixLayoutSetAttribute", location);
}

template <typename T>
void SetPreferenceAttribute(cublasLtMatmulPreference_t preference, cublasLtMatmulPreferenceAttributes_t attribute,
                            const T &value, std::source_location location) {
  internal::CheckCublas(
      internal::GetCublasApi().lt_matmul_preference_set_attribute_(preference, attribute, &value, sizeof(value)),
      "cublasLtMatmulPreferenceSetAttribute", location);
}

void InitializeLayout(cublasLtMatrixLayout_t descriptor, cudaDataType_t dtype, const MatrixInfo &matrix,
                      int32_t batch_count, std::source_location location) {
  internal::CheckCublas(internal::GetCublasApi().lt_matrix_layout_init_(descriptor, dtype, matrix.rows_,
                                                                        matrix.columns_, matrix.leading_dimension_),
                        "cublasLtMatrixLayoutInit", location);
  SetLayoutAttribute(descriptor, CUBLASLT_MATRIX_LAYOUT_ORDER, matrix.order_, location);
  if (batch_count > 1) {
    SetLayoutAttribute(descriptor, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_count, location);
    SetLayoutAttribute(descriptor, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, matrix.batch_stride_, location);
  }
}

void InitializeLayout(cublasLtMatrixLayout_t descriptor, cudaDataType_t dtype, const MutableMatrixInfo &matrix,
                      int32_t batch_count, std::source_location location) {
  InitializeLayout(descriptor, dtype,
                   MatrixInfo{
                       .data_ = matrix.data_,
                       .rows_ = matrix.rows_,
                       .columns_ = matrix.columns_,
                       .leading_dimension_ = matrix.leading_dimension_,
                       .batch_stride_ = matrix.batch_stride_,
                       .order_ = matrix.order_,
                   },
                   batch_count, location);
}

[[nodiscard]] auto GetLayoutSignature(const MatrixInfo &matrix) noexcept -> internal::MatrixLayoutSignature {
  return {
      .rows_ = matrix.rows_,
      .columns_ = matrix.columns_,
      .leading_dimension_ = matrix.leading_dimension_,
      .batch_stride_ = matrix.batch_stride_,
      .order_ = static_cast<int32_t>(matrix.order_),
  };
}

[[nodiscard]] auto GetLayoutSignature(const MutableMatrixInfo &matrix) noexcept -> internal::MatrixLayoutSignature {
  return {
      .rows_ = matrix.rows_,
      .columns_ = matrix.columns_,
      .leading_dimension_ = matrix.leading_dimension_,
      .batch_stride_ = matrix.batch_stride_,
      .order_ = static_cast<int32_t>(matrix.order_),
  };
}

[[nodiscard]] auto TryLaunchCublasLt(internal::OpGuard &guard, const MatrixInfo &lhs, const MatrixInfo &rhs,
                                     const MutableMatrixInfo &output, DType dtype, int64_t batch_count,
                                     cublasOperation_t lhs_operation, cublasOperation_t rhs_operation,
                                     cublasLtEpilogue_t epilogue, const void *bias, const MatmulOptions &options,
                                     std::source_location location) -> bool {
  const auto data_type = GetCudaDataType(dtype, location);
  cublasLtMatmulDescOpaque_t operation_storage{};
  auto *operation = &operation_storage;
  internal::CheckCublas(
      internal::GetCublasApi().lt_matmul_desc_init_(operation, GetComputeType(dtype, options), CUDA_R_32F),
      "cublasLtMatmulDescInit", location);
  SetMatmulAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, lhs_operation, location);
  SetMatmulAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, rhs_operation, location);
  SetMatmulAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue, location);
  if (bias != nullptr) {
    SetMatmulAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias, location);
  }

  cublasLtMatrixLayoutOpaque_t lhs_layout_storage{};
  cublasLtMatrixLayoutOpaque_t rhs_layout_storage{};
  cublasLtMatrixLayoutOpaque_t output_layout_storage{};
  auto *lhs_layout = &lhs_layout_storage;
  auto *rhs_layout = &rhs_layout_storage;
  auto *output_layout = &output_layout_storage;
  const auto narrowed_batch_count = internal::CheckedNarrow<int32_t>(batch_count, "matmul batch count", location);
  InitializeLayout(lhs_layout, data_type, lhs, narrowed_batch_count, location);
  InitializeLayout(rhs_layout, data_type, rhs, narrowed_batch_count, location);
  InitializeLayout(output_layout, data_type, output, narrowed_batch_count, location);

  cublasLtMatmulPreferenceOpaque_t preference_storage{};
  auto *preference = &preference_storage;
  internal::CheckCublas(internal::GetCublasApi().lt_matmul_preference_init_(preference), "cublasLtMatmulPreferenceInit",
                        location);
  const auto workspace_bytes = guard.GetBlasWorkspaceBytes();
  const auto lhs_alignment = GetPointerAlignment(lhs.data_);
  const auto rhs_alignment = GetPointerAlignment(rhs.data_);
  const auto output_alignment = GetPointerAlignment(output.data_);
  const auto bias_alignment = GetPointerAlignment(bias);
  SetPreferenceAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_bytes, location);
  SetPreferenceAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, lhs_alignment, location);
  SetPreferenceAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, rhs_alignment, location);
  SetPreferenceAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, output_alignment, location);
  SetPreferenceAttribute(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, output_alignment, location);

  const auto key = internal::MatmulAlgorithmKey{
      .lhs_ = GetLayoutSignature(lhs),
      .rhs_ = GetLayoutSignature(rhs),
      .output_ = GetLayoutSignature(output),
      .batch_count_ = batch_count,
      .data_type_ = static_cast<int32_t>(data_type),
      .compute_type_ = static_cast<int32_t>(GetComputeType(dtype, options)),
      .lhs_operation_ = static_cast<int32_t>(lhs_operation),
      .rhs_operation_ = static_cast<int32_t>(rhs_operation),
      .epilogue_ = static_cast<int32_t>(epilogue),
      .lhs_alignment_ = lhs_alignment,
      .rhs_alignment_ = rhs_alignment,
      .output_alignment_ = output_alignment,
      .bias_alignment_ = bias_alignment,
      .workspace_limit_bytes_ = workspace_bytes,
  };
  auto choice = guard.GetMatmulAlgorithmCache().Find(key);
  if (!choice.has_value()) {
    // cuBLASLt heuristic discovery performs host-side work and is not capture-safe. Cache both successful choices and
    // unsupported signatures so capture either reuses a warmed decision or fails before native graph mutation.
    if (guard.IsCapturing()) {
      throw CaptureError("cuBLASLt algorithm must be warmed up before CUDA graph capture", location);
    }
    auto heuristic_results = std::array<cublasLtMatmulHeuristicResult_t, MATMUL_HEURISTIC_RESULT_COUNT>{};
    int result_count = 0;
    const auto heuristic_status = internal::GetCublasApi().lt_matmul_algo_get_heuristic_(
        guard.GetCublasLtHandle(), operation, lhs_layout, rhs_layout, output_layout, output_layout, preference,
        MATMUL_HEURISTIC_RESULT_COUNT, heuristic_results.data(), &result_count);
    if (heuristic_status == CUBLAS_STATUS_NOT_SUPPORTED) {
      choice = internal::MatmulAlgorithmChoice{.algorithm_ = {}, .workspace_bytes_ = 0, .supported_ = false};
    } else {
      internal::CheckCublas(heuristic_status, "cublasLtMatmulAlgoGetHeuristic", location);
      if (result_count < 0 || result_count > MATMUL_HEURISTIC_RESULT_COUNT) {
        throw InternalError("cublasLtMatmulAlgoGetHeuristic returned an invalid result count", location);
      }
      const auto result = std::ranges::find_if(heuristic_results.begin(), heuristic_results.begin() + result_count,
                                               [workspace_bytes](const auto &candidate) {
                                                 return candidate.state == CUBLAS_STATUS_SUCCESS &&
                                                        candidate.workspaceSize <= workspace_bytes;
                                               });
      choice = result == heuristic_results.begin() + result_count
                   ? internal::MatmulAlgorithmChoice{.algorithm_ = {}, .workspace_bytes_ = 0, .supported_ = false}
                   : internal::MatmulAlgorithmChoice{
                         .algorithm_ = result->algo,
                         .workspace_bytes_ = result->workspaceSize,
                         .supported_ = true,
                     };
    }
    guard.GetMatmulAlgorithmCache().Insert(key, *choice);
  }
  if (!choice->supported_) {
    return false;
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  auto *workspace = guard.GetBlasWorkspace();
  internal::CheckCublas(internal::GetCublasApi().lt_matmul_(
                            guard.GetCublasLtHandle(), operation, &alpha, lhs.data_, lhs_layout, rhs.data_, rhs_layout,
                            &beta, output.data_, output_layout, output.data_, output_layout, &choice->algorithm_,
                            workspace, choice->workspace_bytes_, guard.GetNativeStream()),
                        "cublasLtMatmul", location);
  return true;
}

[[nodiscard]] auto GetRequestedEpilogue(bool fuse_bias, bool fuse_relu) noexcept -> cublasLtEpilogue_t {
  if (fuse_relu) {
    return fuse_bias ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_RELU;
  }
  return fuse_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
}

[[nodiscard]] auto ExecuteMatmul(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                                 const std::optional<Tensor> &bias, const MatmulProblem &problem,
                                 std::source_location location) -> MatmulExecutionResult {
  internal::OpGuard guard{context, GetName(problem.kind_), location, internal::CapturePolicy::SAFE};
  ValidateMatmulOutput(guard, output, lhs, rhs, bias, problem, location);
  if (output.GetNumElements() == 0) {
    return {};
  }

  const auto &shape = problem.shape_;
  const auto is_linear = problem.kind_ == MatmulKind::LINEAR;
  const auto batch_count = shape.batch_count_;
  const auto lhs_rows = static_cast<uint64_t>(shape.m_);
  const auto lhs_columns = static_cast<uint64_t>(shape.k_);
  const auto rhs_rows = static_cast<uint64_t>(is_linear ? shape.n_ : shape.k_);
  const auto rhs_columns = static_cast<uint64_t>(is_linear ? shape.k_ : shape.n_);
  const auto output_rows = static_cast<uint64_t>(shape.m_);
  const auto output_columns = static_cast<uint64_t>(shape.n_);
  auto scratch = guard.MakeScratchScope();

  const auto lhs_logical_shape =
      is_linear ? lhs.GetShape() : MakeBatchedMatrixShape(shape.batch_shape_, shape.m_, shape.k_, location);
  const auto rhs_logical_shape =
      is_linear ? rhs.GetShape() : MakeBatchedMatrixShape(shape.batch_shape_, shape.k_, shape.n_, location);

  auto lhs_matrix = TryDirectMatrix(lhs, shape.batch_shape_, lhs_rows, lhs_columns, is_linear, location);
  if (!lhs_matrix.has_value()) {
    const auto bytes = internal::CheckedBytes(lhs_logical_shape.GetNumElements(),
                                              GetDTypeInfo(lhs.GetDType(), location).size_bytes_, location);
    const auto allocation = scratch.AllocateBytes(bytes, 256, location);
    const auto parameters = BuildMaterializationParameters(allocation.GetData(), lhs, lhs_logical_shape, location);
    internal::LaunchCompositionCopy(guard.GetNativeStream(), lhs.GetDType(),
                                    internal::GetCompositionIndexWidth(parameters), parameters, location);
    lhs_matrix = MakeContiguousMatrix(allocation.GetData(), lhs_rows, lhs_columns, batch_count, location);
  }

  auto rhs_matrix = TryDirectMatrix(rhs, shape.batch_shape_, rhs_rows, rhs_columns, false, location);
  if (!rhs_matrix.has_value()) {
    const auto bytes = internal::CheckedBytes(rhs_logical_shape.GetNumElements(),
                                              GetDTypeInfo(rhs.GetDType(), location).size_bytes_, location);
    const auto allocation = scratch.AllocateBytes(bytes, 256, location);
    const auto parameters = BuildMaterializationParameters(allocation.GetData(), rhs, rhs_logical_shape, location);
    internal::LaunchCompositionCopy(guard.GetNativeStream(), rhs.GetDType(),
                                    internal::GetCompositionIndexWidth(parameters), parameters, location);
    rhs_matrix = MakeContiguousMatrix(allocation.GetData(), rhs_rows, rhs_columns, batch_count, location);
  }

  auto output_matrix = TryDirectOutput(output, shape.batch_shape_, output_rows, output_columns, is_linear, location);
  auto output_scratch = static_cast<void *>(nullptr);
  if (!output_matrix.has_value()) {
    const auto bytes = internal::CheckedBytes(output.GetNumElements(),
                                              GetDTypeInfo(output.GetDType(), location).size_bytes_, location);
    const auto allocation = scratch.AllocateBytes(bytes, 256, location);
    output_scratch = allocation.GetData();
    output_matrix = MakeContiguousOutput(output_scratch, output_rows, output_columns, batch_count, location);
  }

  guard.RecordTensor(output);
  guard.RecordTensor(lhs);
  guard.RecordTensor(rhs);
  if (bias.has_value()) {
    guard.RecordTensor(*bias);
  }

  const auto fuse_bias = problem.bias_mode_ == LinearBiasMode::FUSED_CONTIGUOUS;
  const auto fuse_relu = is_linear && problem.linear_options_.activation_ == LinearActivation::RELU &&
                         problem.bias_mode_ != LinearBiasMode::POST_ADD;
  const auto requested_epilogue = is_linear ? GetRequestedEpilogue(fuse_bias, fuse_relu) : CUBLASLT_EPILOGUE_DEFAULT;
  const auto requested_bias = fuse_bias ? internal::TensorAccess::GetData(*bias, location) : nullptr;
  auto launched = TryLaunchCublasLt(guard, *lhs_matrix, *rhs_matrix, *output_matrix, output.GetDType(), batch_count,
                                    CUBLAS_OP_N, is_linear ? CUBLAS_OP_T : CUBLAS_OP_N, requested_epilogue,
                                    requested_bias, problem.linear_options_.matmul_, location);
  auto bias_fused = requested_bias != nullptr;
  auto activation_fused = fuse_relu;
  if (!launched && requested_epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
    launched = TryLaunchCublasLt(guard, *lhs_matrix, *rhs_matrix, *output_matrix, output.GetDType(), batch_count,
                                 CUBLAS_OP_N, is_linear ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLASLT_EPILOGUE_DEFAULT, nullptr,
                                 problem.linear_options_.matmul_, location);
    bias_fused = false;
    activation_fused = false;
  }
  if (!launched) {
    throw NotSupportedError("cuBLASLt did not provide a compatible matmul algorithm", location);
  }
  if (output_scratch != nullptr) {
    const auto parameters = BuildCopyBackParameters(output, output_scratch, location);
    internal::LaunchCompositionCopy(guard.GetNativeStream(), output.GetDType(),
                                    internal::GetCompositionIndexWidth(parameters), parameters, location);
  }
  guard.CheckLaunch();
  return {
      .bias_fused_ = bias_fused,
      .activation_fused_ = activation_fused,
  };
}

void ApplyLinearFallback(ExecutionContext &context, Tensor &output, const std::optional<Tensor> &bias,
                         const LinearOptions &options, const MatmulExecutionResult &execution,
                         std::source_location location) {
  if (bias.has_value() && !execution.bias_fused_) {
    AddOut(context, output, output, *bias, location);
  }
  if (execution.activation_fused_) {
    return;
  }
  switch (options.activation_) {
    case LinearActivation::NONE:
      return;
    case LinearActivation::RELU:
      ReluOut(context, output, output, location);
      return;
    case LinearActivation::GELU:
      GeluOut(context, output, output, options.gelu_approximation_, location);
      return;
  }
  throw InternalError("invalid Linear activation reached fallback", location);
}

void ExecuteMatmulProblem(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                          const std::optional<Tensor> &bias, const MatmulProblem &problem,
                          std::source_location location) {
  if (problem.shape_.k_ == 0) {
    {
      internal::OpGuard guard{context, GetName(problem.kind_), location, internal::CapturePolicy::SAFE};
      ValidateMatmulOutput(guard, output, lhs, rhs, bias, problem, location);
    }
    FillOut(context, output, Scalar{int64_t{0}}, location);
    if (problem.kind_ == MatmulKind::LINEAR) {
      ApplyLinearFallback(context, output, bias, problem.linear_options_, {}, location);
    }
    return;
  }
  const auto execution = ExecuteMatmul(context, output, lhs, rhs, bias, problem, location);
  if (problem.kind_ == MatmulKind::LINEAR) {
    ApplyLinearFallback(context, output, bias, problem.linear_options_, execution, location);
  }
}

void MatmulOutImpl(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs, MatmulKind kind,
                   const MatmulOptions &options, std::source_location location) {
  const auto problem =
      BuildMatmulProblem(context, lhs, rhs, kind, std::nullopt, LinearOptions{.matmul_ = options}, location);
  ExecuteMatmulProblem(context, output, lhs, rhs, std::nullopt, problem, location);
}

[[nodiscard]] auto MatmulImpl(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs, MatmulKind kind,
                              const MatmulOptions &options, std::source_location location) -> Tensor {
  const auto problem =
      BuildMatmulProblem(context, lhs, rhs, kind, std::nullopt, LinearOptions{.matmul_ = options}, location);
  auto output = Empty(context, problem.shape_.output_shape_, lhs.GetDType(), location);
  ExecuteMatmulProblem(context, output, lhs, rhs, std::nullopt, problem, location);
  return output;
}

}  // namespace

void MatmulOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
               const MatmulOptions &options, std::source_location location) {
  MatmulOutImpl(context, output, lhs, rhs, MatmulKind::MATMUL, options, location);
}

auto Matmul(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs, const MatmulOptions &options,
            std::source_location location) -> Tensor {
  return MatmulImpl(context, lhs, rhs, MatmulKind::MATMUL, options, location);
}

void BatchedMatmulOut(ExecutionContext &context, Tensor &output, const Tensor &lhs, const Tensor &rhs,
                      const MatmulOptions &options, std::source_location location) {
  MatmulOutImpl(context, output, lhs, rhs, MatmulKind::BATCHED_MATMUL, options, location);
}

auto BatchedMatmul(ExecutionContext &context, const Tensor &lhs, const Tensor &rhs, const MatmulOptions &options,
                   std::source_location location) -> Tensor {
  return MatmulImpl(context, lhs, rhs, MatmulKind::BATCHED_MATMUL, options, location);
}

void LinearOut(ExecutionContext &context, Tensor &output, const Tensor &input, const Tensor &weight,
               const std::optional<Tensor> &bias, const LinearOptions &options, std::source_location location) {
  const auto problem = BuildMatmulProblem(context, input, weight, MatmulKind::LINEAR, bias, options, location);
  ExecuteMatmulProblem(context, output, input, weight, bias, problem, location);
}

auto Linear(ExecutionContext &context, const Tensor &input, const Tensor &weight, const std::optional<Tensor> &bias,
            const LinearOptions &options, std::source_location location) -> Tensor {
  const auto problem = BuildMatmulProblem(context, input, weight, MatmulKind::LINEAR, bias, options, location);
  auto output = Empty(context, problem.shape_.output_shape_, input.GetDType(), location);
  ExecuteMatmulProblem(context, output, input, weight, bias, problem, location);
  return output;
}

}  // namespace ttl
