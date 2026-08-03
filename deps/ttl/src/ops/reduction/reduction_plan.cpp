#include "ttl/internal/ops/reduction.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <source_location>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/device_properties.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::internal {
namespace {

constexpr uint64_t REDUCTION_THREADS_PER_BLOCK = 256;
constexpr uint64_t TARGET_VALUES_PER_PARTIAL = REDUCTION_THREADS_PER_BLOCK * 8;
constexpr uint32_t MAXIMUM_PARTIALS_PER_OUTPUT = 32;
constexpr uint64_t TARGET_BLOCKS_PER_MULTIPROCESSOR = 4;
constexpr uint64_t WARPS_PER_BLOCK = REDUCTION_THREADS_PER_BLOCK / 32;
constexpr uint64_t MAXIMUM_LAUNCH_BLOCKS = 65535;
constexpr size_t INDEXED_VALUE_BYTES = 16;

[[nodiscard]] auto GetAccumulatorSize(ReductionOp operation, DType dtype, std::source_location location) -> size_t {
  switch (operation) {
    case ReductionOp::SUM:
      return IsFloating(dtype, location) ? sizeof(float) : sizeof(uint64_t);
    case ReductionOp::MEAN:
      return sizeof(float);
    case ReductionOp::MINIMUM:
    case ReductionOp::MAXIMUM:
    case ReductionOp::ARG_MIN:
    case ReductionOp::ARG_MAX:
      return INDEXED_VALUE_BYTES;
    case ReductionOp::ANY:
    case ReductionOp::ALL:
      return sizeof(uint32_t);
  }
  throw InternalError("invalid reduction operation while computing accumulator size", location);
}

[[nodiscard]] auto GetPartialCount(uint64_t output_count, uint64_t reduction_count,
                                   const DeviceProperties &properties) noexcept -> uint32_t {
  if (output_count == 0 || reduction_count <= TARGET_VALUES_PER_PARTIAL) {
    return 1;
  }

  const auto useful_partials = CeilDivide(reduction_count, TARGET_VALUES_PER_PARTIAL);
  const auto target_blocks = static_cast<uint64_t>(properties.multiprocessor_count_) * TARGET_BLOCKS_PER_MULTIPROCESSOR;
  const auto occupancy_partials = CeilDivide(target_blocks, output_count);
  const auto partial_count =
      std::min({useful_partials, occupancy_partials, static_cast<uint64_t>(MAXIMUM_PARTIALS_PER_OUTPUT)});
  return static_cast<uint32_t>(std::max(uint64_t{1}, partial_count));
}

[[nodiscard]] auto GetLaunchBlockCount(ReductionPath path, uint64_t output_count, uint32_t partial_count,
                                       const DeviceProperties &properties) noexcept -> uint32_t {
  auto task_count = output_count;
  if (path == ReductionPath::WARP) {
    task_count = CeilDivide(output_count, WARPS_PER_BLOCK);
  } else if (path == ReductionPath::TWO_STAGE) {
    task_count = output_count * partial_count;
  }
  const auto occupancy_blocks =
      static_cast<uint64_t>(properties.multiprocessor_count_) * TARGET_BLOCKS_PER_MULTIPROCESSOR;
  const auto block_count =
      std::min({std::max(uint64_t{1}, task_count), std::max(uint64_t{1}, occupancy_blocks), MAXIMUM_LAUNCH_BLOCKS});
  return static_cast<uint32_t>(block_count);
}

[[nodiscard]] auto IsReductionContiguous(const Tensor &input, std::span<const size_t> axes, size_t element_size,
                                         std::source_location location) -> bool {
  auto expected_stride_bytes = element_size;
  const auto dimensions = input.GetShape().GetDimensions();
  const auto strides = input.GetStrides().GetValues();
  for (size_t remaining = axes.size(); remaining > 0; --remaining) {
    const auto axis = axes[remaining - 1];
    if (dimensions[axis] > 1) {
      const auto stride_bytes = CheckedBytes(strides[axis], element_size, location);
      if (stride_bytes != expected_stride_bytes) {
        return false;
      }
    }
    const auto extent = dimensions[axis] == 0 ? uint64_t{1} : static_cast<uint64_t>(dimensions[axis]);
    expected_stride_bytes =
        CheckedMultiply(expected_stride_bytes, CheckedNarrow<size_t>(extent, "reduction extent", location),
                        "contiguous reduction byte stride", location);
  }
  return true;
}

[[nodiscard]] auto CanUse32BitIndexing(const ReductionParameters64 &parameters) noexcept -> bool {
  constexpr auto maximum = uint64_t{std::numeric_limits<uint32_t>::max()};
  if (parameters.output_count_ > maximum || parameters.reduction_count_ > maximum) {
    return false;
  }

  auto maximum_input_offset = uint64_t{0};
  auto maximum_output_offset = uint64_t{0};
  for (size_t axis = 0; axis < parameters.output_rank_; ++axis) {
    if (parameters.output_shape_[axis] > maximum || parameters.output_input_strides_bytes_[axis] > maximum ||
        parameters.output_strides_bytes_[axis] > maximum) {
      return false;
    }
    if (parameters.output_shape_[axis] == 0) {
      continue;
    }
    const auto extent = parameters.output_shape_[axis] - 1;
    const auto input_stride = parameters.output_input_strides_bytes_[axis];
    const auto output_stride = parameters.output_strides_bytes_[axis];
    if ((input_stride != 0 && extent > (maximum - maximum_input_offset) / input_stride) ||
        (output_stride != 0 && extent > (maximum - maximum_output_offset) / output_stride)) {
      return false;
    }
    maximum_input_offset += extent * input_stride;
    maximum_output_offset += extent * output_stride;
  }

  auto maximum_reduction_offset = uint64_t{0};
  for (size_t axis = 0; axis < parameters.reduction_rank_; ++axis) {
    if (parameters.reduction_shape_[axis] > maximum || parameters.reduction_strides_bytes_[axis] > maximum) {
      return false;
    }
    if (parameters.reduction_shape_[axis] == 0) {
      continue;
    }
    const auto extent = parameters.reduction_shape_[axis] - 1;
    const auto stride = parameters.reduction_strides_bytes_[axis];
    if (stride != 0 && extent > (maximum - maximum_reduction_offset) / stride) {
      return false;
    }
    maximum_reduction_offset += extent * stride;
  }
  return maximum_input_offset <= maximum - maximum_reduction_offset;
}

template <typename Destination>
void CopyParameters(const ReductionParameters64 &source, Destination &destination, std::source_location location) {
  using Index = std::remove_cvref_t<decltype(destination.output_count_)>;
  destination.output_ = source.output_;
  destination.input_ = source.input_;
  destination.output_count_ = CheckedNarrow<Index>(source.output_count_, "reduction output count", location);
  destination.reduction_count_ = CheckedNarrow<Index>(source.reduction_count_, "reduction element count", location);
  destination.partial_count_ = source.partial_count_;
  destination.output_rank_ = source.output_rank_;
  destination.reduction_rank_ = source.reduction_rank_;
  destination.contiguous_reduction_ = source.contiguous_reduction_;
  for (size_t axis = 0; axis < TTL_MAX_RANK; ++axis) {
    destination.output_shape_[axis] =
        CheckedNarrow<Index>(source.output_shape_[axis], "reduction output shape", location);
    destination.output_input_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.output_input_strides_bytes_[axis], "reduction input base stride", location);
    destination.output_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.output_strides_bytes_[axis], "reduction output stride", location);
    destination.reduction_shape_[axis] =
        CheckedNarrow<Index>(source.reduction_shape_[axis], "reduction shape", location);
    destination.reduction_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.reduction_strides_bytes_[axis], "reduction input stride", location);
  }
}

}  // namespace

auto GetReductionName(ReductionOp operation) noexcept -> std::string_view {
  switch (operation) {
    case ReductionOp::SUM:
      return "SumOut";
    case ReductionOp::MEAN:
      return "MeanOut";
    case ReductionOp::MINIMUM:
      return "MinimumOut";
    case ReductionOp::MAXIMUM:
      return "MaximumOut";
    case ReductionOp::ARG_MIN:
      return "ArgMinOut";
    case ReductionOp::ARG_MAX:
      return "ArgMaxOut";
    case ReductionOp::ANY:
      return "AnyOut";
    case ReductionOp::ALL:
      return "AllOut";
  }
  return "<invalid reduction operation>";
}

auto ResolveReductionAxes(std::span<const int64_t> axes, size_t rank, std::source_location location)
    -> std::vector<size_t> {
  if (!axes.empty()) {
    return NormalizeAxes(axes, rank, location);
  }
  std::vector<size_t> normalized(rank);
  std::iota(normalized.begin(), normalized.end(), size_t{0});
  return normalized;
}

auto InferReductionShape(const Shape &input_shape, std::span<const size_t> axes, bool keep_dimensions,
                         std::source_location location) -> Shape {
  std::array<bool, TTL_MAX_RANK> reduced{};
  for (const auto axis : axes) {
    if (axis >= input_shape.GetRank() || reduced[axis]) {
      throw InvalidArgumentError("reduction axes must be normalized, unique, and in range", location);
    }
    reduced[axis] = true;
  }

  std::array<int64_t, TTL_MAX_RANK> output_dimensions{};
  size_t output_rank = 0;
  for (size_t axis = 0; axis < input_shape.GetRank(); ++axis) {
    if (reduced[axis]) {
      if (keep_dimensions) {
        output_dimensions[output_rank] = 1;
        ++output_rank;
      }
      continue;
    }
    output_dimensions[output_rank] = input_shape.GetDimension(axis, location);
    ++output_rank;
  }
  return Shape{std::span<const int64_t>{output_dimensions.data(), output_rank}, location};
}

auto BuildReductionPlan(Tensor &output, const Tensor &input, std::span<const size_t> axes, ReductionOp operation,
                        const DeviceProperties &properties, std::source_location location) -> ReductionPlan {
  ReductionPlan plan;
  auto &parameters = plan.parameters_;
  parameters.output_ = static_cast<std::byte *>(TensorAccess::GetMutableData(output, location));
  parameters.input_ = static_cast<const std::byte *>(TensorAccess::GetData(input, location));
  parameters.output_count_ = static_cast<uint64_t>(output.GetNumElements());

  const auto input_dimensions = input.GetShape().GetDimensions();
  const auto input_strides = input.GetStrides().GetValues();
  const auto output_strides = output.GetStrides().GetValues();
  const auto element_size = GetDTypeSize(input.GetDType(), location);

  std::array<bool, TTL_MAX_RANK> reduced{};
  for (const auto axis : axes) {
    if (axis >= input.GetRank() || reduced[axis]) {
      throw InvalidArgumentError("reduction plan requires normalized, unique axes", location);
    }
    reduced[axis] = true;
  }

  auto output_axis = size_t{0};
  auto reduction_count = uint64_t{1};
  for (size_t input_axis = 0; input_axis < input.GetRank(); ++input_axis) {
    const auto extent = static_cast<uint64_t>(input_dimensions[input_axis]);
    const auto input_stride_bytes = CheckedBytes(input_strides[input_axis], element_size, location);
    if (reduced[input_axis]) {
      const auto reduction_axis = static_cast<size_t>(parameters.reduction_rank_);
      parameters.reduction_shape_[reduction_axis] = extent;
      parameters.reduction_strides_bytes_[reduction_axis] = input_stride_bytes;
      ++parameters.reduction_rank_;
      reduction_count = CheckedMultiply(reduction_count, extent, "reduction group element count", location);
      if (output.GetRank() == input.GetRank()) {
        ++output_axis;
      }
      continue;
    }

    const auto plan_axis = static_cast<size_t>(parameters.output_rank_);
    parameters.output_shape_[plan_axis] = extent;
    parameters.output_input_strides_bytes_[plan_axis] = input_stride_bytes;
    parameters.output_strides_bytes_[plan_axis] =
        CheckedBytes(output_strides[output_axis], GetDTypeSize(output.GetDType(), location), location);
    ++parameters.output_rank_;
    ++output_axis;
  }
  parameters.reduction_count_ = reduction_count;
  parameters.contiguous_reduction_ = IsReductionContiguous(input, axes, element_size, location);

  const auto partial_count = GetPartialCount(parameters.output_count_, reduction_count, properties);
  parameters.partial_count_ = partial_count;
  if (reduction_count <= 32) {
    plan.path_ = ReductionPath::WARP;
  } else if (partial_count > 1) {
    plan.path_ = ReductionPath::TWO_STAGE;
  } else {
    plan.path_ = ReductionPath::BLOCK;
  }

  if (plan.path_ == ReductionPath::TWO_STAGE) {
    const auto partial_values = CheckedMultiply(parameters.output_count_, static_cast<uint64_t>(partial_count),
                                                "reduction partial accumulator count", location);
    plan.scratch_bytes_ =
        CheckedBytes(partial_values, GetAccumulatorSize(operation, input.GetDType(), location), location);
  }
  plan.launch_block_count_ = GetLaunchBlockCount(plan.path_, parameters.output_count_, partial_count, properties);
  plan.index_width_ = CanUse32BitIndexing(parameters) ? IndexWidth::UINT32 : IndexWidth::UINT64;
  return plan;
}

auto ReductionPlan::GetPath() const noexcept -> ReductionPath { return path_; }

auto ReductionPlan::GetIndexWidth() const noexcept -> IndexWidth { return index_width_; }

auto ReductionPlan::GetOutputCount() const noexcept -> uint64_t { return parameters_.output_count_; }

auto ReductionPlan::GetReductionCount() const noexcept -> uint64_t { return parameters_.reduction_count_; }

auto ReductionPlan::GetPartialCount() const noexcept -> uint32_t { return parameters_.partial_count_; }

auto ReductionPlan::GetScratchBytes() const noexcept -> size_t { return scratch_bytes_; }

auto ReductionPlan::GetLaunchBlockCount() const noexcept -> uint32_t { return launch_block_count_; }

auto ReductionPlan::MakeParameters32(std::source_location location) const -> ReductionParameters32 {
  if (index_width_ != IndexWidth::UINT32) {
    throw InternalError("cannot create 32-bit parameters for a 64-bit reduction plan", location);
  }
  ReductionParameters32 parameters{};
  CopyParameters(parameters_, parameters, location);
  return parameters;
}

auto ReductionPlan::MakeParameters64() const noexcept -> ReductionParameters64 { return parameters_; }

}  // namespace ttl::internal
