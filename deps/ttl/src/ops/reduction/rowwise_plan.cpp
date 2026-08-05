#include "ttl/internal/ops/rowwise.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <source_location>
#include <span>
#include <type_traits>

#include "ttl/common/error.hpp"
#include "ttl/internal/common/checked_math.hpp"
#include "ttl/internal/common/index_width.hpp"
#include "ttl/internal/tensor/tensor_impl.hpp"
#include "ttl/runtime/device_properties.hpp"
#include "ttl/tensor/dtype.hpp"
#include "ttl/tensor/shape.hpp"
#include "ttl/tensor/tensor.hpp"

namespace ttl::internal {
namespace {

constexpr uint64_t ROWWISE_THREADS_PER_BLOCK = 256;
constexpr uint64_t TARGET_VALUES_PER_PARTIAL = ROWWISE_THREADS_PER_BLOCK * 8;
constexpr uint32_t MAXIMUM_PARTIALS_PER_GROUP = 32;
constexpr uint64_t TARGET_BLOCKS_PER_MULTIPROCESSOR = 4;
constexpr uint64_t WARPS_PER_BLOCK = ROWWISE_THREADS_PER_BLOCK / 32;
constexpr uint64_t MAXIMUM_LAUNCH_BLOCKS = 65535;

[[nodiscard]] auto GetPartialCount(uint64_t group_count, uint64_t reduction_count, const DeviceProperties &properties,
                                   std::source_location location) -> uint32_t {
  if (group_count == 0 || reduction_count <= TARGET_VALUES_PER_PARTIAL) {
    return 1;
  }
  // Keep roughly eight values per thread in each partial, but add blocks only until the independent groups saturate the
  // target device occupancy. The cap keeps both scratch consumption and the final combine bounded.
  const auto useful_partials =
      (reduction_count / TARGET_VALUES_PER_PARTIAL) + (reduction_count % TARGET_VALUES_PER_PARTIAL != 0 ? 1 : 0);
  const auto target_blocks = CheckedMultiply(static_cast<uint64_t>(properties.multiprocessor_count_),
                                             TARGET_BLOCKS_PER_MULTIPROCESSOR, "row-wise target block count", location);
  const auto occupancy_partials = (target_blocks / group_count) + (target_blocks % group_count != 0 ? 1 : 0);
  return static_cast<uint32_t>(std::max(
      uint64_t{1}, std::min({useful_partials, occupancy_partials, static_cast<uint64_t>(MAXIMUM_PARTIALS_PER_GROUP)})));
}

[[nodiscard]] auto GetLaunchBlockCount(RowwisePath path, uint64_t group_count, uint32_t partial_count,
                                       const DeviceProperties &properties, std::source_location location) -> uint32_t {
  auto tasks = group_count;
  if (path == RowwisePath::WARP) {
    tasks = (group_count / WARPS_PER_BLOCK) + (group_count % WARPS_PER_BLOCK != 0 ? 1 : 0);
  } else if (path == RowwisePath::TWO_STAGE) {
    tasks = CheckedMultiply(group_count, static_cast<uint64_t>(partial_count), "row-wise task count", location);
  }
  const auto occupancy = CheckedMultiply(static_cast<uint64_t>(properties.multiprocessor_count_),
                                         TARGET_BLOCKS_PER_MULTIPROCESSOR, "row-wise occupancy block count", location);
  return static_cast<uint32_t>(
      std::min({std::max(uint64_t{1}, tasks), std::max(uint64_t{1}, occupancy), MAXIMUM_LAUNCH_BLOCKS}));
}

[[nodiscard]] auto IsContiguousReduction(const Shape &shape, const Strides &strides, std::span<const size_t> axes,
                                         size_t element_size, std::source_location location) -> bool {
  auto expected_stride_bytes = element_size;
  const auto dimensions = shape.GetDimensions();
  const auto stride_values = strides.GetValues();
  for (size_t remaining = axes.size(); remaining > 0; --remaining) {
    const auto axis = axes[remaining - 1];
    if (dimensions[axis] > 1 && CheckedBytes(stride_values[axis], element_size, location) != expected_stride_bytes) {
      return false;
    }
    const auto extent = dimensions[axis] == 0 ? size_t{1} : static_cast<size_t>(dimensions[axis]);
    expected_stride_bytes =
        CheckedMultiply(expected_stride_bytes, extent, "row-wise contiguous reduction byte stride", location);
  }
  return true;
}

[[nodiscard]] auto AddOffset(uint64_t current, uint64_t extent, uint64_t stride, uint64_t maximum) noexcept
    -> uint64_t {
  if (extent == 0 || stride == 0) {
    return current;
  }
  const auto coordinate = extent - 1;
  if (coordinate > (maximum - current) / stride) {
    return maximum + 1;
  }
  return current + (coordinate * stride);
}

[[nodiscard]] auto CanUse32BitIndexing(const RowwiseParameters64 &parameters) noexcept -> bool {
  constexpr auto maximum = uint64_t{std::numeric_limits<uint32_t>::max()};
  if (parameters.group_count_ > maximum || parameters.reduction_count_ > maximum) {
    return false;
  }

  auto input_group_offset = uint64_t{0};
  auto output_group_offset = uint64_t{0};
  for (size_t axis = 0; axis < parameters.group_rank_; ++axis) {
    if (parameters.group_shape_[axis] > maximum || parameters.group_input_strides_bytes_[axis] > maximum ||
        parameters.group_output_strides_bytes_[axis] > maximum) {
      return false;
    }
    input_group_offset = AddOffset(input_group_offset, parameters.group_shape_[axis],
                                   parameters.group_input_strides_bytes_[axis], maximum);
    output_group_offset = AddOffset(output_group_offset, parameters.group_shape_[axis],
                                    parameters.group_output_strides_bytes_[axis], maximum);
    if (input_group_offset > maximum || output_group_offset > maximum) {
      return false;
    }
  }

  auto input_reduction_offset = uint64_t{0};
  auto output_reduction_offset = uint64_t{0};
  for (size_t axis = 0; axis < parameters.reduction_rank_; ++axis) {
    if (parameters.reduction_shape_[axis] > maximum || parameters.reduction_input_strides_bytes_[axis] > maximum ||
        parameters.reduction_output_strides_bytes_[axis] > maximum) {
      return false;
    }
    input_reduction_offset = AddOffset(input_reduction_offset, parameters.reduction_shape_[axis],
                                       parameters.reduction_input_strides_bytes_[axis], maximum);
    output_reduction_offset = AddOffset(output_reduction_offset, parameters.reduction_shape_[axis],
                                        parameters.reduction_output_strides_bytes_[axis], maximum);
    if (input_reduction_offset > maximum || output_reduction_offset > maximum) {
      return false;
    }
  }
  return input_group_offset <= maximum - input_reduction_offset &&
         output_group_offset <= maximum - output_reduction_offset;
}

template <typename Destination>
void CopyParameters(const RowwiseParameters64 &source, Destination &destination, std::source_location location) {
  using Index = std::remove_cvref_t<decltype(destination.group_count_)>;
  destination.output_ = source.output_;
  destination.input_ = source.input_;
  destination.group_count_ = CheckedNarrow<Index>(source.group_count_, "row-wise group count", location);
  destination.reduction_count_ = CheckedNarrow<Index>(source.reduction_count_, "row-wise reduction count", location);
  destination.partial_count_ = source.partial_count_;
  destination.group_rank_ = source.group_rank_;
  destination.reduction_rank_ = source.reduction_rank_;
  destination.contiguous_input_reduction_ = source.contiguous_input_reduction_;
  destination.contiguous_output_reduction_ = source.contiguous_output_reduction_;
  for (size_t axis = 0; axis < TTL_MAX_RANK; ++axis) {
    destination.group_shape_[axis] = CheckedNarrow<Index>(source.group_shape_[axis], "row-wise group shape", location);
    destination.group_input_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.group_input_strides_bytes_[axis], "row-wise input group stride", location);
    destination.group_output_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.group_output_strides_bytes_[axis], "row-wise output group stride", location);
    destination.reduction_shape_[axis] =
        CheckedNarrow<Index>(source.reduction_shape_[axis], "row-wise reduction shape", location);
    destination.reduction_input_strides_bytes_[axis] =
        CheckedNarrow<Index>(source.reduction_input_strides_bytes_[axis], "row-wise input reduction stride", location);
    destination.reduction_output_strides_bytes_[axis] = CheckedNarrow<Index>(
        source.reduction_output_strides_bytes_[axis], "row-wise output reduction stride", location);
  }
}

}  // namespace

auto BuildRowwisePlan(Tensor &output, const Tensor &input, std::span<const size_t> axes, size_t accumulator_size,
                      const DeviceProperties &properties, std::source_location location) -> RowwisePlan {
  if (axes.empty() || accumulator_size == 0 || output.GetShape() != input.GetShape()) {
    throw InvalidArgumentError("row-wise plan requires axes, an accumulator, and matching shapes", location);
  }

  RowwisePlan plan;
  auto &parameters = plan.parameters_;
  parameters.output_ = static_cast<std::byte *>(TensorAccess::GetMutableData(output, location));
  parameters.input_ = static_cast<const std::byte *>(TensorAccess::GetData(input, location));
  const auto dimensions = input.GetShape().GetDimensions();
  const auto input_strides = input.GetStrides().GetValues();
  const auto output_strides = output.GetStrides().GetValues();
  const auto input_element_size = GetDTypeInfo(input.GetDType(), location).size_bytes_;
  const auto output_element_size = GetDTypeInfo(output.GetDType(), location).size_bytes_;

  std::array<bool, TTL_MAX_RANK> reduced{};
  for (const auto axis : axes) {
    if (axis >= input.GetRank() || reduced[axis]) {
      throw InvalidArgumentError("row-wise plan requires normalized, unique axes", location);
    }
    reduced[axis] = true;
  }

  auto group_count = uint64_t{1};
  auto reduction_count = uint64_t{1};
  for (size_t axis = 0; axis < input.GetRank(); ++axis) {
    const auto extent = static_cast<uint64_t>(dimensions[axis]);
    if (reduced[axis]) {
      const auto plan_axis = static_cast<size_t>(parameters.reduction_rank_);
      parameters.reduction_shape_[plan_axis] = extent;
      parameters.reduction_input_strides_bytes_[plan_axis] =
          CheckedBytes(input_strides[axis], input_element_size, location);
      parameters.reduction_output_strides_bytes_[plan_axis] =
          CheckedBytes(output_strides[axis], output_element_size, location);
      ++parameters.reduction_rank_;
      reduction_count = CheckedMultiply(reduction_count, extent, "row-wise reduction element count", location);
    } else {
      const auto plan_axis = static_cast<size_t>(parameters.group_rank_);
      parameters.group_shape_[plan_axis] = extent;
      parameters.group_input_strides_bytes_[plan_axis] =
          CheckedBytes(input_strides[axis], input_element_size, location);
      parameters.group_output_strides_bytes_[plan_axis] =
          CheckedBytes(output_strides[axis], output_element_size, location);
      ++parameters.group_rank_;
      group_count = CheckedMultiply(group_count, extent, "row-wise group count", location);
    }
  }
  parameters.group_count_ = group_count;
  parameters.reduction_count_ = reduction_count;
  parameters.contiguous_input_reduction_ =
      IsContiguousReduction(input.GetShape(), input.GetStrides(), axes, input_element_size, location);
  parameters.contiguous_output_reduction_ =
      IsContiguousReduction(output.GetShape(), output.GetStrides(), axes, output_element_size, location);

  parameters.partial_count_ = GetPartialCount(group_count, reduction_count, properties, location);
  // Small rows map one group to a warp. Larger rows map one group to a block unless cooperative partials are useful.
  if (reduction_count <= 32) {
    plan.path_ = RowwisePath::WARP;
  } else if (parameters.partial_count_ > 1) {
    plan.path_ = RowwisePath::TWO_STAGE;
  } else {
    plan.path_ = RowwisePath::BLOCK;
  }
  if (plan.path_ == RowwisePath::TWO_STAGE) {
    // Fused operators define their own accumulator type, so the caller supplies its exact scratch element size.
    const auto partials = CheckedMultiply(group_count, static_cast<uint64_t>(parameters.partial_count_),
                                          "row-wise partial count", location);
    plan.scratch_bytes_ = CheckedBytes(partials, accumulator_size, location);
  }
  plan.launch_block_count_ =
      GetLaunchBlockCount(plan.path_, group_count, parameters.partial_count_, properties, location);
  // Select uint32 only when counts and the complete reachable byte ranges of both tensors are representable.
  plan.index_width_ = CanUse32BitIndexing(parameters) ? IndexWidth::UINT32 : IndexWidth::UINT64;
  return plan;
}

auto RowwisePlan::GetPath() const noexcept -> RowwisePath { return path_; }
auto RowwisePlan::GetIndexWidth() const noexcept -> IndexWidth { return index_width_; }
auto RowwisePlan::GetGroupCount() const noexcept -> uint64_t { return parameters_.group_count_; }
auto RowwisePlan::GetReductionCount() const noexcept -> uint64_t { return parameters_.reduction_count_; }
auto RowwisePlan::GetPartialCount() const noexcept -> uint32_t { return parameters_.partial_count_; }
auto RowwisePlan::GetScratchBytes() const noexcept -> size_t { return scratch_bytes_; }
auto RowwisePlan::GetLaunchBlockCount() const noexcept -> uint32_t { return launch_block_count_; }

auto RowwisePlan::MakeParameters32(std::source_location location) const -> RowwiseParameters32 {
  if (index_width_ != IndexWidth::UINT32) {
    throw InternalError("cannot create 32-bit parameters for a 64-bit row-wise plan", location);
  }
  RowwiseParameters32 parameters{};
  CopyParameters(parameters_, parameters, location);
  return parameters;
}

auto RowwisePlan::MakeParameters64() const noexcept -> RowwiseParameters64 { return parameters_; }

}  // namespace ttl::internal
