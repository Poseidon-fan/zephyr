#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <type_traits>

#include "ttl/device_properties.hpp"
#include "ttl/internal/index_width.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl::internal {

enum class RowwisePath : uint8_t {
  WARP,
  BLOCK,
  TWO_STAGE,
};

template <typename Index>
struct RowwiseParameters final {
  std::byte *output_;
  const std::byte *input_;
  Index group_shape_[TTL_MAX_RANK]{};
  Index group_input_strides_bytes_[TTL_MAX_RANK]{};
  Index group_output_strides_bytes_[TTL_MAX_RANK]{};
  Index reduction_shape_[TTL_MAX_RANK]{};
  Index reduction_input_strides_bytes_[TTL_MAX_RANK]{};
  Index reduction_output_strides_bytes_[TTL_MAX_RANK]{};
  Index group_count_;
  Index reduction_count_;
  uint32_t partial_count_;
  uint8_t group_rank_;
  uint8_t reduction_rank_;
  bool contiguous_input_reduction_;
  bool contiguous_output_reduction_;
};

using RowwiseParameters32 = RowwiseParameters<uint32_t>;
using RowwiseParameters64 = RowwiseParameters<uint64_t>;

static_assert(std::is_trivially_copyable_v<RowwiseParameters32>);
static_assert(std::is_standard_layout_v<RowwiseParameters32>);
static_assert(std::is_trivially_copyable_v<RowwiseParameters64>);
static_assert(std::is_standard_layout_v<RowwiseParameters64>);

/** Immutable layout and launch lowering shared by fused row-wise operators. */
class RowwisePlan final {
 public:
  [[nodiscard]] auto GetPath() const noexcept -> RowwisePath;
  [[nodiscard]] auto GetIndexWidth() const noexcept -> IndexWidth;
  [[nodiscard]] auto GetGroupCount() const noexcept -> uint64_t;
  [[nodiscard]] auto GetReductionCount() const noexcept -> uint64_t;
  [[nodiscard]] auto GetPartialCount() const noexcept -> uint32_t;
  [[nodiscard]] auto GetScratchBytes() const noexcept -> size_t;
  [[nodiscard]] auto GetLaunchBlockCount() const noexcept -> uint32_t;

  [[nodiscard]] auto MakeParameters32(std::source_location location = std::source_location::current()) const
      -> RowwiseParameters32;
  [[nodiscard]] auto MakeParameters64() const noexcept -> RowwiseParameters64;

 private:
  friend auto BuildRowwisePlan(Tensor &output, const Tensor &input, std::span<const size_t> axes,
                               size_t accumulator_size, const DeviceProperties &properties,
                               std::source_location location) -> RowwisePlan;

  RowwiseParameters64 parameters_{};
  size_t scratch_bytes_{0};
  uint32_t launch_block_count_{0};
  RowwisePath path_{RowwisePath::BLOCK};
  IndexWidth index_width_{IndexWidth::UINT64};
};

[[nodiscard]] auto BuildRowwisePlan(Tensor &output, const Tensor &input, std::span<const size_t> axes,
                                    size_t accumulator_size, const DeviceProperties &properties,
                                    std::source_location location = std::source_location::current()) -> RowwisePlan;

}  // namespace ttl::internal
