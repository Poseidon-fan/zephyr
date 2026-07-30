#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <driver_types.h>

#include "ttl/device_properties.hpp"
#include "ttl/dtype.hpp"
#include "ttl/internal/index_width.hpp"
#include "ttl/shape.hpp"
#include "ttl/tensor.hpp"

namespace ttl::internal {

enum class ReductionOp : uint8_t {
  SUM,
  MEAN,
  MINIMUM,
  MAXIMUM,
  ARG_MIN,
  ARG_MAX,
  ANY,
  ALL,
};

enum class ReductionPath : uint8_t {
  WARP,
  BLOCK,
  TWO_STAGE,
};

template <typename Index>
struct ReductionParameters final {
  std::byte *output_;
  const std::byte *input_;
  Index output_shape_[TTL_MAX_RANK]{};
  Index output_input_strides_bytes_[TTL_MAX_RANK]{};
  Index output_strides_bytes_[TTL_MAX_RANK]{};
  Index reduction_shape_[TTL_MAX_RANK]{};
  Index reduction_strides_bytes_[TTL_MAX_RANK]{};
  Index output_count_;
  Index reduction_count_;
  uint32_t partial_count_;
  uint8_t output_rank_;
  uint8_t reduction_rank_;
  bool contiguous_reduction_;
};

using ReductionParameters32 = ReductionParameters<uint32_t>;
using ReductionParameters64 = ReductionParameters<uint64_t>;

static_assert(std::is_trivially_copyable_v<ReductionParameters32>);
static_assert(std::is_standard_layout_v<ReductionParameters32>);
static_assert(std::is_trivially_copyable_v<ReductionParameters64>);
static_assert(std::is_standard_layout_v<ReductionParameters64>);

/** Immutable lowering of one validated output/input reduction pair. */
class ReductionPlan final {
 public:
  [[nodiscard]] auto GetPath() const noexcept -> ReductionPath;
  [[nodiscard]] auto GetIndexWidth() const noexcept -> IndexWidth;
  [[nodiscard]] auto GetOutputCount() const noexcept -> uint64_t;
  [[nodiscard]] auto GetReductionCount() const noexcept -> uint64_t;
  [[nodiscard]] auto GetPartialCount() const noexcept -> uint32_t;
  [[nodiscard]] auto GetScratchBytes() const noexcept -> size_t;
  [[nodiscard]] auto GetLaunchBlockCount() const noexcept -> uint32_t;

  [[nodiscard]] auto MakeParameters32(std::source_location location = std::source_location::current()) const
      -> ReductionParameters32;
  [[nodiscard]] auto MakeParameters64() const noexcept -> ReductionParameters64;

 private:
  friend auto BuildReductionPlan(Tensor &output, const Tensor &input, std::span<const size_t> axes,
                                 ReductionOp operation, const DeviceProperties &properties,
                                 std::source_location location) -> ReductionPlan;

  ReductionParameters64 parameters_{};
  size_t scratch_bytes_{0};
  uint32_t launch_block_count_{0};
  ReductionPath path_{ReductionPath::BLOCK};
  IndexWidth index_width_{IndexWidth::UINT64};
};

/** Empty axes mean every input axis; otherwise normalize, sort, and reject duplicates. */
[[nodiscard]] auto ResolveReductionAxes(std::span<const int64_t> axes, size_t rank,
                                        std::source_location location = std::source_location::current())
    -> std::vector<size_t>;

[[nodiscard]] auto InferReductionShape(const Shape &input_shape, std::span<const size_t> axes, bool keep_dimensions,
                                       std::source_location location = std::source_location::current()) -> Shape;

[[nodiscard]] auto BuildReductionPlan(Tensor &output, const Tensor &input, std::span<const size_t> axes,
                                      ReductionOp operation, const DeviceProperties &properties,
                                      std::source_location location = std::source_location::current()) -> ReductionPlan;

void LaunchSummationReduction(cudaStream_t stream, DType dtype, ReductionOp operation, const ReductionPlan &plan,
                              void *scratch, std::source_location location);
void LaunchExtremaReduction(cudaStream_t stream, DType dtype, ReductionOp operation, const ReductionPlan &plan,
                            void *scratch, std::source_location location);
void LaunchLogicalReduction(cudaStream_t stream, ReductionOp operation, const ReductionPlan &plan, void *scratch,
                            std::source_location location);

[[nodiscard]] auto GetReductionName(ReductionOp operation) noexcept -> std::string_view;

}  // namespace ttl::internal
