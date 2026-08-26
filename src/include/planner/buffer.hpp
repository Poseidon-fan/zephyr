#pragma once

#include <cstdint>
#include <vector>

#include <ttl/tensor/dtype.hpp>

#include "common/tensor_type.hpp"

namespace zephyr::planner {

/** Identifies one allocation in a plan. */
using buffer_id_t = uint32_t;

/** Identifies the lifetime and initialization owner of one planned buffer. */
enum class BufferKind : uint8_t {
  /** Per-invocation model input written by the Executor. */
  INPUT,

  /** Per-invocation model output retained for the caller. */
  OUTPUT,

  /** Checkpoint-backed storage initialized by the WeightLoader. */
  WEIGHT,

  /** Intermediate storage whose lifetime is bounded by one invocation. */
  ACTIVATION,
};

/** Describes one contiguous allocation in a plan. */
struct BufferSpec final {
  /** Stable identifier referenced by BufferView. */
  buffer_id_t id_;

  /** Lifetime and initialization category. */
  BufferKind kind_;

  /** Element type of the allocation. */
  ttl::DType dtype_;

  /** Concrete capacity shape; all dimensions are positive and static. */
  std::vector<int64_t> capacity_;
};

/** Describes a logical contiguous view into a planned buffer. */
struct BufferView final {
  /** Allocation containing the view. */
  buffer_id_t buffer_;

  /** Element offset from the beginning of the allocation. */
  int64_t element_offset_;

  /** Logical shape used by the instruction; dimensions may be dynamic symbols. */
  Shape shape_;
};

}  // namespace zephyr::planner
