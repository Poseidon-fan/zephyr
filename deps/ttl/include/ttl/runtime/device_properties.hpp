#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "ttl/common/device.hpp"

namespace ttl {

/** NVIDIA compute capability represented as its major and minor components. */
struct ComputeCapability final {
  int32_t major_;
  int32_t minor_;

  /** Return the conventional SM number, for example 80 for compute capability 8.0. */
  [[nodiscard]] constexpr auto GetSmVersion() const noexcept -> int64_t {
    return (static_cast<int64_t>(major_) * 10) + minor_;
  }

  [[nodiscard]] constexpr auto operator<=>(const ComputeCapability &) const noexcept -> std::strong_ordering = default;
};

/**
 * CUDA device metadata used by launch validation, architecture dispatch, stream creation, and memory management.
 *
 * Runtime queries one validated instance per configured device. Pairwise properties such as peer access are
 * deliberately not stored here.
 */
struct DeviceProperties final {
  Device device_;
  std::string name_;
  ComputeCapability compute_capability_;
  int32_t multiprocessor_count_;
  size_t max_dynamic_shared_memory_per_block_bytes_;
  bool supports_cluster_launch_;

  [[nodiscard]] auto operator==(const DeviceProperties &) const noexcept -> bool = default;
};

static_assert(std::is_trivially_copyable_v<ComputeCapability>);
static_assert(std::is_standard_layout_v<ComputeCapability>);

}  // namespace ttl
