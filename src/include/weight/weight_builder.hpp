#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include <ttl/runtime/execution_context.hpp>
#include <ttl/runtime/runtime.hpp>
#include <ttl/tensor/dtype.hpp>
#include <ttl/tensor/shape.hpp>
#include <ttl/tensor/tensor.hpp>

#include "weight/checkpoint.hpp"

namespace zephyr::weight {

/** Describe either an equal split or an explicit range along one tensor axis. */
class Shard {
 public:
  /** Split an axis into world_size equal pieces and select rank's piece. */
  [[nodiscard]] static auto Uniform(size_t axis, size_t rank, size_t world_size) -> Shard;
  /** Select [offset, offset + length) along one axis. */
  [[nodiscard]] static auto Range(size_t axis, size_t offset, size_t length) -> Shard;

  Shard(const Shard &) = default;
  auto operator=(const Shard &) -> Shard & = default;
  Shard(Shard &&) noexcept = default;
  auto operator=(Shard &&) noexcept -> Shard & = default;

 private:
  friend class WeightBuilder;

  /** Keep the two user-visible shard forms distinct until a parameter shape is available. */
  struct UniformSpec {
    /** Axis to partition. */
    size_t axis_;
    /** Selected rank. */
    size_t rank_;
    /** Number of equal partitions. */
    size_t world_size_;
  };

  struct RangeSpec {
    /** Axis to partition. */
    size_t axis_;
    /** First selected element along the axis. */
    size_t offset_;
    /** Number of selected elements. */
    size_t length_;
  };

  explicit Shard(UniformSpec spec) noexcept : spec_(spec) {}
  explicit Shard(RangeSpec spec) noexcept : spec_(spec) {}

  /** Deferred shard description; the tensor shape is needed to resolve it. */
  std::variant<UniformSpec, RangeSpec> spec_;
};

/**
 * Resolve checkpoint names and asynchronously materialize their weights on a
 * TTL execution context.
 *
 * A builder is a cheap immutable view: PushPrefix and WithDType return copies
 * that share the checkpoint mapping and runtime references. The runtime and
 * checkpoint must outlive every builder copy and every in-flight transfer;
 * contexts passed to Get must be created by that runtime.
 */
class WeightBuilder {
 public:
  WeightBuilder(const Checkpoint &checkpoint, ttl::Runtime &runtime, ttl::DType target_dtype);

  WeightBuilder(const WeightBuilder &) = default;
  auto operator=(const WeightBuilder &) -> WeightBuilder & = delete;
  WeightBuilder(WeightBuilder &&) noexcept = default;
  auto operator=(WeightBuilder &&) noexcept -> WeightBuilder & = delete;

  /** Return a view whose names are prefixed with component followed by '.'. */
  [[nodiscard]] auto PushPrefix(std::string_view component) const -> WeightBuilder;
  /** Numeric convenience form used for layer and expert indices. */
  [[nodiscard]] auto PushPrefix(size_t component) const -> WeightBuilder;
  /** Return a view that converts loaded tensors to target_dtype. */
  [[nodiscard]] auto WithDType(ttl::DType target_dtype) const -> WeightBuilder;

  /** Return metadata for a name under this builder's current prefix. */
  [[nodiscard]] auto GetParameterInfo(std::string_view parameter_name) const -> std::optional<ParameterInfo>;

  /**
   * Load one parameter, optionally selecting a row-major shard, and enqueue its
   * host-to-device transfer on context. expected_shape is checked before any
   * device allocation so model wiring errors fail early.
   */
  [[nodiscard]] auto Get(ttl::ExecutionContext &context, const ttl::Shape &expected_shape,
                         std::string_view parameter_name, std::optional<Shard> shard = std::nullopt) const
      -> ttl::Tensor;

 private:
  struct ResolvedShard {
    /** Axis to partition. */
    size_t axis_;
    /** First selected element along the axis. */
    size_t start_;
    /** Number of selected elements. */
    size_t length_;
  };

  [[nodiscard]] static auto ResolveShard(const ttl::Shape &shape, const std::optional<Shard> &shard)
      -> std::optional<ResolvedShard>;
  [[nodiscard]] auto FullName(std::string_view parameter_name) const -> std::string;

  /** Checkpoint whose mappings back this builder. */
  const Checkpoint &checkpoint_;
  /** Runtime used for pinned staging allocations. */
  ttl::Runtime &runtime_;
  /** Fully qualified name prefix, including a trailing '.'. */
  std::string prefix_;
  /** Dtype returned after the source bytes are materialized. */
  ttl::DType target_dtype_;
};

}  // namespace zephyr::weight
