#pragma once

#include <span>
#include <vector>

#include <ttl/runtime/runtime.hpp>

#include "common/tensor_type.h"
#include "config/config.h"
#include "ir/model.h"
#include "planner/plan.h"

namespace zephyr::planner {

/**
 * Converts one logical Model into one rank-neutral WorkerPlan.
 */
class Planner final {
 public:
  /** Constructs a lower-only Planner. */
  Planner();

  Planner(const Planner &) = delete;
  auto operator=(const Planner &) -> Planner & = delete;
  Planner(Planner &&) noexcept = default;
  auto operator=(Planner &&) noexcept -> Planner & = default;
  ~Planner() = default;

  /**
   * Lowers the model into one WorkerPlan. Parallel expansion is intentionally not part of this stage.
   *
   * @param model logical typed SSA model
   * @param runtime Runtime supplying the device for the rank-neutral plan
   * @param config validated planner configuration
   * @param bindings dynamic dimension capacities used to materialize buffers
   * @return a vector containing the single lowered plan
   */
  [[nodiscard]] auto Plan(const ir::Model &model, const ttl::Runtime &runtime, const PlanConfig &config,
                          std::span<const DynamicDimensionBinding> bindings) const -> std::vector<WorkerPlan>;

 private:
  /** Lowers the model into a logical single-Worker plan. */
  [[nodiscard]] auto Lower(const ir::Model &model, ttl::Device device, const PlanConfig &config,
                           std::span<const DynamicDimensionBinding> bindings) const -> WorkerPlan;
};

}  // namespace zephyr::planner
