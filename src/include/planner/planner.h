#pragma once

#include <span>
#include <vector>

#include <ttl/runtime/runtime.hpp>

#include "common/tensor_type.h"
#include "config/config.h"
#include "ir/model.h"
#include "planner/plan.h"

namespace zephyr::planner {

/** Converts one logical Model into one or more rank-local WorkerPlans. */
class Planner final {
 public:
  /** Constructs a Planner. */
  Planner();

  Planner(const Planner &) = delete;
  auto operator=(const Planner &) -> Planner & = delete;
  Planner(Planner &&) noexcept = default;
  auto operator=(Planner &&) noexcept -> Planner & = default;
  ~Planner() = default;

  /**
   * Lowers the model into a rank-neutral template and expands it into WorkerPlans.
   *
   * @param model logical typed SSA model
   * @param runtime Runtime supplying devices for the rank-local plans
   * @param config validated planner configuration
   * @param bindings dynamic dimension capacities used to materialize buffers
   * @return rank-local WorkerPlans
   */
  [[nodiscard]] auto Plan(const ir::Model &model, const ttl::Runtime &runtime, const PlanConfig &config,
                          std::span<const DynamicDimensionBinding> bindings) const -> std::vector<WorkerPlan>;

 private:
  /** Lowers the model into a rank-neutral serial template. */
  [[nodiscard]] auto Lower(const ir::Model &model, const PlanConfig &config,
                           std::span<const DynamicDimensionBinding> bindings) const -> TemplatePlan;

  /** Expands a template into rank-local WorkerPlans. */
  [[nodiscard]] auto Parallelize(const TemplatePlan &plan, const ttl::Runtime &runtime,
                                 const PlanConfig &config) const -> std::vector<WorkerPlan>;
};

}  // namespace zephyr::planner
