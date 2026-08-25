#include "planner/planner.h"

namespace zephyr::planner {

Planner::Planner() = default;

auto Planner::Plan(const ir::Model &model, const ttl::Runtime &runtime, const PlanConfig &config,
                   std::span<const DynamicDimensionBinding> bindings) const -> std::vector<WorkerPlan> {
  auto plans = std::vector<WorkerPlan>{};
  plans.emplace_back(Lower(model, runtime.GetDevices().front(), config, bindings));
  return plans;
}

}  // namespace zephyr::planner
