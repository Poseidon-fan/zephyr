#include "planner/planner.h"

#include "common/exception.h"

namespace zephyr::planner {

Planner::Planner() = default;

auto Planner::Plan(const ir::Model &model, const ttl::Runtime &runtime, const PlanConfig &config,
                   std::span<const DynamicDimensionBinding> bindings) const -> std::vector<WorkerPlan> {
  const auto template_plan = Lower(model, config, bindings);
  return Parallelize(template_plan, runtime, config);
}

auto Planner::Parallelize(const TemplatePlan & /*unused*/, const ttl::Runtime & /*unused*/,
                          const PlanConfig & /*unused*/) const -> std::vector<WorkerPlan> {
  throw NotImplementedException("parallelize todo");
}

}  // namespace zephyr::planner
