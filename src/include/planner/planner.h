#pragma once

#include <memory>
#include <vector>

#include <ttl/runtime/runtime.hpp>

#include "config/config.h"
#include "ir/model.h"
#include "planner/plan.h"
#include "planner/transformer.h"

namespace zephyr::planner {

/**
 * Converts one logical Model into one static WorkerPlan per configured rank.
 *
 * Planning is deliberately split into single-Worker lowering, an ordered Transformer chain, and rank expansion. The
 * Planner only reads Runtime metadata; ownership of CUDA resources remains with Runtime and the later Executor.
 */
class Planner final {
 public:
  /** Constructs a Planner with Zephyr's built-in Transformer chain. */
  Planner();

  /** Constructs a Planner with an explicitly supplied Transformer chain. */
  explicit Planner(std::vector<std::unique_ptr<Transformer>> transformers);

  Planner(const Planner &) = delete;
  auto operator=(const Planner &) -> Planner & = delete;
  Planner(Planner &&) noexcept = default;
  auto operator=(Planner &&) noexcept -> Planner & = default;
  ~Planner() = default;

  /** Appends an extension pass after the currently registered passes. */
  void AddTransformer(std::unique_ptr<Transformer> transformer);

  /**
   * Builds one executable plan for every Runtime device.
   *
   * @param model logical typed SSA model
   * @param runtime Runtime whose device order defines global rank order
   * @param config parallelism, capacities, and KV planning options
   * @param runner_mode task contract selected by the Engine
   * @return plans in global rank order
   */
  [[nodiscard]] auto Plan(const ir::Model &model, const ttl::Runtime &runtime, const PlanConfig &config,
                          RunnerMode runner_mode) const -> std::vector<WorkerPlan>;

 private:
  /** Lowers the model into a single-Worker template plan. */
  [[nodiscard]] auto Lower(const ir::Model &model, const PlanConfig &config, RunnerMode runner_mode) const
      -> WorkerPlan;

  /** Expands a transformed single-Worker template into one WorkerPlan per configured rank. */
  [[nodiscard]] auto Expand(WorkerPlan plan, const PlanConfig &config, const ttl::Runtime &runtime) const
      -> std::vector<WorkerPlan>;

  /** Ordered lowering/optimization extension chain. */
  std::vector<std::unique_ptr<Transformer>> transformers_;
};

}  // namespace zephyr::planner
