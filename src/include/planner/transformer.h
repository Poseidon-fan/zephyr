#pragma once

#include <ttl/runtime/runtime.hpp>

#include "planner/plan.h"

namespace zephyr::planner {

/** Rewrites a WorkerPlan while preserving its model semantics. */
class Transformer {
 public:
  virtual ~Transformer() = default;

  /** Applies this pass to the supplied single-Worker template plan. */
  virtual void Transform(WorkerPlan *plan, const ttl::Runtime &runtime) const = 0;
};

#define DEFINE_TRANSFORMER(transformer_name)                                      \
  class transformer_name final : public Transformer {                             \
   public:                                                                        \
    void Transform(WorkerPlan *plan, const ttl::Runtime &runtime) const override; \
  }

/** Fuses adjacent compatible Linear instructions, such as QKV and gate-up projections. */
DEFINE_TRANSFORMER(LinearFusion);

}  // namespace zephyr::planner
