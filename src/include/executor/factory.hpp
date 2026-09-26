#pragma once

#include "executor/causal_lm.hpp"
#include "executor/execution.hpp"
#include "model/loader.hpp"

namespace zephyr::executor {

/** Pair a resolved model loader with its execution category; cache options apply only to causal models. */
[[nodiscard]] auto CreateExecutionFactory(model::ResolvedModelLoader loader, CausalLMOptions options = {})
    -> ExecutionFactory;

}  // namespace zephyr::executor
