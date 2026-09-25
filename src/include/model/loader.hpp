#pragma once

#include <filesystem>
#include <functional>
#include <memory>

#include <ttl/runtime/execution_context.hpp>

#include "parallel/tp.hpp"
#include "weight/weight_builder.hpp"

namespace zephyr::model {

namespace causal_lm {
class CausalLM;
}

/** Load a model interface on one rank using a shared checkpoint and that rank's TP context. */
template <typename Model>
using ModelLoader =
    std::function<std::unique_ptr<Model>(ttl::ExecutionContext &, const std::filesystem::path &,
                                         const weight::WeightBuilder &, const parallel::TpRankContext &)>;

/** Select a supported causal language model from its configuration. */
[[nodiscard]] auto LoadCausalLM(ttl::ExecutionContext &context, const std::filesystem::path &config_path,
                                const weight::WeightBuilder &builder, const parallel::TpRankContext &rank)
    -> std::unique_ptr<causal_lm::CausalLM>;

}  // namespace zephyr::model
