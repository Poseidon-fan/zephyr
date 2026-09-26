#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <variant>

#include <ttl/runtime/execution_context.hpp>

#include "parallel/tp.hpp"
#include "weight/weight_builder.hpp"

namespace zephyr::model {

namespace causal_lm {
class CausalLM;
}

namespace embedding {
class Embedding;
}

enum class ModelTask : uint8_t { GENERATION, EMBEDDING };

/** Load a model interface on one rank using a shared checkpoint and that rank's TP context. */
template <typename Model>
using ModelLoader =
    std::function<std::unique_ptr<Model>(ttl::ExecutionContext &, const std::filesystem::path &,
                                         const weight::WeightBuilder &, const parallel::TpRankContext &)>;

using ResolvedModelLoader = std::variant<ModelLoader<causal_lm::CausalLM>, ModelLoader<embedding::Embedding>>;

/**
 * Select a built-in model loader before loading rank-local weights. An explicit task overrides automatic detection;
 * otherwise sentence-transformers metadata selects embedding. The returned loader supports concurrent rank calls.
 */
[[nodiscard]] auto ResolveModelLoader(const std::filesystem::path &model_dir,
                                      std::optional<ModelTask> task = std::nullopt) -> ResolvedModelLoader;

}  // namespace zephyr::model
