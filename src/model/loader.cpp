#include "model/loader.hpp"

#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/exception.hpp"
#include "model/causal_lm/qwen3.hpp"

namespace zephyr::model {

auto ResolveModelLoader(const std::filesystem::path &model_dir, std::optional<ModelTask> task) -> ResolvedModelLoader {
  struct Loader {
    std::string_view model_type_;
    std::string_view architecture_;
    ResolvedModelLoader load_;
  };
  static const std::array loaders{
      Loader{.model_type_ = "qwen3",
             .architecture_ = "Qwen3ForCausalLM",
             .load_ =
                 ModelLoader<causal_lm::CausalLM>{
                     [](ttl::ExecutionContext &execution, const std::filesystem::path &path,
                        const weight::WeightBuilder &weights,
                        const parallel::TpRankContext &tp) -> std::unique_ptr<causal_lm::CausalLM> {
                       const auto config = causal_lm::Qwen3Config::Load(path);
                       return std::make_unique<causal_lm::Qwen3Model>(
                           causal_lm::Qwen3Model::Load(execution, config, weights, tp));
                     }}},
  };

  const auto config_path = model_dir / "config.json";
  std::ifstream input(config_path);
  if (!input) {
    throw ConfigurationException("cannot open model configuration: " + config_path.string());
  }

  try {
    const auto config = nlohmann::json::parse(input);
    const auto has_model_type = config.contains("model_type");
    const auto model_type = config.value("model_type", std::string{});
    const auto architectures = config.value("architectures", std::vector<std::string>{});
    if (architectures.size() > 1) {
      throw ConfigurationException("model configuration must select a single architecture");
    }
    if (!task.has_value() && std::filesystem::exists(model_dir / "config_sentence_transformers.json")) {
      task = ModelTask::EMBEDDING;
    }

    // Every supplied identifier must select the same model loader.
    const Loader *selected = nullptr;
    for (const auto &loader : loaders) {
      const auto matches_type = has_model_type ? model_type == loader.model_type_ : !architectures.empty();
      const auto matches_architecture = architectures.empty() || architectures.front() == loader.architecture_;
      if (!matches_type || !matches_architecture) {
        continue;
      }
      const auto loader_task = std::holds_alternative<ModelLoader<causal_lm::CausalLM>>(loader.load_)
                                   ? ModelTask::GENERATION
                                   : ModelTask::EMBEDDING;
      if (task.has_value() && *task != loader_task) {
        continue;
      }
      if (selected != nullptr) {
        throw ConfigurationException("ambiguous model configuration; specify task and architecture: " +
                                     config_path.string());
      }
      selected = &loader;
    }
    if (selected != nullptr) {
      return selected->load_;
    }
  } catch (const nlohmann::json::exception &error) {
    throw ConfigurationException(config_path.string() + ": invalid model configuration: " + error.what());
  }

  std::string message = "unsupported model configuration";
  if (task.has_value()) {
    message += *task == ModelTask::EMBEDDING ? " for embedding" : " for generation";
  }
  throw ConfigurationException(message + ": " + config_path.string());
}

}  // namespace zephyr::model
