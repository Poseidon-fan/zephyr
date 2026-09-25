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

auto LoadCausalLM(ttl::ExecutionContext &context, const std::filesystem::path &config_path,
                  const weight::WeightBuilder &builder, const parallel::TpRankContext &rank)
    -> std::unique_ptr<causal_lm::CausalLM> {
  struct Loader {
    std::string_view model_type_;
    std::string_view architecture_;
    auto (*load_)(ttl::ExecutionContext &, const std::filesystem::path &, const weight::WeightBuilder &,
                  const parallel::TpRankContext &) -> std::unique_ptr<causal_lm::CausalLM>;
  };
  static constexpr std::array loaders{
      Loader{.model_type_ = "qwen3",
             .architecture_ = "Qwen3ForCausalLM",
             .load_ = [](ttl::ExecutionContext &execution, const std::filesystem::path &path,
                         const weight::WeightBuilder &weights,
                         const parallel::TpRankContext &tp) -> std::unique_ptr<causal_lm::CausalLM> {
               const auto config = causal_lm::Qwen3Config::Load(path);
               return std::make_unique<causal_lm::Qwen3Model>(
                   causal_lm::Qwen3Model::Load(execution, config, weights, tp));
             }},
  };

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

    // Every supplied identifier must select the same model loader.
    for (const auto &loader : loaders) {
      const auto matches_type = has_model_type ? model_type == loader.model_type_ : !architectures.empty();
      const auto matches_architecture = architectures.empty() || architectures.front() == loader.architecture_;
      if (matches_type && matches_architecture) {
        return loader.load_(context, config_path, builder, rank);
      }
    }
  } catch (const nlohmann::json::exception &error) {
    throw ConfigurationException(config_path.string() + ": invalid model configuration: " + error.what());
  }

  throw ConfigurationException("unsupported causal language model configuration: " + config_path.string());
}

}  // namespace zephyr::model
