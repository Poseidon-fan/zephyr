#include "model/causal_lm/qwen3.hpp"

#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "common/exception.hpp"

namespace zephyr::model::causal_lm {

auto Qwen3Config::Load(const std::filesystem::path &config_path) -> Qwen3Config {
  std::ifstream input(config_path);
  if (!input) {
    throw ConfigurationException("cannot open Qwen3 configuration: " + config_path.string());
  }

  try {
    const auto config = nlohmann::json::parse(input);
    if (config.value("model_type", std::string{"qwen3"}) != "qwen3") {
      throw ConfigurationException("Qwen3Config requires a dense Qwen3 model");
    }
    if (config.contains("quantization_config") && !config.at("quantization_config").is_null()) {
      throw ConfigurationException("Qwen3 quantized checkpoints are unsupported");
    }
    if (config.at("hidden_act").get<std::string>() != "silu" || config.value("attention_bias", false)) {
      throw ConfigurationException("Qwen3 requires SiLU activation and bias-free attention projections");
    }

    // JSON numeric conversions otherwise silently truncate fractional or out-of-range dimensions.
    const auto read_dimension = [&](std::string_view name) -> int64_t {
      const auto &value = config.at(name);
      if (!value.is_number_integer() ||
          (value.is_number_unsigned() &&
           value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
        throw ConfigurationException("Qwen3 configuration field '" + std::string{name} + "' must be an int64 integer");
      }
      return value.get<int64_t>();
    };

    Qwen3Config result;
    result.vocab_size_ = read_dimension("vocab_size");
    result.hidden_size_ = read_dimension("hidden_size");
    result.intermediate_size_ = read_dimension("intermediate_size");
    result.num_hidden_layers_ = read_dimension("num_hidden_layers");
    result.num_attention_heads_ = read_dimension("num_attention_heads");
    result.num_key_value_heads_ = read_dimension("num_key_value_heads");
    result.max_position_embeddings_ = read_dimension("max_position_embeddings");
    if (config.contains("head_dim") && !config.at("head_dim").is_null()) {
      result.head_dim_ = read_dimension("head_dim");
    } else if (result.num_attention_heads_ > 0) {
      result.head_dim_ = result.hidden_size_ / result.num_attention_heads_;
    }
    result.rms_norm_eps_ = config.at("rms_norm_eps").get<double>();
    result.rope_theta_ = config.value("rope_theta", result.rope_theta_);
    result.tie_word_embeddings_ = config.value("tie_word_embeddings", false);

    // A configured window has no effect until enabled on at least one decoder layer.
    if (config.value("use_sliding_window", false) && config.contains("sliding_window") &&
        !config.at("sliding_window").is_null()) {
      const auto first_window_layer = config.contains("max_window_layers") ? read_dimension("max_window_layers") : 0;
      if (first_window_layer < result.num_hidden_layers_) {
        throw ConfigurationException("Qwen3 sliding-window attention is unsupported");
      }
    }
    if (config.contains("layer_types") && !config.at("layer_types").is_null()) {
      const auto &layer_types = config.at("layer_types");
      if (!layer_types.is_array() || layer_types.size() != static_cast<uint64_t>(result.num_hidden_layers_)) {
        throw ConfigurationException("Qwen3 layer_types must describe every decoder layer");
      }
      for (const auto &layer_type : layer_types) {
        if (layer_type != "full_attention") {
          throw ConfigurationException("Qwen3 supports only full-attention decoder layers");
        }
      }
    }

    // Both checkpoint schemas can describe ordinary RoPE; scaled variants require a different implementation.
    for (const auto *field : {"rope_scaling", "rope_parameters"}) {
      if (!config.contains(field) || config.at(field).is_null()) {
        continue;
      }
      const auto &rope = config.at(field);
      if (!rope.is_object()) {
        throw ConfigurationException("Qwen3 rotary parameters must be a JSON object");
      }
      const auto rope_type =
          rope.contains("rope_type") ? rope.at("rope_type").get<std::string>() : rope.at("type").get<std::string>();
      if (rope_type != "default" || rope.value("partial_rotary_factor", 1.0) != 1.0) {
        throw ConfigurationException("Qwen3 supports only full-head, unscaled rotary embeddings");
      }
      result.rope_theta_ = rope.value("rope_theta", result.rope_theta_);
    }
    return result;
  } catch (const nlohmann::json::exception &error) {
    throw ConfigurationException(config_path.string() + ": invalid Qwen3 configuration: " + error.what());
  }
}

}  // namespace zephyr::model::causal_lm
