#include "bindings.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/types.hpp"
#include "engine/output.hpp"
#include "engine/request.hpp"
#include "sampler/sampler.hpp"

namespace zephyr::bindings {

namespace py = pybind11;

void BindTypes(py::module_ &module) {
  const sampler::SamplingParams sampling_defaults{};
  py::class_<sampler::SamplingParams>(
      module, "SamplingParams",
      "Effective sampling settings; logprobs are measured after penalties and temperature, before filtering.")
      .def(py::init([](double temperature, int64_t top_k, double top_p, double min_p, float frequency_penalty,
                       float presence_penalty, float repetition_penalty,
                       std::unordered_map<token_id_t, float> logits_bias,
                       std::optional<size_t> top_logprobs) -> sampler::SamplingParams {
             return sampler::SamplingParams{.temperature_ = temperature,
                                            .top_k_ = top_k,
                                            .top_p_ = top_p,
                                            .min_p_ = min_p,
                                            .frequency_penalty_ = frequency_penalty,
                                            .presence_penalty_ = presence_penalty,
                                            .repetition_penalty_ = repetition_penalty,
                                            .logits_bias_ = std::move(logits_bias),
                                            .top_logprobs_ = top_logprobs};
           }),
           py::kw_only(), py::arg("temperature") = sampling_defaults.temperature_,
           py::arg("top_k") = sampling_defaults.top_k_, py::arg("top_p") = sampling_defaults.top_p_,
           py::arg("min_p") = sampling_defaults.min_p_,
           py::arg("frequency_penalty") = sampling_defaults.frequency_penalty_,
           py::arg("presence_penalty") = sampling_defaults.presence_penalty_,
           py::arg("repetition_penalty") = sampling_defaults.repetition_penalty_,
           py::arg("logits_bias") = sampling_defaults.logits_bias_,
           py::arg("top_logprobs") = sampling_defaults.top_logprobs_)
      .def_readonly("temperature", &sampler::SamplingParams::temperature_)
      .def_readonly("top_k", &sampler::SamplingParams::top_k_)
      .def_readonly("top_p", &sampler::SamplingParams::top_p_)
      .def_readonly("min_p", &sampler::SamplingParams::min_p_)
      .def_readonly("frequency_penalty", &sampler::SamplingParams::frequency_penalty_)
      .def_readonly("presence_penalty", &sampler::SamplingParams::presence_penalty_)
      .def_readonly("repetition_penalty", &sampler::SamplingParams::repetition_penalty_)
      .def_readonly("logits_bias", &sampler::SamplingParams::logits_bias_)
      .def_readonly("top_logprobs", &sampler::SamplingParams::top_logprobs_);

  const engine::GenerationRequest generation_defaults{};
  py::class_<engine::GenerationRequest>(
      module, "GenerationRequest",
      "Tokenized generation input; max_new_tokens=None permits generation until a stop condition or the context limit.")
      .def(py::init([](std::vector<token_id_t> token_ids, sampler::SamplingParams sampling,
                       std::optional<size_t> max_new_tokens, std::vector<token_id_t> stop_token_ids,
                       std::optional<uint64_t> seed, size_t num_choices, bool ignore_eos) -> engine::GenerationRequest {
             return engine::GenerationRequest{.token_ids_ = std::move(token_ids),
                                              .sampling_ = std::move(sampling),
                                              .max_new_tokens_ = max_new_tokens,
                                              .stop_token_ids_ = std::move(stop_token_ids),
                                              .seed_ = seed,
                                              .num_choices_ = num_choices,
                                              .ignore_eos_ = ignore_eos};
           }),
           py::arg("token_ids"), py::kw_only(), py::arg("sampling") = generation_defaults.sampling_,
           py::arg("max_new_tokens") = generation_defaults.max_new_tokens_,
           py::arg("stop_token_ids") = generation_defaults.stop_token_ids_, py::arg("seed") = generation_defaults.seed_,
           py::arg("num_choices") = generation_defaults.num_choices_,
           py::arg("ignore_eos") = generation_defaults.ignore_eos_)
      .def_readonly("token_ids", &engine::GenerationRequest::token_ids_)
      .def_readonly("sampling", &engine::GenerationRequest::sampling_)
      .def_readonly("max_new_tokens", &engine::GenerationRequest::max_new_tokens_)
      .def_readonly("stop_token_ids", &engine::GenerationRequest::stop_token_ids_)
      .def_readonly("seed", &engine::GenerationRequest::seed_)
      .def_readonly("num_choices", &engine::GenerationRequest::num_choices_)
      .def_readonly("ignore_eos", &engine::GenerationRequest::ignore_eos_);

  py::class_<engine::EmbeddingRequest>(module, "EmbeddingRequest")
      .def(py::init([](std::vector<token_id_t> token_ids) -> engine::EmbeddingRequest {
             return engine::EmbeddingRequest{.token_ids_ = std::move(token_ids)};
           }),
           py::arg("token_ids"))
      .def_readonly("token_ids", &engine::EmbeddingRequest::token_ids_);

  py::enum_<engine::FinishReason>(module, "FinishReason")
      .value("EOS", engine::FinishReason::EOS)
      .value("STOP_TOKEN", engine::FinishReason::STOP_TOKEN)
      .value("LENGTH", engine::FinishReason::LENGTH)
      .value("MODEL_LENGTH", engine::FinishReason::MODEL_LENGTH)
      .value("CANCELED", engine::FinishReason::CANCELED)
      .value("ERROR", engine::FinishReason::ERROR);

  py::enum_<engine::RequestStatus>(module, "RequestStatus")
      .value("COMPLETED", engine::RequestStatus::COMPLETED)
      .value("CANCELED", engine::RequestStatus::CANCELED)
      .value("REJECTED", engine::RequestStatus::REJECTED)
      .value("ERROR", engine::RequestStatus::ERROR);

  py::class_<sampler::TokenLogprob>(module, "TokenLogprob")
      .def_readonly("token_id", &sampler::TokenLogprob::token_id_)
      .def_readonly("logprob", &sampler::TokenLogprob::logprob_);

  py::class_<sampler::SamplingLogprobs>(module, "SamplingLogprobs")
      .def_readonly("logprob", &sampler::SamplingLogprobs::logprob_)
      .def_readonly("top_logprobs", &sampler::SamplingLogprobs::top_logprobs_);

  py::class_<sampler::SamplingResult>(module, "SamplingResult")
      .def_readonly("token_id", &sampler::SamplingResult::token_id_)
      .def_readonly("logprobs", &sampler::SamplingResult::logprobs_);

  py::class_<engine::Usage>(module, "Usage")
      .def_readonly("prompt_tokens", &engine::Usage::prompt_tokens_)
      .def_readonly("completion_tokens", &engine::Usage::completion_tokens_);

  py::class_<engine::ChoiceOutput>(module, "ChoiceOutput",
                                   "Token increments for one choice, including any terminating EOS or stop token.")
      .def_readonly("index", &engine::ChoiceOutput::index_)
      .def_readonly("tokens", &engine::ChoiceOutput::tokens_)
      .def_readonly("finish_reason", &engine::ChoiceOutput::finish_reason_);

  py::class_<engine::RequestOutput>(
      module, "RequestOutput",
      "CPU-owned generation increments or a complete embedding; status is present only on the final request output.")
      .def_readonly("request_id", &engine::RequestOutput::request_id_)
      .def_readonly("result", &engine::RequestOutput::result_)
      .def_readonly("status", &engine::RequestOutput::status_)
      .def_readonly("usage", &engine::RequestOutput::usage_)
      .def_readonly("error_message", &engine::RequestOutput::error_message_);
}

}  // namespace zephyr::bindings
