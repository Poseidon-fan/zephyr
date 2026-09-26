#include "bindings.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <ttl/common/device.hpp>
#include <ttl/tensor/dtype.hpp>

#include "common/exception.hpp"
#include "common/types.hpp"
#include "engine/engine.hpp"
#include "executor/causal_lm.hpp"
#include "model/loader.hpp"
#include "scheduler/scheduler.hpp"

namespace zephyr::bindings {

namespace {

auto ParseDType(std::string_view name) -> ttl::DType {
  constexpr std::array supported{ttl::DType::FLOAT16, ttl::DType::BFLOAT16, ttl::DType::FLOAT32};
  for (const auto dtype : supported) {
    if (ttl::GetDTypeInfo(dtype).name_ == name) {
      return dtype;
    }
  }
  throw InvalidArgumentException("dtype must be float16, bfloat16, or float32");
}

auto GetDevices(const engine::EngineOptions &options) -> std::vector<int> {
  std::vector<int> devices;
  devices.reserve(options.executor_.devices_.size());
  for (const auto device : options.executor_.devices_) {
    devices.push_back(device.GetOrdinal());
  }
  return devices;
}

}  // namespace

void BindOptions(pybind11::module_ &module) {
  namespace py = pybind11;
  const engine::EngineOptions defaults;
  const auto paged_defaults = std::get<scheduler::PagedSchedulerConfig>(defaults.scheduler_);
  const executor::KvCacheOptions cache_defaults;

  py::enum_<model::ModelTask>(module, "ModelTask")
      .value("GENERATION", model::ModelTask::GENERATION)
      .value("EMBEDDING", model::ModelTask::EMBEDDING);

  py::class_<scheduler::PagedSchedulerConfig>(module, "PagedSchedulerConfig")
      .def(py::init([](size_t max_num_seqs, size_t max_num_batched_tokens, size_t max_prefill_chunk_tokens,
                       size_t max_decode_steps_before_prefill) {
             return scheduler::PagedSchedulerConfig{
                 .max_num_seqs_ = max_num_seqs,
                 .max_num_batched_tokens_ = max_num_batched_tokens,
                 .max_prefill_chunk_tokens_ = max_prefill_chunk_tokens,
                 .max_decode_steps_before_prefill_ = max_decode_steps_before_prefill};
           }),
           py::kw_only(), py::arg("max_num_seqs") = paged_defaults.max_num_seqs_,
           py::arg("max_num_batched_tokens") = paged_defaults.max_num_batched_tokens_,
           py::arg("max_prefill_chunk_tokens") = paged_defaults.max_prefill_chunk_tokens_,
           py::arg("max_decode_steps_before_prefill") = paged_defaults.max_decode_steps_before_prefill_)
      .def_readonly("max_num_seqs", &scheduler::PagedSchedulerConfig::max_num_seqs_)
      .def_readonly("max_num_batched_tokens", &scheduler::PagedSchedulerConfig::max_num_batched_tokens_)
      .def_readonly("max_prefill_chunk_tokens", &scheduler::PagedSchedulerConfig::max_prefill_chunk_tokens_)
      .def_readonly("max_decode_steps_before_prefill",
                    &scheduler::PagedSchedulerConfig::max_decode_steps_before_prefill_);

  py::class_<scheduler::LengthBucketSchedulerConfig>(module, "LengthBucketSchedulerConfig")
      .def(py::init([](size_t max_num_seqs) {
             return scheduler::LengthBucketSchedulerConfig{.max_num_seqs_ = max_num_seqs};
           }),
           py::kw_only(), py::arg("max_num_seqs") = paged_defaults.max_num_seqs_)
      .def_readonly("max_num_seqs", &scheduler::LengthBucketSchedulerConfig::max_num_seqs_);

  py::class_<executor::KvCacheOptions>(
      module, "KvCacheOptions", "KV cache settings; memory_bytes=None automatically estimates the per-device budget.")
      .def(py::init([](size_t block_size, std::optional<size_t> memory_bytes) {
             return executor::KvCacheOptions{.block_size_ = block_size, .memory_bytes_ = memory_bytes};
           }),
           py::kw_only(), py::arg("block_size") = cache_defaults.block_size_,
           py::arg("memory_bytes") = cache_defaults.memory_bytes_)
      .def_readonly("block_size", &executor::KvCacheOptions::block_size_)
      .def_readonly("memory_bytes", &executor::KvCacheOptions::memory_bytes_);

  py::class_<engine::EngineOptions>(
      module, "EngineOptions",
      "Devices define tensor-parallel rank order; max_seq_len=0 inherits the model context limit.\n"
      "Constructing Engine profiles the selected GPUs.")
      .def(py::init([](std::filesystem::path model_dir, const std::vector<int> &devices, std::string_view dtype,
                       int64_t max_seq_len, double gpu_memory_utilization, scheduler::SchedulerConfig scheduler,
                       size_t max_outstanding_sequences, size_t max_buffered_output_bytes,
                       std::vector<token_id_t> eos_token_ids) {
             engine::EngineOptions options;
             options.executor_.model_dir_ = std::move(model_dir);
             options.executor_.devices_.clear();
             options.executor_.devices_.reserve(devices.size());
             for (const auto device : devices) {
               options.executor_.devices_.emplace_back(device);
             }
             options.executor_.dtype_ = ParseDType(dtype);
             options.executor_.execution_limits_.max_seq_len_ = max_seq_len;
             options.executor_.gpu_memory_utilization_ = gpu_memory_utilization;
             options.scheduler_ = scheduler;
             options.max_outstanding_sequences_ = max_outstanding_sequences;
             options.max_buffered_output_bytes_ = max_buffered_output_bytes;
             options.eos_token_ids_ = std::move(eos_token_ids);
             return options;
           }),
           py::arg("model_dir"), py::kw_only(), py::arg("devices") = GetDevices(defaults),
           py::arg("dtype") = std::string{ttl::GetDTypeInfo(defaults.executor_.dtype_).name_},
           py::arg("max_seq_len") = defaults.executor_.execution_limits_.max_seq_len_,
           py::arg("gpu_memory_utilization") = defaults.executor_.gpu_memory_utilization_,
           py::arg("scheduler") = defaults.scheduler_,
           py::arg("max_outstanding_sequences") = defaults.max_outstanding_sequences_,
           py::arg("max_buffered_output_bytes") = defaults.max_buffered_output_bytes_,
           py::arg("eos_token_ids") = defaults.eos_token_ids_)
      .def_property_readonly("model_dir",
                             [](const engine::EngineOptions &options) { return options.executor_.model_dir_.string(); })
      .def_property_readonly("devices", &GetDevices)
      .def_property_readonly(
          "dtype",
          [](const engine::EngineOptions &options) { return ttl::GetDTypeInfo(options.executor_.dtype_).name_; })
      .def_property_readonly(
          "max_seq_len",
          [](const engine::EngineOptions &options) { return options.executor_.execution_limits_.max_seq_len_; })
      .def_property_readonly(
          "gpu_memory_utilization",
          [](const engine::EngineOptions &options) { return options.executor_.gpu_memory_utilization_; })
      .def_property_readonly("scheduler", [](const engine::EngineOptions &options) { return options.scheduler_; })
      .def_readonly("max_outstanding_sequences", &engine::EngineOptions::max_outstanding_sequences_)
      .def_readonly("max_buffered_output_bytes", &engine::EngineOptions::max_buffered_output_bytes_)
      .def_readonly("eos_token_ids", &engine::EngineOptions::eos_token_ids_);
}

}  // namespace zephyr::bindings
