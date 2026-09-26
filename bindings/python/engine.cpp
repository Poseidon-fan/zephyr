#include "bindings.hpp"

#include <exception>
#include <optional>
#include <utility>

#include <pybind11/stl.h>

#include "common/logger.hpp"
#include "engine/engine.hpp"
#include "executor/causal_lm.hpp"

namespace py = pybind11;

namespace zephyr::bindings {

void BindEngine(py::module_ &module) {
  py::class_<engine::EngineInfo>(module, "EngineInfo", "Effective capabilities of the loaded model.")
      .def_readonly("task", &engine::EngineInfo::task_)
      .def_readonly("max_seq_len", &engine::EngineInfo::max_seq_len_)
      .def_readonly("vocab_size", &engine::EngineInfo::vocab_size_);

  py::class_<engine::Engine>(
      module, "Engine", py::release_gil_before_calling_cpp_dtor(),
      "Native engine owning its runtime and worker threads. Exactly one caller consumes outputs.")
      .def(py::init([](engine::EngineOptions options, std::optional<model::ModelTask> task,
                       std::optional<executor::KvCacheOptions> kv_cache) {
             // Copy Python-owned configuration before releasing the GIL; the workers retain only C++ values.
             py::gil_scoped_release release;
             return engine::Engine::Create(std::move(options), task, {.kv_cache_ = kv_cache});
           }),
           py::arg("options"), py::kw_only(), py::arg("task") = py::none(),
           py::arg("kv_cache") = executor::KvCacheOptions{},
           "Resolve a supported model and its task from the checkpoint; task=None selects automatically. "
           "kv_cache applies to generation models only; None disables KV storage.")
      .def(
          "submit",
          [](engine::Engine &instance, engine::Request request) {
            // The request is a value snapshot, independent of the Python object's lifetime.
            py::gil_scoped_release release;
            return instance.Submit(std::move(request));
          },
          py::arg("request"),
          "Validate and enqueue a request, returning its ID. Admission overload raises OverloadedError.")
      .def("cancel", &engine::Engine::Cancel, py::arg("request_id"), py::call_guard<py::gil_scoped_release>(),
           "Cancel at the next safe execution boundary. Unknown or finished IDs are harmless.")
      .def("stop_choice", &engine::Engine::StopChoice, py::arg("request_id"), py::arg("choice_index"),
           py::call_guard<py::gil_scoped_release>(),
           "Normally finish one choice after a text stop, leaving other choices running. "
           "Unknown or finished choices are harmless; already published outputs remain unchanged.")
      .def_property_readonly("info", &engine::Engine::GetInfo)
      .def("wait_for_outputs", &engine::Engine::WaitForOutputs, py::call_guard<py::gil_scoped_release>(),
           "Block for CPU-owned increments or final results. Single consumer only; an empty list means shutdown. "
           "Engine failures raise after queued outputs have been drained.")
      .def("close", &engine::Engine::Close, py::call_guard<py::gil_scoped_release>(),
           "Stop admission, finish in-flight work, cancel remaining requests, and release GPU resources. Safe to "
           "repeat.")
      .def(
          "__enter__", [](engine::Engine &instance) -> engine::Engine & { return instance; },
          py::return_value_policy::reference_internal)
      .def("__exit__",
           [](engine::Engine &instance, py::handle exception_type, py::handle /*exception*/, py::handle /*traceback*/) {
             const bool unwinding = !exception_type.is_none();
             py::gil_scoped_release release;
             try {
               instance.Close();
             } catch (const std::exception &error) {
               if (!unwinding) {
                 throw;
               }
               ZEPHYR_LOG_ERROR("engine shutdown failed while propagating a Python exception: {}", error.what());
             } catch (...) {
               if (!unwinding) {
                 throw;
               }
               ZEPHYR_LOG_ERROR("engine shutdown failed with an unknown error while propagating a Python exception");
             }
             return false;
           });
}

}  // namespace zephyr::bindings
