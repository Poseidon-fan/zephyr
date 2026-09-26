#include "bindings.hpp"

#include <exception>
#include <utility>

#include <pybind11/pybind11.h>

#include <ttl/common/error.hpp>

#include "common/exception.hpp"
#include "common/logger.hpp"

namespace py = pybind11;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
PYBIND11_MODULE(_C, module) {
  module.doc() = "Zephyr native bindings";

  py::enum_<zephyr::LogLevel>(module, "LogLevel")
      .value("TRACE", zephyr::LogLevel::TRACE)
      .value("DEBUG", zephyr::LogLevel::DEBUG)
      .value("INFO", zephyr::LogLevel::INFO)
      .value("WARN", zephyr::LogLevel::WARN)
      .value("ERROR", zephyr::LogLevel::ERROR)
      .value("OFF", zephyr::LogLevel::OFF);
  module.def("set_log_level", &zephyr::SetLogLevel, py::arg("level"),
             "Set the process-wide logging threshold for the native engine.");

  py::register_local_exception<zephyr::OverloadedException>(module, "OverloadedError", PyExc_RuntimeError);
  // Keep translation local to this extension so other TTL users retain their own exception policies.
  py::register_local_exception_translator([](std::exception_ptr exception) {
    if (exception == nullptr) {
      return;
    }
    try {
      std::rethrow_exception(std::move(exception));
    } catch (const zephyr::InvalidArgumentException &error) {
      py::set_error(PyExc_ValueError, error.what());
    } catch (const zephyr::ConfigurationException &error) {
      py::set_error(PyExc_ValueError, error.what());
    } catch (const zephyr::NotImplementedException &error) {
      py::set_error(PyExc_NotImplementedError, error.what());
    } catch (const zephyr::OutOfMemoryException &error) {
      py::set_error(PyExc_MemoryError, error.what());
    } catch (const ttl::Error &error) {
      PyObject *type = PyExc_RuntimeError;
      switch (error.GetCode()) {
        case ttl::ErrorCode::INVALID_ARGUMENT:
          type = PyExc_ValueError;
          break;
        case ttl::ErrorCode::OVERFLOW:
          type = PyExc_OverflowError;
          break;
        case ttl::ErrorCode::NOT_SUPPORTED:
          type = PyExc_NotImplementedError;
          break;
        case ttl::ErrorCode::OUT_OF_MEMORY:
          type = PyExc_MemoryError;
          break;
        default:
          break;
      }
      py::set_error(type, error.what());
    }
  });

  zephyr::bindings::BindOptions(module);
  zephyr::bindings::BindTypes(module);
  zephyr::bindings::BindEngine(module);
}
