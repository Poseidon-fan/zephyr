#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <zephyr/vector_adder.hpp>

namespace py = pybind11;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
PYBIND11_MODULE(_C, module) {
  module.doc() = "Zephyr native bindings";

  py::class_<zephyr::VectorAdder>(module, "VectorAdder")
      .def(py::init<>())
      .def("add", &zephyr::VectorAdder::Add, py::arg("left"), py::arg("right"));
}
