#pragma once

#include <pybind11/pybind11.h>

namespace zephyr::bindings {

void BindOptions(pybind11::module_ &module);
void BindTypes(pybind11::module_ &module);
void BindEngine(pybind11::module_ &module);

}  // namespace zephyr::bindings
