// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "pybind11.h"

namespace py = pybind11;

void init_buckets(py::module &);
void init_choose(py::module &);
void init_convolution(py::module &);
void init_comparison(py::module &);
void init_counts(py::module &);
void init_creation(py::module &);
void init_cumulative(py::module &);
void init_dataset(py::module &);
void init_dtype(py::module &);
void init_element_array_view(py::module &);
void init_exceptions(py::module &);
void init_groupby(py::module &);
void init_geometry(py::module &);
void init_histogram(py::module &);
void init_operations(py::module &);
void init_shape(py::module &);
void init_reduction(py::module &);
void init_trigonometry(py::module &);
void init_unary(py::module &);
void init_units(py::module &);
void init_variable(py::module &);

void init_generated_arithmetic(py::module &);
void init_generated_comparison(py::module &);
void init_generated_logical(py::module &);
void init_generated_math(py::module &);
void init_generated_trigonometry(py::module &);
void init_generated_util(py::module &);
void init_generated_special_values(py::module &);

void init_core(py::module &m) {
  auto core = m.def_submodule("core");
  init_units(core);
  init_exceptions(core);
  init_dtype(core);
  init_variable(core);
  init_buckets(core);
  init_convolution(core);
  init_choose(core);
  init_counts(core);
  init_creation(core);
  init_cumulative(core);
  init_dataset(core);
  init_groupby(core);
  init_comparison(core);
  init_operations(core);
  init_shape(core);
  init_geometry(core);
  init_histogram(core);
  init_reduction(core);
  init_trigonometry(core);
  init_unary(core);
  init_element_array_view(core);

  init_generated_arithmetic(core);
  init_generated_comparison(core);
  init_generated_logical(core);
  init_generated_math(core);
  init_generated_trigonometry(core);
  init_generated_util(core);
  init_generated_special_values(core);
}

PYBIND11_MODULE(_scipp, m) {
#ifdef SCIPP_VERSION
  m.attr("__version__") = py::str(SCIPP_VERSION);
#else
  m.attr("__version__") = py::str("unknown version");
#endif
#ifdef NDEBUG
  m.attr("_debug_") = py::cast(false);
#else
  m.attr("_debug_") = py::cast(true);
#endif
  init_core(m);
}
