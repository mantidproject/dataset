// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "pybind11.h"

#include "scipp/variable/convolution.h"

using namespace scipp;

namespace py = pybind11;

template <class T> void bind_convolve(py::module &m) {
  m.def(
      "convolve",
      [](const T &x, const T &kernel) { return convolve(x, kernel); },
      py::arg("x"), py::arg("kernel"),
      py::call_guard<py::gil_scoped_release>());
}

void init_convolution(py::module &m) { bind_convolve<Variable>(m); }
