// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Jan-Lukas Wynen

#include "numpy.h"

#include "dtype.h"

scipp::core::time_point make_time_point(const pybind11::buffer &buffer,
                                        const int64_t scale) {
  // buffer.cast does not always work because numpy.datetime64.__int__
  // delegates to datetime.datetime if the unit is larger than ns and
  // that cannot be converted to long.
  using PyType = typename ElementTypeMap<core::time_point>::PyType;
  return core::time_point{
      buffer.attr("astype")(py::dtype::of<PyType>()).cast<PyType>() * scale};
}
