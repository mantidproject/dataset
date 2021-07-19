# ~~~
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# ~~~

# Use Python_add_library since pybind11_add_module causes linking to libpython,
# which should be avoided on Linux and OSX.
python_add_library(
  _scipp
  MODULE
  ${python_SRC_FILES}
  bind_units.cpp
  bins.cpp
  choose.cpp
  comparison.cpp
  convolution.cpp
  counts.cpp
  variable_creation.cpp
  cumulative.cpp
  dataset.cpp
  docstring.cpp
  dtype.cpp
  except.cpp
  geometry.cpp
  groupby.cpp
  histogram.cpp
  numpy.cpp
  operations.cpp
  py_object.cpp
  scipp.cpp
  reduction.cpp
  trigonometry.cpp
  unary.cpp
  unit.cpp
  variable.cpp
  variable_init.cpp
  variable_instantiate_py_object.cpp
  element_array_view.cpp
  shape.cpp
)

target_include_directories(
  _scipp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(_scipp LINK_PRIVATE scipp-dataset pybind11::headers)

# SCIPP_EXPORT is used in macros defined in variable/
target_compile_definitions(_scipp PRIVATE SCIPP_EXPORT=)

# Set symbol visibility to hidden to reduce binary size, as recommended in
# pybind11 FAQ.
set_target_properties(_scipp PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(
  _scipp PROPERTIES INSTALL_RPATH
                    "${CMAKE_INSTALL_RPATH}/../${CMAKE_INSTALL_LIBDIR}"
)

pybind11_strip(_scipp)
pybind11_extension(_scipp)

if(PRECOMPILED_HEADERS)
  target_precompile_headers(_scipp PRIVATE pybind11.h)
endif()

add_sanitizers(_scipp)

install(DIRECTORY "src/scipp/" DESTINATION ${PYTHONDIR})
install(TARGETS _scipp DESTINATION ${PYTHONDIR})
