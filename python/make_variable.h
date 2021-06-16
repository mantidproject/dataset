// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp/core/dtype.h"
#include "scipp/units/unit.h"

#include "unit.h"

using namespace scipp;
using namespace scipp::variable;

namespace py = pybind11;

template <class T> struct MakeVariable {
  static Variable apply(const std::vector<Dim> &labels, const py::array &values,
                        const std::optional<py::array> &variances,
                        const units::Unit unit) {
    scipp::span shape{values.shape(), static_cast<size_t>(values.ndim())};
    Dimensions dims(labels, {shape.begin(), shape.end()});
    auto var = variances
                   ? makeVariable<T>(
                         Dimensions{dims},
                         Values(dims.volume(), core::default_init_elements),
                         Variances(dims.volume(), core::default_init_elements))
                   : makeVariable<T>(
                         Dimensions(dims),
                         Values(dims.volume(), core::default_init_elements));
    var.setUnit(unit);
    if (dims.empty()) {
      copy_element<ElementTypeMap<T>::convert>(
          cast_array_to_scalar<T>(values, unit), var.template value<T>());
      if (variances) {
        copy_element<ElementTypeMap<T>::convert>(
            cast_array_to_scalar<T>(*variances, unit),
            var.template variance<T>());
      }
    } else {
      copy_array_into_view(cast_to_array_like<T>(values, unit),
                           var.template values<T>(), dims);
      if (variances) {
        copy_array_into_view(cast_to_array_like<T>(*variances, unit),
                             var.template variances<T>(), dims);
      }
    }
    return var;
  }
};

template <class T> struct MakeVariableDefaultInit {
  static Variable apply(const std::vector<Dim> &labels,
                        const std::vector<scipp::index> &shape,
                        const units::Unit unit, const bool variances) {
    Dimensions dims(labels, shape);
    auto var = variances
                   ? makeVariable<T>(Dimensions{dims}, Values{}, Variances{})
                   : makeVariable<T>(Dimensions{dims});
    var.setUnit(unit);
    return var;
  }
};

template <class ST> struct MakeODFromNativePythonTypes {
  template <class T> struct Maker {
    static Variable apply([[maybe_unused]] const units::Unit unit,
                          const ST &value, const std::optional<ST> &variance) {
      if constexpr (std::is_same_v<T, core::time_point>) {
        if constexpr (std::is_integral_v<ST>) {
          if (variance.has_value()) {
            throw except::VariancesError("datetimes cannot have variances.");
          }
          return makeVariable<core::time_point>(Values{core::time_point{value}},
                                                unit);
        } else {
          throw except::TypeError(
              "Unsupported dtype for constructing datetime64: " +
              to_string(core::dtype<ST>));
        }
      } else {
        auto var = variance ? makeVariable<T>(Values{T(value)},
                                              Variances{T(variance.value())})
                            : makeVariable<T>(Values{T(value)});
        var.setUnit(unit);
        return var;
      }
    }
  };

  static Variable make(const units::Unit unit, const ST &value,
                       const std::optional<ST> &variance,
                       const py::object &dtype) {
    return core::CallDType<double, float, int64_t, int32_t, bool,
                           core::time_point>::apply<Maker>(scipp_dtype(dtype),
                                                           unit, value,
                                                           variance);
  }
};

template <class T>
Variable init_1D_no_variance(const std::vector<Dim> &labels,
                             const std::vector<scipp::index> &shape,
                             const std::vector<T> &values,
                             const units::Unit &unit) {
  Variable var;
  var = makeVariable<T>(Dims(labels), Shape(shape),
                        Values(values.begin(), values.end()));
  var.setUnit(unit);
  return var;
}

template <class T>
auto do_init_0D(const T &value, const std::optional<T> &variance,
                const units::Unit &unit) {
  using Elem = std::conditional_t<std::is_same_v<T, py::object>,
                                  scipp::python::PyObject, T>;
  Variable var;
  if (variance)
    var = makeVariable<Elem>(Values{value}, Variances{*variance});
  else
    var = makeVariable<Elem>(Values{value});
  var.setUnit(unit);
  return var;
}

// If we use py::array as the type of values and variances, then pybind11 only
// accepts exactly numpy.array as input and not types that are convertible to
// an array (e.g. list). But doing the conversion manually using .cast<py:array>
// allows this function to work with anything that can be converted to an array.
// The downside is that we make an extra copy if the input is not already an
// array. But that is likely not important as lists / tuples should not contain
// large data.
Variable do_make_variable(const std::vector<Dim> &labels,
                          const py::object &values,
                          const std::optional<py::object> &variances,
                          units::Unit unit, const py::object &dtype) {
  const auto &values_array = values.cast<py::array>();
  // Use custom dtype, otherwise dtype of data.
  const auto dtypeTag =
      dtype.is_none() ? scipp_dtype(values_array.dtype()) : scipp_dtype(dtype);

  if (labels.size() == 1 && !variances) {
    if (dtypeTag == core::dtype<std::string>) {
      std::vector<scipp::index> shape(
          values_array.shape(), values_array.shape() + values_array.ndim());
      return init_1D_no_variance(labels, shape,
                                 values.cast<std::vector<std::string>>(), unit);
    }
  }

  if (dtypeTag == scipp::dtype<core::time_point>) {
    if (variances.has_value()) {
      throw except::VariancesError("datetimes cannot have variances.");
    }
    const auto [actual_unit, value_factor] =
        get_time_unit(values_array, dtype, unit);

    if (value_factor != 1) {
      throw std::invalid_argument(
          "Scaling datetimes is not supported. The units of the datetime64 "
          "objects must match the unit of the Variables.");
    }
    unit = actual_unit;
  }

  return core::CallDType<double, float, int64_t, int32_t, bool,
                         scipp::core::time_point, std::string>::
      apply<MakeVariable>(dtypeTag, labels, values_array,
                          variances.has_value()
                              ? std::optional{variances->cast<py::array>()}
                              : std::optional<py::array>{},
                          unit);
}

Variable makeVariableDefaultInit(const std::vector<Dim> &labels,
                                 const std::vector<scipp::index> &shape,
                                 units::Unit unit, const py::object &dtype,
                                 const bool variances) {
  const auto dtypeTag = scipp_dtype(dtype);
  if (dtypeTag == scipp::dtype<core::time_point>) {
    if (variances) {
      throw except::VariancesError("datetimes cannot have variances.");
    }
    const auto [actual_unit, value_factor] =
        get_time_unit(std::nullopt,
                      dtype.is_none() ? std::optional<units::Unit>{}
                                      : parse_datetime_dtype(dtype),
                      unit);
    if (value_factor != 1) {
      throw std::invalid_argument(
          "Scaling datetimes is not supported. The units of the datetime64 "
          "objects must match the unit of the Variables.");
    }
    unit = actual_unit;
  }
  return core::CallDType<
      double, float, int64_t, int32_t, bool, scipp::core::time_point,
      std::string, Variable, DataArray, Dataset, Eigen::Vector3d,
      Eigen::Matrix3d>::apply<MakeVariableDefaultInit>(dtypeTag, labels, shape,
                                                       unit, variances);
}
