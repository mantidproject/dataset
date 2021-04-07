// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

#include "unit.h"

#include "scipp/units/string.h"
#include "scipp/variable/to_unit.h"

#include "dtype.h"

using namespace scipp;
namespace py = pybind11;

namespace {
bool temporal_or_dimensionless(const units::Unit unit) {
  return unit == units::one || unit.has_same_base(units::s);
}
} // namespace

std::tuple<units::Unit, double>
get_time_unit(const std::optional<scipp::units::Unit> value_unit,
              const std::optional<scipp::units::Unit> dtype_unit,
              const units::Unit sc_unit) {
  if (!temporal_or_dimensionless(sc_unit)) {
    throw std::invalid_argument("Invalid unit for dtype=datetime64: " +
                                to_string(sc_unit));
  }
  if (dtype_unit.value_or(units::one) != units::one &&
      (sc_unit != units::one && *dtype_unit != sc_unit)) {
    throw std::invalid_argument(
        "dtype (datetime64[" + to_string(*dtype_unit) +
        "]) has a different time unit from 'unit' argument (" +
        to_string(sc_unit) + ")");
  }
  units::Unit actual_unit = units::one;
  if (sc_unit != units::one)
    actual_unit = sc_unit;
  else if (dtype_unit.value_or(units::one) != units::one)
    actual_unit = *dtype_unit;
  else if (value_unit.has_value())
    actual_unit = *value_unit;

  if (value_unit && value_unit != actual_unit) {
    const auto scale = variable::conversion_scale(*value_unit, actual_unit,
                                                  dtype<core::time_point>);
    return {actual_unit, scale};
  }
  return {actual_unit, 1.0};
}

std::tuple<units::Unit, double> get_time_unit(const py::buffer &value,
                                              const py::object &dtype,
                                              const units::Unit unit) {
  return get_time_unit(
      value.is_none() || value.attr("dtype").attr("kind").cast<char>() != 'M'
          ? std::optional<units::Unit>{}
          : parse_datetime_dtype(value),
      dtype.is_none() ? std::optional<units::Unit>{}
                      : parse_datetime_dtype(dtype),
      unit);
}

std::string to_numpy_time_string(const scipp::units::Unit unit) {
  return unit == units::us
             ? std::string("us")
             : unit == units::Unit("min") ? std::string("m") : to_string(unit);
}