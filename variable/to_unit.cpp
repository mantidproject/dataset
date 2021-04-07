// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <units/units.hpp>

#include "scipp/core/element/to_unit.h"
#include "scipp/core/time_point.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/to_unit.h"
#include "scipp/variable/transform.h"

namespace scipp::variable {

namespace {
constexpr double days_multiplier = llnl::units::precise::day.multiplier();
}

[[nodiscard]] SCIPP_VARIABLE_EXPORT double
conversion_scale(const units::Unit &from, const units::Unit &to,
                 const core::DType &dtype) {
  const auto scale =
      llnl::units::quick_convert(from.underlying(), to.underlying());
  if (std::isnan(scale))
    throw except::UnitError("Conversion from `" + to_string(from) + "` to `" +
                            to_string(to) + "` is not valid.");
  if (dtype == core::dtype<core::time_point> &&
      (from.underlying().multiplier() > days_multiplier ||
       to.underlying().multiplier() > days_multiplier)) {
    throw except::UnitError(
        "Unit conversion for datetimes with a unit greater than days"
        " is not implemented. Attempted conversion from `" +
        to_string(from) + "` to `" + to_string(to) + "`.");
  }
  return scale;
}

Variable to_unit(const VariableConstView &var, const units::Unit &unit) {
  return transform(var, conversion_scale(var.unit(), unit, var.dtype()) * unit,
                   core::element::to_unit);
}

} // namespace scipp::variable
