// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp-variable_export.h"
#include "scipp/variable/variable.h"

namespace scipp::variable {

[[nodiscard]] SCIPP_VARIABLE_EXPORT double
conversion_scale(const units::Unit &a, const units::Unit &b,
                 const core::DType &dtype);
[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable
to_unit(const VariableConstView &var, const units::Unit &unit);

} // namespace scipp::variable
