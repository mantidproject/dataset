// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp-units_export.h"
#include "scipp/common/except.h"
#include "scipp/units/string.h"
#include "scipp/units/unit.h"

namespace scipp::except {

using UnitError = Error<units::Unit>;
using UnitMismatchError = MismatchError<units::Unit>;

// We need deduction guides such that, e.g., the exception for a Variable
// mismatch and VariableView mismatch are the same type.
template <class T>
MismatchError(const units::Unit &, const T &) -> MismatchError<units::Unit>;
template <class T>
MismatchError(const units::Dim &, const T &) -> MismatchError<units::Dim>;

template struct SCIPP_UNITS_EXPORT Error<units::Unit>;
template struct SCIPP_UNITS_EXPORT MismatchError<units::Unit>;

} // namespace scipp::except
