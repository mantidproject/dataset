// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#pragma once

#include "scipp-variable_export.h"
#include "scipp/variable/variable.h"

#include "scipp/variable/generated_special_values.h"

namespace scipp::variable {

SCIPP_VARIABLE_EXPORT Variable &nan_to_num(const VariableConstView &var,
                                           const VariableConstView &replacement,
                                           Variable &out);
SCIPP_VARIABLE_EXPORT Variable &
positive_inf_to_num(const VariableConstView &var,
                    const VariableConstView &replacement, Variable &out);
SCIPP_VARIABLE_EXPORT Variable &
negative_inf_to_num(const VariableConstView &var,
                    const VariableConstView &replacement, Variable &out);

[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable
nan_to_num(const VariableConstView &var, const VariableConstView &replacement);
[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable pos_inf_to_num(
    const VariableConstView &var, const VariableConstView &replacement);
[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable neg_inf_to_num(
    const VariableConstView &var, const VariableConstView &replacement);

} // namespace scipp::variable
