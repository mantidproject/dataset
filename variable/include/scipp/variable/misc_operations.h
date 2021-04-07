// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp-variable_export.h"
#include "scipp/variable/variable.h"

namespace scipp::variable {

SCIPP_VARIABLE_EXPORT Variable astype(const Variable &var, const DType type);

SCIPP_VARIABLE_EXPORT std::vector<Variable>
split(const Variable &var, const Dim dim,
      const std::vector<scipp::index> &indices);
SCIPP_VARIABLE_EXPORT Variable filter(const Variable &var,
                                      const Variable &filter);

SCIPP_VARIABLE_EXPORT Variable masked_to_zero(const Variable &var,
                                              const Variable &mask);

namespace geometry {
[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable position(const Variable &x,
                                                      const Variable &y,
                                                      const Variable &z);
[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable x(const Variable &pos);
[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable y(const Variable &pos);
[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable z(const Variable &pos);

} // namespace geometry

} // namespace scipp::variable
