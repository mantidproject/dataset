// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp/variable/variable.h"

namespace scipp::variable {

SCIPP_VARIABLE_EXPORT void copy_slices(const Variable &src, Variable dst,
                                       const Dim dim,
                                       const Variable &srcIndices,
                                       const Variable &dstIndices);

[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable resize_default_init(
    const Variable &var, const Dim dim, const scipp::index size);

[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable make_bins(Variable indices,
                                                       const Dim dim,
                                                       Variable buffer);

} // namespace scipp::variable
