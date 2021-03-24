// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Thibault Chatel
#pragma once

#include <vector>

#include "scipp-variable_export.h"
#include "scipp/variable/util.h" // for enum ascending, descending
#include "scipp/variable/variable.h"

namespace scipp::variable {

[[nodiscard]] SCIPP_VARIABLE_EXPORT Variable sort(const Variable &var,
                                                  const Dim dim,
                                                  const SortOrder order);

} // namespace scipp::variable
