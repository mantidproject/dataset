// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp/dataset/dataset.h"

namespace scipp::dataset {

[[nodiscard]] SCIPP_DATASET_EXPORT DataArray rebin(const DataArrayConstView &a,
                                                   const Dim dim,
                                                   const Variable &coord);
[[nodiscard]] SCIPP_DATASET_EXPORT Dataset rebin(const DatasetConstView &d,
                                                 const Dim dim,
                                                 const Variable &coord);

} // namespace scipp::dataset
