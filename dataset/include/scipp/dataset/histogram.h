// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <algorithm>
#include <set>
#include <tuple>

#include "scipp/dataset/dataset.h"

namespace scipp::dataset {

SCIPP_DATASET_EXPORT DataArray histogram(const DataArrayConstView &events,
                                         const Variable &binEdges);
SCIPP_DATASET_EXPORT Dataset histogram(const DatasetConstView &dataset,
                                       const Variable &bins);

SCIPP_DATASET_EXPORT std::set<Dim> edge_dimensions(const DataArrayConstView &a);
SCIPP_DATASET_EXPORT Dim edge_dimension(const DataArrayConstView &a);
SCIPP_DATASET_EXPORT bool is_histogram(const DataArrayConstView &a,
                                       const Dim dim);
SCIPP_DATASET_EXPORT bool is_histogram(const DatasetConstView &a,
                                       const Dim dim);

} // namespace scipp::dataset
