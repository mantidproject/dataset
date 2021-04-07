// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <scipp/dataset/dataset.h>
#include <scipp/variable/util.h>
#include <scipp/variable/variable.h>

namespace scipp::dataset {
using scipp::variable::SortOrder;

SCIPP_DATASET_EXPORT Variable
sort(const Variable &var, const Variable &key,
     const SortOrder &order = SortOrder::Ascending);
SCIPP_DATASET_EXPORT DataArray
sort(const DataArray &array, const Variable &key,
     const SortOrder &order = SortOrder::Ascending);
SCIPP_DATASET_EXPORT DataArray
sort(const DataArray &array, const Dim &key,
     const SortOrder &order = SortOrder::Ascending);
SCIPP_DATASET_EXPORT Dataset
sort(const Dataset &dataset, const Variable &key,
     const SortOrder &order = SortOrder::Ascending);
SCIPP_DATASET_EXPORT Dataset
sort(const Dataset &dataset, const Dim &key,
     const SortOrder &order = SortOrder::Ascending);

} // namespace scipp::dataset
