// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <scipp/dataset/dataset.h>

namespace scipp::dataset {

SCIPP_DATASET_EXPORT DataArray
bin(const DataArrayConstView &array,
    const std::vector<VariableConstView> &edges,
    const std::vector<VariableConstView> &groups = {},
    const std::vector<Dim> &erase = {});

template <class Coords, class Masks, class Attrs>
SCIPP_DATASET_EXPORT DataArray
bin(const VariableConstView &data, const Coords &coords, const Masks &masks,
    const Attrs &attrs, const std::vector<VariableConstView> &edges,
    const std::vector<VariableConstView> &groups = {},
    const std::vector<Dim> &erase = {});

} // namespace scipp::dataset
