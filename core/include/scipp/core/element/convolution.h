// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp/common/overloaded.h"
#include "scipp/core/element/arg_list.h"
#include "scipp/core/transform_common.h"
#include "scipp/units/unit.h"

namespace scipp::core::element {

constexpr auto convolve = overloaded{
    transform_flags::expect_in_variance_if_out_variance,
    transform_flags::expect_no_variance_arg<2>, arg_list<double>,
    [](auto &out, const auto &x, const auto &kernel) { out += x * kernel; },
    [](units::Unit &out, const units::Unit &x, const units::Unit &kernel) {
      core::expect::equals(out, x * kernel);
    }};

} // namespace scipp::core::element
