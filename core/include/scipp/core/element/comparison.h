// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Piotr Rozyczko
#pragma once

#include <cmath>

#include "scipp/common/numeric.h"
#include "scipp/common/overloaded.h"
#include "scipp/core/element/arg_list.h"
#include "scipp/core/transform_common.h"

/// Operators to be used with transform and transform_in_place to implement
/// operations for Variable.
namespace scipp::core::element {

constexpr auto is_approx = overloaded{
    transform_flags::no_out_variance,
    arg_list<double, float, int64_t, int32_t, std::tuple<double, double, float>,
             std::tuple<int64_t, int32_t, int32_t>,
             std::tuple<int64_t, int64_t, int32_t>,
             std::tuple<int64_t, int32_t, int64_t>,
             std::tuple<int32_t, int32_t, int64_t>,
             std::tuple<int32_t, int64_t, int64_t>>,
    [](const units::Unit &x, const units::Unit &y, const units::Unit &t) {
      expect::equals(x, y);
      expect::equals(x, t);
      return units::dimensionless;
    },
    [](const auto &x, const auto &y, const auto &t) {
      using std::abs;
      return abs(x - y) <= t;
    }};

struct comparison_types_t {
  constexpr void operator()() const noexcept;
  using types = decltype(std::tuple_cat(std::declval<arithmetic_type_pairs>(),
                                        std::tuple<bool>{},
                                        std::tuple<core::time_point>{}));
};

constexpr auto comparison =
    overloaded{comparison_types_t{}, transform_flags::no_out_variance,
               [](const units::Unit &x, const units::Unit &y) {
                 expect::equals(x, y);
                 return units::dimensionless;
               }};

constexpr auto less = overloaded{
    comparison,
    [](const auto &x, const auto &y) { return x < y; },
};

constexpr auto greater = overloaded{
    comparison,
    [](const auto &x, const auto &y) { return x > y; },
};

constexpr auto less_equal = overloaded{
    comparison,
    [](const auto &x, const auto &y) { return x <= y; },
};

constexpr auto greater_equal =
    overloaded{comparison, [](const auto &x, const auto &y) { return x >= y; }};

constexpr auto equal = overloaded{
    comparison,
    [](const auto &x, const auto &y) { return x == y; },
};
constexpr auto not_equal =
    overloaded{comparison, [](const auto &x, const auto &y) { return x != y; }};

constexpr auto max_equals =
    overloaded{arg_list<double, float, int64_t, int32_t, bool>,
               transform_flags::expect_in_variance_if_out_variance,
               [](auto &&a, const auto &b) {
                 using std::max;
                 a = max(a, b);
               }};

constexpr auto nanmax_equals =
    overloaded{arg_list<double, float, int64_t, int32_t, bool>,
               transform_flags::expect_in_variance_if_out_variance,
               [](auto &&a, const auto &b) {
                 using numeric::isnan;
                 using std::max;
                 if (isnan(a))
                   a = b;
                 if (!isnan(b))
                   a = max(a, b);
               }};

constexpr auto min_equals =
    overloaded{arg_list<double, float, int64_t, int32_t, bool>,
               transform_flags::expect_in_variance_if_out_variance,
               [](auto &&a, const auto &b) {
                 using std::min;
                 a = min(a, b);
               }};

constexpr auto nanmin_equals =
    overloaded{arg_list<double, float, int64_t, int32_t, bool>,
               transform_flags::expect_in_variance_if_out_variance,
               [](auto &&a, const auto &b) {
                 using numeric::isnan;
                 using std::min;
                 if (isnan(a))
                   a = b;
                 if (!isnan(b))
                   a = min(a, b);
               }};

} // namespace scipp::core::element
