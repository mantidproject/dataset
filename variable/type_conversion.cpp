// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Igor Gudich
#include <cmath>

#include "scipp/core/tag_util.h"
#include "scipp/core/transform_common.h"
#include "scipp/variable/misc_operations.h"
#include "scipp/variable/transform.h"
#include "scipp/variable/variable.h"

namespace scipp::variable {

struct MakeVariableWithType {
  template <class T> struct Maker {
    static Variable apply(const VariableConstView &parent) {
      using namespace core::transform_flags;
      constexpr auto expect_input_variances =
          conditional_flag<!core::canHaveVariances<T>()>(
              expect_no_variance_arg<0>);
      return transform<double, float, int64_t, int32_t, bool>(
          parent,
          overloaded{
              expect_input_variances, [](const units::Unit &x) { return x; },
              [](const auto &x) {
                if constexpr (is_ValueAndVariance_v<std::decay_t<decltype(x)>>)
                  return ValueAndVariance<T>{static_cast<T>(x.value),
                                             static_cast<T>(x.variance)};
                else
                  return static_cast<T>(x);
              }});
    }
  };
  static Variable make(const VariableConstView &var, DType type) {
    return core::CallDType<double, float, int64_t, int32_t, bool>::apply<Maker>(
        type, var);
  }
};

Variable astype(const VariableConstView &var, DType type) {
  return type == var.dtype() ? Variable(var)
                             : MakeVariableWithType::make(var, type);
}
} // namespace scipp::variable
