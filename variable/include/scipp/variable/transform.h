// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file Various transform functions for variables.
///
/// The underlying mechanism of the implementation is as follows:
/// 1. `visit<...>::apply` obtains the concrete underlying data type(s).
/// 2. `Transform` is applied to that concrete container, calling
///    `do_transform`. `Transform` essentially builds a callable accepting a
///    container from a callable accepting an element of the container.
/// 3. `do_transform` is essentially a fancy std::transform. It uses recursion
///    to process optional flags (provided as base classes of the user-provided
///    operator). It provides automatic handling of data that has variances in
///    addition to values, calling a different transform implementation for each
///    case (different instantiations of `transform_elements`).
/// 4. The `transform_elements` function calls the overloaded operator for
///    each element. This is also were multi-threading for the majority of
///    scipp's operations is implemented.
///
/// Handling of binned data is mostly hidden in this implementation, reducing
/// code duplication:
/// - `variableFactory()` is used for output creation and unit access.
/// - `variableFactory()` is used in `visit.h` to obtain a direct pointer to the
///   underlying buffer.
/// - MultiIndex contains special handling for binned data, i.e., it can iterate
///   the buffer in a binning-aware way.
///
/// The mechanism for in-place transformation is mostly identical to the one
/// outlined above.
///
/// @author Simon Heybrock
#pragma once

#include <algorithm>
#include <cassert>
#include <string_view>

#include "scipp/common/overloaded.h"

#include "scipp/core/has_eval.h"
#include "scipp/core/multi_index.h"
#include "scipp/core/parallel.h"
#include "scipp/core/transform_common.h"
#include "scipp/core/value_and_variance.h"
#include "scipp/core/values_and_variances.h"

#include "scipp/variable/except.h"
#include "scipp/variable/variable.h"
#include "scipp/variable/variable_factory.h"
#include "scipp/variable/visit.h"

namespace scipp::variable {

namespace detail {

template <class T> struct has_variances : std::false_type {};
template <class T>
struct has_variances<ValueAndVariance<T>> : std::true_type {};
template <class T>
struct has_variances<ValuesAndVariances<T>> : std::true_type {};
template <class T>
inline constexpr bool has_variances_v = has_variances<T>::value;

/// Helper for the transform implementation to unify iteration of data with and
/// without variances.
template <class T>
static constexpr decltype(auto) value_maybe_variance(T &&range,
                                                     const scipp::index i) {
  if constexpr (has_variances_v<std::decay_t<T>>) {
    return ValueAndVariance{range.values.data()[i], range.variances.data()[i]};
  } else {
    return range.data()[i];
  }
}

// Helpers for handling a tuple of indices (integers or ViewIndex).
namespace iter {

template <class T> static constexpr auto array_params(T &&iterable) noexcept {
  if constexpr (is_ValuesAndVariances_v<std::decay_t<T>>)
    return iterable.values;
  else
    return iterable;
}

template <int N, class T> static constexpr auto get(const T &index) noexcept {
  if constexpr (visit_detail::is_tuple<T>::value ||
                visit_detail::is_array<T>::value) {
    if constexpr (std::is_integral_v<std::tuple_element_t<0, T>>)
      return std::get<N>(index);
    else
      return std::get<N>(index).get();
  } else
    return std::get<N>(index.get());
}

} // namespace iter

template <size_t N_Operands, bool in_place>
inline constexpr auto stride_special_cases =
    std::array<std::array<scipp::index, N_Operands>, 0>{};

template <>
inline constexpr auto stride_special_cases<1, true> =
    std::array<std::array<scipp::index, 1>, 2>{{{1}}};

template <>
inline constexpr auto stride_special_cases<2, true> =
    std::array<std::array<scipp::index, 2>, 4>{{{1, 1}, {0, 1}, {1, 0}}};

template <>
inline constexpr auto stride_special_cases<2, false> =
    std::array<std::array<scipp::index, 2>, 1>{{{1, 1}}};

template <>
inline constexpr auto stride_special_cases<3, false> =
    std::array<std::array<scipp::index, 3>, 3>{
        {{1, 1, 1}, {1, 0, 1}, {1, 1, 0}}};

template <size_t I, size_t N_Operands, bool in_place, size_t... Is>
auto stride_sequence_impl(std::index_sequence<Is...>) -> std::integer_sequence<
    scipp::index, stride_special_cases<N_Operands, in_place>.at(I)[Is]...>;
// THe above uses std::array::at instead of operator[] in order to circumvent
// a false positive error in MSVC 19.

template <size_t I, size_t N_Operands, bool in_place> struct stride_sequence {
  using type = decltype(stride_sequence_impl<I, N_Operands, in_place>(
      std::make_index_sequence<N_Operands>{}));
};

template <size_t I, size_t N_Operands, bool in_place>
using make_stride_sequence =
    typename stride_sequence<I, N_Operands, in_place>::type;

template <scipp::index... Strides, size_t... Is>
void increment_impl(std::array<scipp::index, sizeof...(Strides)> &indices,
                    std::integer_sequence<size_t, Is...>) noexcept {
  ((indices[Is] += Strides), ...);
}

template <scipp::index... Strides>
void increment(std::array<scipp::index, sizeof...(Strides)> &indices) noexcept {
  increment_impl<Strides...>(indices,
                             std::make_index_sequence<sizeof...(Strides)>{});
}

template <size_t N>
void increment(std::array<scipp::index, N> &indices,
               const scipp::span<const scipp::index> strides) noexcept {
  for (size_t i = 0; i < N; ++i) {
    indices[i] += strides[i];
  }
}

template <class Op, class Indices, class... Args, size_t... I>
static constexpr auto call_impl(Op &&op, const Indices &indices,
                                std::index_sequence<I...>, Args &&... args) {
  return op(value_maybe_variance(args, iter::get<I + 1>(indices))...);
}
template <class Op, class Indices, class Out, class... Args>
static constexpr void call(Op &&op, const Indices &indices, Out &&out,
                           Args &&... args) {
  const auto i = iter::get<0>(indices);
  auto &&out_ = value_maybe_variance(out, i);
  out_ = call_impl(std::forward<Op>(op), indices,
                   std::make_index_sequence<sizeof...(Args)>{},
                   std::forward<Args>(args)...);
  if constexpr (is_ValueAndVariance_v<std::decay_t<decltype(out_)>>) {
    out.values.data()[i] = out_.value;
    out.variances.data()[i] = out_.variance;
  }
}

template <class Op, class Indices, class Arg, class... Args, size_t... I>
static constexpr void call_in_place_impl(Op &&op, const Indices &indices,
                                         std::index_sequence<I...>, Arg &&arg,
                                         Args &&... args) {
  static_assert(
      std::is_same_v<decltype(op(arg, value_maybe_variance(
                                          args, iter::get<I + 1>(indices))...)),
                     void>);
  op(arg, value_maybe_variance(args, iter::get<I + 1>(indices))...);
}
template <class Op, class Indices, class Arg, class... Args>
static constexpr void call_in_place(Op &&op, const Indices &indices, Arg &&arg,
                                    Args &&... args) {
  const auto i = iter::get<0>(indices);
  // For dense data we conditionally create ValueAndVariance, which performs an
  // element copy, so the result may have to be updated after the call to `op`.
  auto &&arg_ = value_maybe_variance(arg, i);
  call_in_place_impl(std::forward<Op>(op), indices,
                     std::make_index_sequence<sizeof...(Args)>{},
                     std::forward<decltype(arg_)>(arg_),
                     std::forward<Args>(args)...);
  if constexpr (is_ValueAndVariance_v<std::decay_t<decltype(arg_)>>) {
    arg.values.data()[i] = arg_.value;
    arg.variances.data()[i] = arg_.variance;
  }
}
/// Run transform with strides known at compile time.
template <bool in_place, class Op, class... Operands, scipp::index... Strides>
static void inner_loop(Op &&op,
                       std::array<scipp::index, sizeof...(Operands)> indices,
                       std::integer_sequence<scipp::index, Strides...>,
                       const scipp::index n, Operands &&... operands) {
  static_assert(sizeof...(Operands) == sizeof...(Strides));

  for (scipp::index i = 0; i < n; ++i) {
    if constexpr (in_place) {
      detail::call_in_place(op, indices, std::forward<Operands>(operands)...);
    } else {
      detail::call(op, indices, std::forward<Operands>(operands)...);
    }
    detail::increment<Strides...>(indices);
  }
}

/// Run transform with strides known at run time but bypassing MultiIndex.
template <bool in_place, class Op, class... Operands>
static void inner_loop(Op &&op,
                       std::array<scipp::index, sizeof...(Operands)> indices,
                       const scipp::span<const scipp::index> strides,
                       const scipp::index n, Operands &&... operands) {
  for (scipp::index i = 0; i < n; ++i) {
    if constexpr (in_place) {
      detail::call_in_place(op, indices, std::forward<Operands>(operands)...);
    } else {
      detail::call(op, indices, std::forward<Operands>(operands)...);
    }
    detail::increment(indices, strides);
  }
}

template <bool in_place, size_t I = 0, class Op, class... Operands>
static void dispatch_inner_loop(
    Op &&op, const std::array<scipp::index, sizeof...(Operands)> &indices,
    const scipp::span<const scipp::index> inner_strides, const scipp::index n,
    Operands &&... operands) {
  constexpr auto N_Operands = sizeof...(Operands);
  if constexpr (I ==
                detail::stride_special_cases<N_Operands, in_place>.size()) {
    inner_loop<in_place>(std::forward<Op>(op), indices, inner_strides, n,
                         std::forward<Operands>(operands)...);
  } else {
    if (std::equal(
            inner_strides.begin(), inner_strides.end(),
            detail::stride_special_cases<N_Operands, in_place>[I].begin())) {
      inner_loop<in_place>(
          std::forward<Op>(op), indices,
          detail::make_stride_sequence<I, N_Operands, in_place>{}, n,
          std::forward<Operands>(operands)...);
    } else {
      dispatch_inner_loop<in_place, I + 1>(op, indices, inner_strides, n,
                                           std::forward<Operands>(operands)...);
    }
  }
}

template <class Op, class Out, class... Ts>
static void transform_elements(Op op, Out &&out, Ts &&... other) {
  const auto begin =
      core::MultiIndex(iter::array_params(out), iter::array_params(other)...);

  auto run = [&](auto &indices, const auto &end) {
    const auto inner_strides = indices.inner_strides();
    while (indices != end) {
      // Shape can change when moving between bins -> recompute every time.
      const auto inner_size = indices.in_same_chunk(end, 1)
                                  ? indices.inner_distance_to(end)
                                  : indices.inner_distance_to_end();
      dispatch_inner_loop<false>(op, indices.get(), inner_strides, inner_size,
                                 std::forward<Out>(out),
                                 std::forward<Ts>(other)...);
      indices.increment_inner_by(inner_size != 0 ? inner_size : 1);
      indices.increment_outer();
    }
  };

  auto run_parallel = [&](const auto &range) {
    auto indices = begin;
    indices.set_index(range.begin());
    auto end = begin;
    end.set_index(range.end());
    run(indices, end);
  };
  core::parallel::parallel_for(core::parallel::blocked_range(0, out.size()),
                               run_parallel);
}

template <class T> static constexpr auto maybe_eval(T &&_) {
  if constexpr (core::has_eval_v<std::decay_t<T>>)
    return _.eval();
  else
    return std::forward<T>(_);
}

template <class Op, class... Args>
constexpr bool check_all_or_none_variances =
    std::is_base_of_v<core::transform_flags::expect_all_or_none_have_variance_t,
                      Op> &&
    !std::conjunction_v<is_ValuesAndVariances<std::decay_t<Args>>...> &&
    std::disjunction_v<is_ValuesAndVariances<std::decay_t<Args>>...>;

/// Recursion endpoint for do_transform.
///
/// Call transform_elements with or without variances for output, depending on
/// whether any of the arguments has variances or not.
template <class Op, class Out, class Tuple>
static void do_transform(Op op, Out &&out, Tuple &&processed) {
  auto out_val = out.values();
  std::apply(
      [&op, &out, &out_val](auto &&... args) {
        if constexpr (check_all_or_none_variances<Op, decltype(args)...>) {
          throw except::VariancesError(
              "Expected either all or none of inputs to have variances.");
        } else if constexpr (
            !std::is_base_of_v<core::transform_flags::no_out_variance_t, Op> &&
            core::canHaveVariances<typename Out::value_type>() &&
            (is_ValuesAndVariances_v<std::decay_t<decltype(args)>> || ...)) {
          auto out_var = out.variances();
          transform_elements(op, ValuesAndVariances{out_val, out_var},
                             std::forward<decltype(args)>(args)...);
        } else {
          transform_elements(op, out_val,
                             std::forward<decltype(args)>(args)...);
        }
      },
      std::forward<Tuple>(processed));
}

/// Helper for transform implementation, performing branching between output
/// with and without variances as well as handling other operands with and
/// without variances.
template <class Op, class Out, class Tuple, class Arg, class... Args>
static void do_transform(Op op, Out &&out, Tuple &&processed, const Arg &arg,
                         const Args &... args) {
  auto vals = arg.values();
  if (arg.hasVariances()) {
    if constexpr (std::is_base_of_v<
                      core::transform_flags::expect_no_variance_arg_t<
                          std::tuple_size_v<Tuple>>,
                      Op>) {
      throw except::VariancesError("Variances in argument " +
                                   std::to_string(std::tuple_size_v<Tuple>) +
                                   " not supported.");
    } else if constexpr (
        std::is_base_of_v<
            core::transform_flags::
                expect_no_in_variance_if_out_cannot_have_variance_t,
            Op> &&
        !core::canHaveVariances<typename Out::value_type>()) {
      throw except::VariancesError(
          "Variances in argument " + std::to_string(std::tuple_size_v<Tuple>) +
          " not supported as output dtype cannot have variances");
    } else if constexpr (core::canHaveVariances<typename Arg::value_type>()) {
      auto vars = arg.variances();
      do_transform(
          op, std::forward<Out>(out),
          std::tuple_cat(processed, std::tuple(ValuesAndVariances{vals, vars})),
          args...);
    }
    // else {}  // Cannot happen because args.hasVariances()
    //             implies canHaveVariances<value_type>.
    //             The 2nd test is needed to avoid compilation errors
    //             (hasVariances is a runtime check).
  } else {
    if constexpr (std::is_base_of_v<
                      core::transform_flags::expect_variance_arg_t<
                          std::tuple_size_v<Tuple>>,
                      Op>)
      throw except::VariancesError("Variances missing in argument " +
                                   std::to_string(std::tuple_size_v<Tuple>) +
                                   " . Must be set.");
    do_transform(op, std::forward<Out>(out),
                 std::tuple_cat(processed, std::tuple(vals)), args...);
  }
}

template <class T> struct as_view {
  using value_type = typename T::value_type;
  [[nodiscard]] bool hasVariances() const { return data.hasVariances(); }
  auto values() const { return decltype(data.values())(data.values(), dims); }
  auto variances() const {
    return decltype(data.variances())(data.variances(), dims);
  }
  T &data;
  const Dimensions &dims;
};
template <class T> as_view(T &data, const Dimensions &dims) -> as_view<T>;

template <class Op> struct Transform {
  Op op;
  template <class... Ts> Variable operator()(Ts &&... handles) const {
    const auto dims = merge(handles.dims()...);
    using Out = decltype(maybe_eval(op(handles.values()[0]...)));
    const bool variances =
        !std::is_base_of_v<core::transform_flags::no_out_variance_t, Op> &&
        core::canHaveVariances<Out>() && (handles.hasVariances() || ...);
    auto unit = op.base_op()(variableFactory().elem_unit(*handles.m_var)...);
    auto out = variableFactory().create(dtype<Out>, dims, unit, variances,
                                        *handles.m_var...);
    do_transform(op, variable_access<Out>(out), std::tuple<>(),
                 as_view{handles, dims}...);
    return out;
  }
};
template <class Op> Transform(Op) -> Transform<Op>;

// std::tuple_cat does not work correctly on with clang-7. Issue with
// Eigen::Vector3d.
template <typename T, typename...> struct tuple_cat { using type = T; };
template <template <typename...> class C, typename... Ts1, typename... Ts2,
          typename... Ts3>
struct tuple_cat<C<Ts1...>, C<Ts2...>, Ts3...>
    : public tuple_cat<C<Ts1..., Ts2...>, Ts3...> {};

template <class Op> struct wrap_eigen : Op {
  const Op &base_op() const noexcept { return *this; }
  template <class... Ts> constexpr auto operator()(Ts &&... args) const {
    if constexpr ((core::has_eval_v<std::decay_t<Ts>> || ...))
      // WARNING! The explicit specification of the template arguments of
      // operator() is EXTREMELY IMPORTANT. It ensures that Eigen types are
      // passed BY REFERENCE and NOT BY VALUE. Passing by value leads to
      // construction of expressions of values on the stack, which are then
      // returned from the operator. One way to identify this is using
      // address-sanitizer, which finds a `stack-use-after-scope`.
      return Op::template operator()<Ts...>(std::forward<Ts>(args)...);
    else
      return Op::template operator()(std::forward<Ts>(args)...);
  }
};
template <class... Ts> wrap_eigen(Ts...) -> wrap_eigen<Ts...>;
} // namespace detail

template <class... Ts, class Op>
static constexpr auto type_tuples(Op) noexcept {
  if constexpr (sizeof...(Ts) == 0)
    return typename Op::types{};
  else if constexpr ((visit_detail::is_tuple<Ts>::value || ...))
    return typename detail::tuple_cat<Ts...>::type{};
  else
    return std::tuple<Ts...>{};
}

constexpr auto overlaps = [](const auto &a, const auto &b) {
  if constexpr (std::is_same_v<typename std::decay_t<decltype(a)>::value_type,
                               typename std::decay_t<decltype(b)>::value_type>)
    return a.values().overlaps(b.values());
  else
    return false;
};

/// Helper class wrapping functions for in-place transform.
///
/// The dry_run template argument can be used to disable any actual modification
/// of data. This is used to implement operations on datasets with a strong
/// exception guarantee.
template <bool dry_run> struct in_place {
  template <class Op, class T, class... Ts>
  static void transform_in_place_impl(Op op, T &&arg, Ts &&... other) {
    using namespace detail;
    const auto begin =
        core::MultiIndex(iter::array_params(arg), iter::array_params(other)...);
    if constexpr (dry_run)
      return;

    auto run = [&](auto &indices, const auto &end) {
      const auto inner_strides = indices.inner_strides();
      while (indices != end) {
        // Shape can change when moving between bins -> recompute every time.
        const auto inner_size = indices.in_same_chunk(end, 1)
                                    ? indices.inner_distance_to(end)
                                    : indices.inner_distance_to_end();
        detail::dispatch_inner_loop<true>(op, indices.get(), inner_strides,
                                          inner_size, std::forward<T>(arg),
                                          std::forward<Ts>(other)...);
        indices.increment_inner_by(inner_size != 0 ? inner_size : 1);
        indices.increment_outer();
      }
    };
    if (begin.has_stride_zero()) {
      // The output has a dimension with stride zero so parallelization must
      // be done differently. See parallelization in accumulate.h.
      auto indices = begin;
      auto end = std::move(begin);
      end.set_index(arg.size());
      run(indices, end);
    } else {
      auto run_parallel = [&](const auto &range) {
        auto indices = begin; // copy so that run doesn't modify begin
        indices.set_index(range.begin());
        auto end = begin;
        end.set_index(range.end());
        run(indices, end);
      };
      core::parallel::parallel_for(core::parallel::blocked_range(0, arg.size()),
                                   run_parallel);
    }
  }

  /// Recursion endpoint for do_transform_in_place.
  ///
  /// Calls transform_in_place_impl unless the output has no variance even
  /// though it should.
  template <class Op, class Tuple>
  static void do_transform_in_place(Op op, Tuple &&processed) {
    using namespace detail;
    std::apply(
        [&op](auto &&arg, auto &&... args) {
          if constexpr (check_all_or_none_variances<Op, decltype(arg),
                                                    decltype(args)...>) {
            throw except::VariancesError(
                "Expected either all or none of inputs to have variances.");
          } else {
            constexpr bool in_var_if_out_var = std::is_base_of_v<
                core::transform_flags::expect_in_variance_if_out_variance_t,
                Op>;
            constexpr bool arg_var =
                is_ValuesAndVariances_v<std::decay_t<decltype(arg)>>;
            constexpr bool args_var =
                (is_ValuesAndVariances_v<std::decay_t<decltype(args)>> || ...);
            if constexpr ((in_var_if_out_var ? arg_var == args_var
                                             : arg_var || !args_var) ||
                          std::is_base_of_v<core::transform_flags::
                                                expect_no_variance_arg_t<0>,
                                            Op>) {
              transform_in_place_impl(op, std::forward<decltype(arg)>(arg),
                                      std::forward<decltype(args)>(args)...);
            } else {
              throw except::VariancesError(
                  "Output has no variance but at least one input does.");
            }
          }
        },
        std::forward<Tuple>(processed));
  }

  /// Helper for in-place transform implementation, performing branching between
  /// output with and without variances as well as handling other operands with
  /// and without variances.
  template <class Op, class Tuple, class Arg, class... Args>
  static void do_transform_in_place(Op op, Tuple &&processed, Arg &arg,
                                    const Args &... args) {
    using namespace detail;
    auto vals = arg.values();
    if (arg.hasVariances()) {
      if constexpr (std::is_base_of_v<
                        core::transform_flags::expect_no_variance_arg_t<
                            std::tuple_size_v<Tuple>>,
                        Op>) {
        throw except::VariancesError("Variances in argument " +
                                     std::to_string(std::tuple_size_v<Tuple>) +
                                     " not supported.");
      } else if constexpr (core::canHaveVariances<typename Arg::value_type>()) {
        auto vars = arg.variances();
        do_transform_in_place(
            op,
            std::tuple_cat(processed,
                           std::tuple(ValuesAndVariances{vals, vars})),
            args...);
      }
    } else {
      if constexpr (std::is_base_of_v<
                        core::transform_flags::expect_variance_arg_t<
                            std::tuple_size_v<Tuple>>,
                        Op>) {
        throw except::VariancesError("Argument " +
                                     std::to_string(std::tuple_size_v<Tuple>) +
                                     " must have variances.");
      } else {
        do_transform_in_place(op, std::tuple_cat(processed, std::tuple(vals)),
                              args...);
      }
    }
  }

  /// Functor for in-place transformation, applying `op` to all elements.
  ///
  /// This is responsible for converting the client-provided functor `Op` which
  /// operates on elements to a functor for the data container, which is
  /// required by `visit`.
  template <class Op> struct TransformInPlace {
    Op op;
    template <class T, class... Ts>
    void operator()(T &&out, Ts &&... handles) const {
      using namespace detail;
      // If there is an overlap between lhs and rhs we copy the rhs before
      // applying the operation.
      if ((overlaps(out, handles) || ...)) {
        if constexpr (sizeof...(Ts) == 1) {
          auto copy = (handles.clone(), ...);
          return operator()(std::forward<T>(out), Ts(copy)...);
        } else {
          throw std::runtime_error(
              "Overlap handling only implemented for 2 inputs.");
        }
      }
      const auto dims = merge(out.dims(), handles.dims()...);
      auto out_view = as_view{out, dims};
      do_transform_in_place(op, std::tuple<>{}, out_view,
                            as_view{handles, dims}...);
    }
  };
  // gcc cannot deal with deduction guide for nested class => helper function.
  template <class Op> static auto makeTransformInPlace(Op op) {
    return TransformInPlace<Op>{detail::wrap_eigen{op}};
  }

  template <class... Ts, class Op, class Var, class... Other>
  static void transform_data(const std::tuple<Ts...> &, Op op,
                             const std::string_view name, Var &&var,
                             Other &&... other) {
    using namespace detail;
    try {
      visit<Ts...>::apply(makeTransformInPlace(op), var, other...);
    } catch (const std::bad_variant_access &) {
      throw except::TypeError("'" + std::string(name) +
                                  "' does not support dtypes ",
                              var, other...);
    }
  }
  template <class... Ts, class Op, class Var, class... Other>
  static void transform(Op op, const std::string_view name, Var &&var,
                        const Other &... other) {
    using namespace detail;
    (scipp::expect::includes(var.dims(), other.dims()), ...);
    auto unit = variableFactory().elem_unit(var);
    op(unit, variableFactory().elem_unit(other)...);
    // Stop early in bad cases of changing units (if `var` is a slice):
    variableFactory().expect_can_set_elem_unit(var, unit);
    // Wrapped implementation to convert multiple tuples into a parameter pack.
    transform_data(type_tuples<Ts...>(op), op, name, std::forward<Var>(var),
                   other...);
    if constexpr (dry_run)
      return;
    variableFactory().set_elem_unit(var, unit);
  }
};

/// Transform the data elements of a variable in-place.
///
/// Note that this is deliberately not named `for_each`: Unlike std::for_each,
/// this function does not promise in-order execution. This overload is
/// equivalent to std::transform with a single input range and an output range
/// identical to the input range, but avoids potentially costly element copies.
template <class... Ts, class Var, class Op>
void transform_in_place(Var &&var, Op op, const std::string_view name) {
  in_place<false>::transform<Ts...>(op, name, std::forward<Var>(var));
}

/// Transform the data elements of a variable in-place.
///
/// This overload is equivalent to std::transform with two input ranges and an
/// output range identical to the secound input range, but avoids potentially
/// costly element copies.
template <class... TypePairs, class Var, class Op>
void transform_in_place(Var &&var, const Variable &other, Op op,
                        const std::string_view name) {
  in_place<false>::transform<TypePairs...>(op, name, std::forward<Var>(var),
                                           other);
}

/// Transform the data elements of a variable in-place.
template <class... TypePairs, class Var, class Op>
void transform_in_place(Var &&var, const Variable &var1, const Variable &var2,
                        Op op, const std::string_view name) {
  in_place<false>::transform<TypePairs...>(op, name, std::forward<Var>(var),
                                           var1, var2);
}

/// Transform the data elements of a variable in-place.
template <class... TypePairs, class Var, class Op>
void transform_in_place(Var &&var, const Variable &var1, const Variable &var2,
                        const Variable &var3, Op op,
                        const std::string_view name) {
  in_place<false>::transform<TypePairs...>(op, name, std::forward<Var>(var),
                                           var1, var2, var3);
}

namespace dry_run {
template <class... Ts, class Var, class Op>
void transform_in_place(Var &&var, Op op, const std::string_view name) {
  in_place<true>::transform<Ts...>(op, name, std::forward<Var>(var));
}
template <class... TypePairs, class Var, class Op>
void transform_in_place(Var &&var, const Variable &other, Op op,
                        const std::string_view name) {
  in_place<true>::transform<TypePairs...>(op, name, std::forward<Var>(var),
                                          other);
}
} // namespace dry_run

namespace detail {
template <class... Ts, class Op, class... Vars>
Variable transform(std::tuple<Ts...> &&, Op op, const std::string_view name,
                   const Vars &... vars) {
  using namespace detail;
  try {
    return visit<Ts...>::apply(Transform{wrap_eigen{op}}, vars...);
  } catch (const std::bad_variant_access &) {
    throw except::TypeError(
        "'" + std::string(name) + "' does not support dtypes ", vars...);
  }
}
} // namespace detail

/// Transform the data elements of a variable and return a new Variable.
///
/// This overload is equivalent to std::transform with a single input range, but
/// avoids the need to manually create a new variable for the output and the
/// need for, e.g., std::back_inserter.
template <class... Ts, class Op>
[[nodiscard]] Variable transform(const Variable &var, Op op,
                                 const std::string_view name) {
  return detail::transform(type_tuples<Ts...>(op), op, name, var);
}

/// Transform the data elements of two variables and return a new Variable.
///
/// This overload is equivalent to std::transform with two input ranges, but
/// avoids the need to manually create a new variable for the output and the
/// need for, e.g., std::back_inserter.
template <class... Ts, class Op>
[[nodiscard]] Variable transform(const Variable &var1, const Variable &var2,
                                 Op op, const std::string_view name) {
  return detail::transform(type_tuples<Ts...>(op), op, name, var1, var2);
}

/// Transform the data elements of three variables and return a new Variable.
template <class... Ts, class Op>
[[nodiscard]] Variable transform(const Variable &var1, const Variable &var2,
                                 const Variable &var3, Op op,
                                 const std::string_view name) {
  return detail::transform(type_tuples<Ts...>(op), op, name, var1, var2, var3);
}

/// Transform the data elements of four variables and return a new Variable.
template <class... Ts, class Op>
[[nodiscard]] Variable transform(const Variable &var1, const Variable &var2,
                                 const Variable &var3, const Variable &var4,
                                 Op op, const std::string_view name) {
  return detail::transform(type_tuples<Ts...>(op), op, name, var1, var2, var3,
                           var4);
}

} // namespace scipp::variable
