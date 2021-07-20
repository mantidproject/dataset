// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>
#include <vector>

#include "test_macros.h"

#include "scipp/core/eigen.h"
#include "scipp/core/element/arg_list.h"

#include "scipp/variable/arithmetic.h"
#include "scipp/variable/bins.h"
#include "scipp/variable/shape.h"
#include "scipp/variable/transform.h"
#include "scipp/variable/util.h"
#include "scipp/variable/variable.h"

using namespace scipp;
using namespace scipp::core;
using namespace scipp::variable;

namespace {
const char *name = "transform_test";
template <typename T>
auto make_variable_for_test(const Shape &shape, bool variances) {
  auto ndim = shape.data.size();
  auto dims = ndim == 1 ? Dims{Dim::X}
                        : ndim == 2 ? Dims{Dim::X, Dim::Y}
                                    : Dims{Dim::X, Dim::Y, Dim::Z};
  auto var = variances ? makeVariable<T>(dims, shape, Values{}, Variances{})
                       : makeVariable<T>(dims, shape, Values{});

  const auto total_size = var.dims().volume();
  std::iota(var.template values<T>().begin(), var.template values<T>().end(),
            -total_size / 2.0);
  if (variances) {
    std::generate(var.template variances<T>().begin(),
                  var.template variances<T>().end(),
                  [x = -total_size / 20.0, total_size]() mutable {
                    return x += 10.0 / total_size;
                  });
  }
  return var;
}

// TODO irregular partitions
auto make_bin_indices(const scipp::index size, const scipp::index n_bins) {
  std::vector<index_pair> aux(n_bins);
  std::generate(begin(aux), end(aux),
                [lower = scipp::index{0}, size, n_bins]() mutable {
                  const scipp::index upper = static_cast<scipp::index>(
                      lower + (static_cast<double>(size) / n_bins));
                  const index_pair res{lower, upper};
                  lower = upper;
                  return res;
                });
  aux.back().second = size;
  return makeVariable<index_pair>(Dims{Dim::Event}, Shape{n_bins}, Values(aux));
}

const std::vector<Shape> shapes{Shape{1},       Shape{2},       Shape{3},
                                Shape{5},       Shape{16},      Shape{1, 1},
                                Shape{1, 2},    Shape{2, 8},    Shape{5, 7},
                                Shape{1, 1, 1}, Shape{1, 1, 4}, Shape{1, 5, 1},
                                Shape{7, 1, 1}, Shape{2, 8, 4}};

auto make_slices(const Shape &shape) {
  std::vector<Slice> res{Slice{Dim::X, 0, shape.data.at(0) - 1},
                         Slice{Dim::X, 0, shape.data.at(0) / 2}};
  if (shape.data.size() > 1 && shape.data.at(1) > 1) {
    res.emplace_back(Dim::Y, 0, shape.data.at(1) / 2);
  }
  if (shape.data.at(0) >= 2) {
    res.emplace_back(Dim::X, 2, shape.data.at(0));
  }
  return res;
}
} // namespace

class TransformUnaryTest : public ::testing::Test {
protected:
  static constexpr auto op_in_place{
      overloaded{[](auto &x) { x *= 2.0; }, [](units::Unit &) {}}};
  static constexpr auto op{
      overloaded{[](const auto x) { return x * 2.0; },
                 [](const units::Unit &unit) { return unit; }}};

  template <typename T>
  static auto op_manual_values(const ElementArrayView<T> values) {
    std::vector<double> res;
    res.reserve(values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(res), op);
    return res;
  }

  template <typename T>
  static auto op_manual_variances(const ElementArrayView<T> values,
                                  const ElementArrayView<T> variances) {
    std::vector<double> res;
    res.reserve(values.size());
    std::transform(
        values.begin(), values.end(), variances.begin(),
        std::back_inserter(res), [](auto value, auto variance) {
          return op(ValueAndVariance<double>{value, variance}).variance;
        });
    return res;
  }
};

TEST_F(TransformUnaryTest, dense) {
  for (const auto &shape : shapes) {
    for (bool variances : {false, true}) {
      const auto initial = make_variable_for_test<double>(shape, variances);

      const auto result_return = transform<double>(initial, op, name);
      Variable result_in_place = copy(initial);
      transform_in_place<double>(result_in_place, op_in_place, name);

      EXPECT_TRUE(equals(result_in_place.values<double>(),
                         op_manual_values(initial.values<double>())));
      if (variances) {
        EXPECT_TRUE(equals(result_in_place.variances<double>(),
                           op_manual_variances(initial.values<double>(),
                                               initial.variances<double>())));
      }
      // In-place transform used to check result of non-in-place transform.
      EXPECT_EQ(result_return, result_in_place);
    }
  }
}

TEST_F(TransformUnaryTest, slice) {
  for (const auto &shape : shapes) {
    if (shape.data.size() == 1) {
      continue; // cannot construct strided slices of 1D Variables
    }

    for (bool variances : {false, true}) {
      const auto initial_buffer =
          make_variable_for_test<double>(shape, variances);

      for (const Slice &slice : make_slices(shape)) {
        const auto initial = initial_buffer.slice(slice);

        const auto result_return = transform<double>(initial, op, name);
        Variable result_in_place_buffer = copy(initial_buffer);
        auto result_in_place = result_in_place_buffer.slice(slice);
        transform_in_place<double>(result_in_place, op_in_place, name);

        EXPECT_TRUE(equals(result_return.values<double>(),
                           op_manual_values(initial.values<double>())));
        if (variances) {
          EXPECT_TRUE(equals(result_return.variances<double>(),
                             op_manual_variances(initial.values<double>(),
                                                 initial.variances<double>())));
        }
        // In-place transform used to check result of non-in-place transform.
        EXPECT_EQ(result_return, result_in_place);
      }
    }
  }
}

TEST_F(TransformUnaryTest, elements_of_bins) {
  for (const auto &shape : shapes) {
    for (scipp::index bin_dim = 0;
         bin_dim < static_cast<scipp::index>(shape.data.size()); ++bin_dim) {
      for (scipp::index n_bins : {1, 2, 4, 5}) {
        if (n_bins > shape.data[bin_dim]) {
          continue;
        }
        for (bool variances : {false, true}) {
          const auto buffer = make_variable_for_test<double>(shape, variances);
          const auto bin_dim_label = buffer.dims().labels()[bin_dim];
          const auto indices = make_bin_indices(shape.data[bin_dim], n_bins);
          auto var = make_bins(indices, bin_dim_label, copy(buffer));

          const auto result = transform<double>(var, op, name);
          transform_in_place<double>(var, op_in_place, name);

          const auto expected = transform<double>(buffer, op, name);
          for (scipp::index bin = 0; bin < n_bins; ++bin) {
            const auto [lower, upper] = indices.values<index_pair>()[bin];
            EXPECT_EQ(var.values<bucket<Variable>>()[bin],
                      expected.slice({bin_dim_label, lower, upper}));
          }
          EXPECT_EQ(result, var);
        }
      }
    }
  }
}

TEST_F(TransformUnaryTest, in_place_unit_change) {
  const auto var =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{1.0, 2.0});
  const auto expected =
      makeVariable<double>(Dims{Dim::X}, Shape{2},
                           units::Unit(units::m * units::m), Values{1.0, 4.0});
  auto op_ = [](auto &&a) { a *= a; };
  Variable result;

  result = copy(var);
  transform_in_place<double>(result, op_, name);
  EXPECT_EQ(result, expected);

  // Unit changes but we are transforming only parts of data -> not possible.
  result = copy(var);
  EXPECT_THROW(transform_in_place<double>(result.slice({Dim::X, 1}), op_, name),
               except::UnitError);
}

TEST(TransformTest, apply_unary_implicit_conversion) {
  const auto var =
      makeVariable<float>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  // The functor returns double, so the output type is also double.
  auto out =
      transform<float>(var,
                       overloaded{[](const auto x) { return -1.0 * x; },
                                  [](const units::Unit &unit) { return unit; }},
                       name);
  EXPECT_TRUE(equals(out.values<double>(), {-1.1f, -2.2f}));
}

TEST(TransformTest, apply_unary_dtype_preserved) {
  const auto varD =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  const auto varF =
      makeVariable<float>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  auto outD = transform<double, float>(
      varD, [](const auto x) { return -x; }, name);
  auto outF = transform<double, float>(
      varF, [](const auto x) { return -x; }, name);
  EXPECT_TRUE(equals(outD.values<double>(), {-1.1, -2.2}));
  EXPECT_TRUE(equals(outF.values<float>(), {-1.1f, -2.2f}));
}

TEST(TransformTest, dtype_bool) {
  auto var = makeVariable<bool>(Dims{Dim::X}, Shape{2}, Values{true, false});

  EXPECT_EQ(transform<bool>(var,
                            overloaded{[](const units::Unit &u) { return u; },
                                       [](const auto x) { return !x; }},
                            name),
            makeVariable<bool>(Dims{Dim::X}, Shape{2}, Values{false, true}));

  EXPECT_EQ(transform<pair_self_t<bool>>(
                var, var,
                overloaded{
                    [](const units::Unit &a, const units::Unit &) { return a; },
                    [](const auto x, const auto y) { return !x || y; }},
                name),
            makeVariable<bool>(Dims{Dim::X}, Shape{2}, Values{true, true}));

  transform_in_place<bool>(
      var, overloaded{[](units::Unit &) {}, [](auto &x) { x = !x; }}, name);
  EXPECT_EQ(var,
            makeVariable<bool>(Dims{Dim::X}, Shape{2}, Values{false, true}));

  transform_in_place<pair_self_t<bool>>(
      var, var,
      overloaded{[](units::Unit &, const units::Unit &) {},
                 [](auto &x, const auto &y) { x = !x || y; }},
      name);
  EXPECT_EQ(var,
            makeVariable<bool>(Dims{Dim::X}, Shape{2}, Values{true, true}));
}

class TransformBinaryTest : public ::testing::Test {
protected:
  static constexpr auto op_in_place{[](auto &x, const auto &y) { x *= y; }};
  static constexpr auto op{[](const auto &x, const auto &y) { return x * y; }};
};

TEST_F(TransformBinaryTest, dense) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  const auto b = makeVariable<double>(Values{3.3});

  const auto ab = transform<pair_self_t<double>>(a, b, op, name);
  const auto ba = transform<pair_self_t<double>>(b, a, op, name);
  transform_in_place<pair_self_t<double>>(a, b, op_in_place, name);

  EXPECT_TRUE(equals(a.values<double>(), {1.1 * 3.3, 2.2 * 3.3}));
  EXPECT_EQ(ab, ba);
  EXPECT_EQ(ab, a);
  EXPECT_EQ(ba, a);
}

TEST_F(TransformBinaryTest, dims_and_shape_fail_in_place) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2});
  auto b = makeVariable<double>(Dims{Dim::Y}, Shape{2});
  auto c = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});

  EXPECT_ANY_THROW(
      transform_in_place<pair_self_t<double>>(a, b, op_in_place, name));
  EXPECT_ANY_THROW(
      transform_in_place<pair_self_t<double>>(a, c, op_in_place, name));
}

TEST_F(TransformBinaryTest, dims_and_shape_fail) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{4});
  auto b = makeVariable<double>(Dims{Dim::X}, Shape{2});
  auto c = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});

  EXPECT_ANY_THROW(auto v = transform<pair_self_t<double>>(a, b, op, name));
  EXPECT_ANY_THROW(auto v = transform<pair_self_t<double>>(a, c, op, name));
}

TEST_F(TransformBinaryTest, dense_mixed_type) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  const auto b = makeVariable<float>(Values{3.3});

  const auto ab =
      transform<pair_custom_t<std::tuple<double, float>>>(a, b, op, name);
  const auto ba =
      transform<pair_custom_t<std::tuple<float, double>>>(b, a, op, name);
  transform_in_place<pair_custom_t<std::tuple<double, float>>>(
      a, b, op_in_place, name);

  EXPECT_TRUE(equals(a.values<double>(), {1.1 * 3.3f, 2.2 * 3.3f}));
  EXPECT_EQ(ab, ba);
  EXPECT_EQ(ab, a);
  EXPECT_EQ(ba, a);
}

TEST_F(TransformBinaryTest, var_with_view) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  const auto b = makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{0.1, 3.3});

  auto ab = transform<pair_self_t<double>>(a, b.slice({Dim::Y, 1}), op, name);
  transform_in_place<pair_self_t<double>>(a, b.slice({Dim::Y, 1}), op_in_place,
                                          name);

  EXPECT_TRUE(equals(a.values<double>(), {1.1 * 3.3, 2.2 * 3.3}));
  EXPECT_EQ(ab, a);
}

TEST_F(TransformBinaryTest, in_place_self_overlap_without_variance_1d) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  auto reference = a * a.slice({Dim::X, 1});
  transform_in_place<pair_self_t<double>>(a, a.slice({Dim::X, 1}), op_in_place,
                                          name);
  ASSERT_EQ(a, reference);
}

TEST_F(TransformBinaryTest, in_place_self_overlap_without_variance_2d) {
  auto original = makeVariable<double>(Dimensions{{Dim::X, 2}, {Dim::Y, 2}},
                                       Values{1, 2, 3, 4});
  auto reference = makeVariable<double>(Dimensions{{Dim::X, 2}, {Dim::Y, 2}},
                                        Values{1, 6, 6, 16});
  Variable relabeled =
      fold(flatten(original, std::vector<Dim>{Dim::X, Dim::Y}, Dim::Z), Dim::Z,
           Dimensions{{Dim::Y, 2}, {Dim::X, 2}});
  ASSERT_EQ(original.data_handle(), relabeled.data_handle());
  ASSERT_NE(original.dims(), relabeled.dims());
  transform_in_place<pair_self_t<double>>(original, relabeled, op_in_place,
                                          name);
  ASSERT_EQ(original, reference);
}

TEST_F(TransformBinaryTest, in_place_self_overlap_with_variance) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2},
                                Variances{1.0, 2.0});
  Variable slice_copy(a.slice({Dim::X, 1}));
  auto reference = a * slice_copy;
  // With self-overlap the implementation needs to make a copy of the rhs. This
  // is a regression test: An initial implementation was unintentionally
  // dropping the variances when making that copy.
  transform_in_place<pair_self_t<double>>(a, a.slice({Dim::X, 1}), op_in_place,
                                          name);
  ASSERT_EQ(a, reference);
}

TEST_F(TransformBinaryTest, view_with_var) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  const auto b = makeVariable<double>(Values{3.3});

  transform_in_place<pair_self_t<double>>(a.slice({Dim::X, 1}), b, op_in_place,
                                          name);

  EXPECT_TRUE(equals(a.values<double>(), {1.1, 2.2 * 3.3}));
}

TEST_F(TransformBinaryTest, view_with_view) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  const auto b = makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{0.1, 3.3});

  transform_in_place<pair_self_t<double>>(
      a.slice({Dim::X, 1}), b.slice({Dim::Y, 1}), op_in_place, name);

  EXPECT_TRUE(equals(a.values<double>(), {1.1, 2.2 * 3.3}));
}

TEST_F(TransformBinaryTest, dense_events) {
  const auto indices = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::X}, Shape{2}, Values{std::pair{0, 3}, std::pair{3, 4}});
  const auto table =
      makeVariable<double>(Dims{Dim::Event}, Shape{4}, Values{1, 2, 3, 4});
  auto events = make_bins(indices, Dim::Event, copy(table));
  auto dense = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.5, 0.5});

  const auto ab = transform<pair_self_t<double>>(events, dense, op, name);
  const auto ba = transform<pair_self_t<double>>(dense, events, op, name);
  transform_in_place<pair_self_t<double>>(events, dense, op_in_place, name);

  const auto expected = transform<double>(events, dense, op, name);
  EXPECT_EQ(events.values<bucket<Variable>>()[0],
            transform<pair_self_t<double>>(dense.slice({Dim::X, 0}),
                                           table.slice({Dim::Event, 0, 3}), op,
                                           name));
  EXPECT_EQ(events.values<bucket<Variable>>()[1],
            transform<pair_self_t<double>>(dense.slice({Dim::X, 1}),
                                           table.slice({Dim::Event, 3, 4}), op,
                                           name));
  EXPECT_EQ(ab, events);
  EXPECT_EQ(ba, events);
}

TEST_F(TransformBinaryTest, events_size_fail) {
  const auto indicesA = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::X}, Shape{2}, Values{std::pair{0, 2}, std::pair{3, 4}});
  const auto indicesB = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::X}, Shape{2}, Values{std::pair{0, 3}, std::pair{3, 3}});
  const auto table = makeVariable<double>(Dims{Dim::Event}, Shape{4});
  auto a = make_bins(indicesA, Dim::Event, table);
  auto b = make_bins(indicesB, Dim::Event, table);
  ASSERT_THROW_DISCARD(transform<pair_self_t<double>>(a, b, op, name),
                       except::BinnedDataError);
  ASSERT_THROW(transform_in_place<pair_self_t<double>>(a, b, op_in_place, name),
               except::BinnedDataError);
}

TEST_F(TransformBinaryTest, in_place_unit_change) {
  const auto var =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{1.0, 2.0});
  const auto expected =
      makeVariable<double>(Dims{Dim::X}, Shape{2},
                           units::Unit(units::m * units::m), Values{1.0, 4.0});
  auto op_ = [](auto &&a, auto &&b) { a *= b; };
  Variable result;

  result = var;
  transform_in_place<pair_self_t<double>>(result, var, op_, name);
  EXPECT_EQ(result, expected);

  // Unit changes but we are transforming only parts of data -> not possible.
  result = var;
  EXPECT_THROW(
      transform_in_place<pair_self_t<double>>(
          result.slice({Dim::X, 1}), var.slice({Dim::X, 1}), op_, name),
      except::UnitError);
}

TEST(TransformTest, Eigen_Vector3d_pass_by_value) {
  const auto var = makeVariable<Eigen::Vector3d>(
      Dims{Dim::X}, Shape{2},
      Values{Eigen::Vector3d{1.1, 2.2, 3.3}, Eigen::Vector3d{0.1, 0.2, 0.3}});
  const auto expected = makeVariable<Eigen::Vector3d>(
      Dims(), Shape(), Values{Eigen::Vector3d{1.0, 2.0, 3.0}});
  // Passing Eigen types by value often causes issues, ensure that it works.
  auto op = [](const auto x, const auto y) { return x - y; };

  const auto result = transform<pair_self_t<Eigen::Vector3d>>(
      var.slice({Dim::X, 0}), var.slice({Dim::X, 1}), op, name);

  EXPECT_EQ(result, expected);
}

TEST(TransformTest, mixed_precision) {
  auto d = makeVariable<double>(Values{1e-12});
  auto f = makeVariable<float>(Values{1e-12});
  auto base_d = makeVariable<double>(Values{1.0});
  auto base_f = makeVariable<float>(Values{1.0});
  auto op = [](const auto a, const auto b) { return a + b; };
  const auto sum_fd =
      transform<pair_custom_t<std::tuple<float, double>>>(base_f, d, op, name);
  const auto sum_dd =
      transform<pair_custom_t<std::tuple<double, double>>>(base_d, d, op, name);
  EXPECT_NE(sum_fd.values<double>()[0], 1.0f);
  EXPECT_EQ(sum_fd.values<double>()[0], 1.0f + 1e-12);
  EXPECT_NE(sum_dd.values<double>()[0], 1.0);
  EXPECT_EQ(sum_dd.values<double>()[0], 1.0 + 1e-12);
  const auto sum_ff =
      transform<pair_custom_t<std::tuple<float, float>>>(base_f, f, op, name);
  const auto sum_df =
      transform<pair_custom_t<std::tuple<double, float>>>(base_d, f, op, name);
  EXPECT_EQ(sum_ff.values<float>()[0], 1.0f);
  EXPECT_EQ(sum_ff.values<float>()[0], 1.0f + 1e-12f);
  EXPECT_NE(sum_df.values<double>()[0], 1.0);
  EXPECT_EQ(sum_df.values<double>()[0], 1.0 + 1e-12f);
}

TEST(TransformTest, mixed_precision_in_place) {
  auto d = makeVariable<double>(Values{1e-12});
  auto f = makeVariable<float>(Values{1e-12});
  auto sum_d = makeVariable<double>(Values{1.0});
  auto sum_f = makeVariable<float>(Values{1.0});
  auto op = [](auto &a, const auto b) { a += b; };
  transform_in_place<pair_custom_t<std::tuple<float, double>>>(sum_f, d, op,
                                                               name);
  transform_in_place<pair_custom_t<std::tuple<double, double>>>(sum_d, d, op,
                                                                name);
  EXPECT_EQ(sum_f.values<float>()[0], 1.0f);
  EXPECT_NE(sum_d.values<double>()[0], 1.0);
  EXPECT_EQ(sum_d.values<double>()[0], 1.0 + 1e-12);
  transform_in_place<pair_custom_t<std::tuple<float, float>>>(sum_f, f, op,
                                                              name);
  transform_in_place<pair_custom_t<std::tuple<double, float>>>(sum_d, f, op,
                                                               name);
  EXPECT_EQ(sum_f.values<float>()[0], 1.0f);
  EXPECT_NE(sum_d.values<double>()[0], 1.0 + 1e-12);
  EXPECT_EQ(sum_d.values<double>()[0], 1.0 + 1e-12 + 1e-12);
}

TEST(TransformTest, combined_uncertainty_propagation) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{2.0}, Variances{0.1});
  auto a_2_step = copy(a);
  const auto b = makeVariable<double>(Values{3.0}, Variances{0.2});

  const auto abb = transform<pair_self_t<double>>(
      a, b, [](const auto &x, const auto &y) { return x * y + y; }, name);
  transform_in_place<pair_self_t<double>>(
      a, b, [](auto &x, const auto y) { x = x * y + y; }, name);
  transform_in_place<pair_self_t<double>>(
      a_2_step, b, [](auto &x, const auto y) { x = x * y; }, name);
  transform_in_place<pair_self_t<double>>(
      a_2_step, b, [](auto &x, const auto y) { x = x + y; }, name);

  EXPECT_TRUE(equals(a.values<double>(), {2.0 * 3.0 + 3.0}));
  EXPECT_TRUE(equals(a.variances<double>(), {0.1 * 3 * 3 + 0.2 * 2 * 2 + 0.2}));
  EXPECT_EQ(abb, a);
  EXPECT_EQ(abb, a_2_step);
}

class TransformTest_events_binary_values_variances_size_fail
    : public ::testing::Test {
protected:
  Variable indicesA = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::X}, Shape{2}, Values{std::pair{0, 2}, std::pair{2, 4}});
  Variable indicesB = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::X}, Shape{2}, Values{std::pair{0, 2}, std::pair{2, 3}});
  Variable table =
      makeVariable<double>(Dims{Dim::Event}, Shape{4}, Values{}, Variances{});
  Variable a = make_bins(indicesA, Dim::Event, table);
  Variable b = make_bins(indicesB, Dim::Event, table);
  Variable val_var = a;
  Variable val = make_bins(indicesA, Dim::Event, values(table));
  static constexpr auto op = [](const auto i, const auto j) { return i * j; };
  static constexpr auto op_in_place = [](auto &i, const auto j) { i *= j; };
};

TEST_F(TransformTest_events_binary_values_variances_size_fail, baseline) {
  ASSERT_NO_THROW_DISCARD(transform<pair_self_t<double>>(a, val_var, op, name));
  ASSERT_NO_THROW_DISCARD(transform<pair_self_t<double>>(a, val, op, name));
  ASSERT_NO_THROW(
      transform_in_place<pair_self_t<double>>(a, val_var, op_in_place, name));
  ASSERT_NO_THROW(
      transform_in_place<pair_self_t<double>>(a, val, op_in_place, name));
}

TEST_F(TransformTest_events_binary_values_variances_size_fail, a_size_bad) {
  a = b;
  ASSERT_THROW_DISCARD(transform<pair_self_t<double>>(a, val_var, op, name),
                       except::BinnedDataError);
  ASSERT_THROW_DISCARD(transform<pair_self_t<double>>(a, val, op, name),
                       except::BinnedDataError);
  ASSERT_THROW(
      transform_in_place<pair_self_t<double>>(a, val_var, op_in_place, name),
      except::BinnedDataError);
  ASSERT_THROW(
      transform_in_place<pair_self_t<double>>(a, val, op_in_place, name),
      except::BinnedDataError);
}

class TransformBinsBinaryTest : public TransformBinaryTest {
protected:
  Variable indicesY = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::Y}, Shape{2}, Values{std::pair{0, 3}, std::pair{3, 4}});
  Variable tableA =
      makeVariable<double>(Dims{Dim::Event}, Shape{4}, units::m,
                           Values{1, 2, 3, 4}, Variances{5, 6, 7, 8});
  Variable tableB = makeVariable<double>(Dims{Dim::Event}, Shape{4},
                                         Values{0.1, 0.2, 0.3, 0.4},
                                         Variances{0.5, 0.6, 0.7, 0.8});
  Variable a = make_bins(indicesY, Dim::Event, copy(tableA));
  Variable b = make_bins(indicesY, Dim::Event, copy(tableB));
};

TEST_F(TransformBinsBinaryTest, events_val_var_with_events_val_var) {
  const auto ab = transform<pair_self_t<double>>(a, b, op, name);
  transform_in_place<pair_self_t<double>>(a, b, op_in_place, name);
  // We rely on correctness of *dense* operations (Variable multiplication is
  // also built on transform).
  EXPECT_EQ(a, make_bins(indicesY, Dim::Event, tableA * tableB));
  EXPECT_EQ(ab, a);
}

TEST_F(TransformBinsBinaryTest, events_val_var_with_events_val) {
  const auto table = values(tableB);
  b = make_bins(indicesY, Dim::Event, table);
  const auto ab = transform<pair_self_t<double>>(a, b, op, name);
  transform_in_place<pair_self_t<double>>(a, b, op_in_place, name);
  EXPECT_EQ(a, make_bins(indicesY, Dim::Event, tableA * table));
  EXPECT_EQ(ab, a);
}

TEST_F(TransformBinsBinaryTest, events_val_var_with_val_var) {
  b = makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{1.5, 1.6},
                           Variances{1.7, 1.8});

  const auto ab = transform<pair_self_t<double>>(a, b, op, name);
  transform_in_place<pair_self_t<double>>(a, b, op_in_place, name);

  tableA.slice({Dim::Event, 0, 3}) *= b.slice({Dim::Y, 0});
  tableA.slice({Dim::Event, 3, 4}) *= b.slice({Dim::Y, 1});
  EXPECT_EQ(a, make_bins(indicesY, Dim::Event, tableA));
  EXPECT_EQ(ab, a);
}

TEST_F(TransformBinsBinaryTest, events_val_var_with_val) {
  b = makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{1.5, 1.6});

  const auto ab = transform<pair_self_t<double>>(a, b, op, name);
  transform_in_place<pair_self_t<double>>(a, b, op_in_place, name);

  tableA.slice({Dim::Event, 0, 3}) *= b.slice({Dim::Y, 0});
  tableA.slice({Dim::Event, 3, 4}) *= b.slice({Dim::Y, 1});
  EXPECT_EQ(a, make_bins(indicesY, Dim::Event, tableA));
  EXPECT_EQ(ab, a);
}

TEST_F(TransformBinsBinaryTest, broadcast_events_val_var_with_val) {
  b = makeVariable<float>(Dims{Dim::Z}, Shape{2}, Values{1.5, 1.6});

  const auto ab =
      transform<pair_custom_t<std::tuple<double, float>>>(a, b, op, name);

  EXPECT_EQ(ab.slice({Dim::Z, 0}),
            make_bins(indicesY, Dim::Event, tableA * b.slice({Dim::Z, 0})));
  EXPECT_EQ(ab.slice({Dim::Z, 1}),
            make_bins(indicesY, Dim::Event, tableA * b.slice({Dim::Z, 1})));
  EXPECT_EQ(ab.dims(), Dimensions({Dim::Y, Dim::Z}, {2, 2}));
}

// It is possible to use transform with functors that call non-built-in
// functions. To do so we have to define that function for the ValueAndVariance
// helper. If this turns out to be a useful feature we should move
// ValueAndVariance out of the `detail` namespace and document the mechanism.
constexpr auto user_op(const double) { return 123.0; }
constexpr auto user_op(const ValueAndVariance<double>) {
  return ValueAndVariance<double>{123.0, 456.0};
}
constexpr auto user_op(const units::Unit &) { return units::s; }

TEST(TransformTest, user_op_with_variances) {
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m,
                                  Values{1.1, 2.2}, Variances{1.1, 3.0});

  const auto result = transform<double>(
      var, [](auto x) { return user_op(x); }, name);
  transform_in_place<double>(
      var, [](auto &x) { x = user_op(x); }, name);

  auto expected = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::s,
                                       Values{123, 123}, Variances{456, 456});
  EXPECT_EQ(result, expected);
  EXPECT_EQ(result, var);
}

class TransformInPlaceDryRunTest : public ::testing::Test {
protected:
  static constexpr auto unary{[](auto &x) { x *= x; }};
  static constexpr auto binary{[](auto &x, const auto &y) { x *= y; }};
};

// Strictly speaking we should not have to test the failure cases --- even
// without dry-run, transform_in_place should not touch the data if there is a
// failure. Maybe this should be a parametrized test?
TEST_F(TransformInPlaceDryRunTest, unit_fail) {
  auto a = makeVariable<double>(Dims(), Shape(), units::m);
  const auto original(a);

  EXPECT_THROW(dry_run::transform_in_place<double>(
                   a, [](auto &x) { x += x * x; }, name),
               std::runtime_error);
  EXPECT_EQ(a, original);
  EXPECT_THROW(dry_run::transform_in_place<pair_self_t<double>>(
                   a, a * a, [](auto &x, const auto &y) { x += y; }, name),
               std::runtime_error);
  EXPECT_EQ(a, original);
}

TEST_F(TransformInPlaceDryRunTest, slice_unit_fail) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m);
  const auto original = copy(a);

  EXPECT_THROW(
      dry_run::transform_in_place<double>(a.slice({Dim::X, 0}), unary, name),
      except::UnitError);
  EXPECT_EQ(a, original);
  EXPECT_THROW(dry_run::transform_in_place<pair_self_t<double>>(
                   a.slice({Dim::X, 0}), a.slice({Dim::X, 0}), binary, name),
               except::UnitError);
  EXPECT_EQ(a, original);
}

TEST_F(TransformInPlaceDryRunTest, dimensions_fail) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m);
  auto b = makeVariable<double>(Dims{Dim::Y}, Shape{2}, units::m);
  const auto original = copy(a);

  EXPECT_THROW(
      dry_run::transform_in_place<pair_self_t<double>>(a, b, binary, name),
      except::NotFoundError);
  EXPECT_EQ(a, original);
}

TEST_F(TransformInPlaceDryRunTest, variances_fail) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m);
  auto b = makeVariable<double>(Dimensions{Dim::X, 2}, units::m, Values{},
                                Variances{});
  const auto original = copy(a);

  EXPECT_THROW(
      dry_run::transform_in_place<pair_self_t<double>>(a, b, binary, name),
      except::VariancesError);
  EXPECT_EQ(a, original);
}

class TransformInPlaceBucketsDryRunTest : public TransformInPlaceDryRunTest {
protected:
  Variable indicesA = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::X}, Shape{2}, Values{std::pair{0, 3}, std::pair{3, 4}});
  Variable indicesB = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::X}, Shape{2}, Values{std::pair{0, 3}, std::pair{3, 5}});
  Variable tableA =
      makeVariable<double>(Dims{Dim::Event}, Shape{4}, units::m,
                           Values{1, 2, 3, 4}, Variances{5, 6, 7, 8});
  Variable tableB =
      makeVariable<double>(Dims{Dim::Event}, Shape{5}, units::m,
                           Values{1, 2, 3, 4, 5}, Variances{5, 6, 7, 8, 9});
  Variable a = make_bins(indicesA, Dim::Event, tableA);
  Variable b = make_bins(indicesB, Dim::Event, tableB);
};

TEST_F(TransformInPlaceBucketsDryRunTest, events_length_fail) {
  const auto original(a);
  EXPECT_THROW(
      dry_run::transform_in_place<pair_self_t<double>>(a, b, binary, name),
      except::BinnedDataError);
  EXPECT_EQ(a, original);
}

TEST_F(TransformInPlaceBucketsDryRunTest, variances_fail) {
  a = make_bins(indicesA, Dim::Event, values(tableA));
  const auto original(a);
  EXPECT_THROW(
      dry_run::transform_in_place<pair_self_t<double>>(a, b, binary, name),
      except::VariancesError);
  EXPECT_EQ(a, original);
}

TEST_F(TransformInPlaceBucketsDryRunTest, unchanged_if_success) {
  const auto original(a);
  dry_run::transform_in_place<double>(a, unary, name);
  EXPECT_EQ(a, original);
  dry_run::transform_in_place<pair_self_t<double>>(a, a, binary, name);
  EXPECT_EQ(a, original);
}

TEST(TransformFlagsTest, no_variance_on_arg) {
  auto var_with_variance = makeVariable<double>(Values{1}, Variances{1});
  auto var_no_variance = makeVariable<double>(Values{1});
  constexpr auto binary_op = [](auto x, auto y) { return x + y; };
  constexpr auto op_arg_0_has_flags =
      scipp::overloaded{transform_flags::expect_variance_arg<0>, binary_op};

  Variable out;
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_with_variance, var_no_variance,
                                           op_arg_0_has_flags, name)));
  EXPECT_THROW(
      (out = transform<std::tuple<double>>(var_no_variance, var_with_variance,
                                           op_arg_0_has_flags, name)),
      except::VariancesError);
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_with_variance, var_with_variance,
                                           op_arg_0_has_flags, name)));
  constexpr auto op_arg_1_has_flags =
      scipp::overloaded{transform_flags::expect_variance_arg<1>, binary_op};
  EXPECT_THROW(
      (out = transform<std::tuple<double>>(var_with_variance, var_no_variance,
                                           op_arg_1_has_flags, name)),
      except::VariancesError);
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_no_variance, var_with_variance,
                                           op_arg_1_has_flags, name)));
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_with_variance, var_with_variance,
                                           op_arg_1_has_flags, name)));
  constexpr auto all_args_with_flag =
      scipp::overloaded{transform_flags::expect_variance_arg<0>,
                        transform_flags::expect_variance_arg<1>, binary_op};
  EXPECT_THROW(
      (out = transform<std::tuple<double>>(var_no_variance, var_no_variance,
                                           all_args_with_flag, name)),
      except::VariancesError);
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_with_variance, var_with_variance,
                                           all_args_with_flag, name)));
}

TEST(TransformFlagsTest, no_variance_on_arg_in_place) {
  auto var_with_variance = makeVariable<double>(Values{1}, Variances{1});
  auto var_no_variance = makeVariable<double>(Values{1});
  constexpr auto unary_in_place = [](auto &, auto) {};
  constexpr auto op_arg_0_has_flags = scipp::overloaded{
      transform_flags::expect_variance_arg<0>, unary_in_place};
  EXPECT_THROW(transform_in_place<std::tuple<double>>(
                   var_no_variance, var_no_variance, op_arg_0_has_flags, name),
               except::VariancesError);
  EXPECT_NO_THROW(transform_in_place<std::tuple<double>>(
      var_with_variance, var_with_variance, op_arg_0_has_flags, name));
  constexpr auto op_arg_1_has_flags = scipp::overloaded{
      transform_flags::expect_variance_arg<1>, unary_in_place};
  EXPECT_THROW(transform_in_place<std::tuple<double>>(
                   var_no_variance, var_no_variance, op_arg_1_has_flags, name),
               except::VariancesError);
  EXPECT_NO_THROW(transform_in_place<std::tuple<double>>(
      var_with_variance, var_with_variance, op_arg_1_has_flags, name));
}

TEST(TransformFlagsTest, variance_on_arg) {
  auto var_with_variance = makeVariable<double>(Values{1}, Variances{1});
  auto var_no_variance = makeVariable<double>(Values{1});
  constexpr auto binary_op = [](auto x, auto y) { return x + y; };
  constexpr auto op_arg_0_has_flags =
      scipp::overloaded{transform_flags::expect_no_variance_arg<0>, binary_op};
  Variable out;
  EXPECT_THROW(
      (out = transform<std::tuple<double>>(var_with_variance, var_no_variance,
                                           op_arg_0_has_flags, name)),
      except::VariancesError);
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_no_variance, var_with_variance,
                                           op_arg_0_has_flags, name)));
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_no_variance, var_no_variance,
                                           op_arg_0_has_flags, name)));
  constexpr auto op_arg_1_has_flags =
      scipp::overloaded{transform_flags::expect_no_variance_arg<1>, binary_op};
  EXPECT_THROW(
      (out = transform<std::tuple<double>>(var_no_variance, var_with_variance,
                                           op_arg_1_has_flags, name)),
      except::VariancesError);
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_with_variance, var_no_variance,
                                           op_arg_1_has_flags, name)));
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_no_variance, var_no_variance,
                                           op_arg_1_has_flags, name)));
  constexpr auto all_args_with_flag =
      scipp::overloaded{transform_flags::expect_no_variance_arg<0>,
                        transform_flags::expect_no_variance_arg<1>, binary_op};
  EXPECT_THROW(
      (out = transform<std::tuple<double>>(var_with_variance, var_with_variance,
                                           all_args_with_flag, name)),
      except::VariancesError);
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_no_variance, var_no_variance,
                                           all_args_with_flag, name)));
}

TEST(TransformFlagsTest, no_out_variance) {
  constexpr auto op =
      overloaded{transform_flags::no_out_variance, element::arg_list<double>,
                 [](const auto) { return true; },
                 [](const units::Unit &) { return units::one; }};
  const auto var = makeVariable<double>(Values{1.0}, Variances{1.0});
  EXPECT_EQ(transform(var, op, name), true * units::one);
}

TEST(TransformFlagsTest, variance_on_arg_in_place) {
  auto var_with_variance = makeVariable<double>(Values{1}, Variances{1});
  auto var_no_variance = makeVariable<double>(Values{1});
  constexpr auto unary_in_place = [](auto &, auto) {};
  constexpr auto op_arg_0_has_flags = scipp::overloaded{
      transform_flags::expect_no_variance_arg<0>, unary_in_place};
  EXPECT_THROW(transform_in_place<std::tuple<double>>(var_with_variance,
                                                      var_with_variance,
                                                      op_arg_0_has_flags, name),
               except::VariancesError);
  EXPECT_NO_THROW(transform_in_place<std::tuple<double>>(
      var_no_variance, var_no_variance, op_arg_0_has_flags, name));
  constexpr auto op_arg_1_has_flags = scipp::overloaded{
      transform_flags::expect_no_variance_arg<1>, unary_in_place};
  EXPECT_THROW(transform_in_place<std::tuple<double>>(var_with_variance,
                                                      var_with_variance,
                                                      op_arg_1_has_flags, name),
               except::VariancesError);
  EXPECT_NO_THROW(transform_in_place<std::tuple<double>>(
      var_no_variance, var_no_variance, op_arg_1_has_flags, name));
}

TEST(TransformFlagsTest, expect_in_variance_if_out_variance) {
  auto var_with_variance = makeVariable<double>(Values{1}, Variances{1});
  auto var_no_variance = makeVariable<double>(Values{1});
  constexpr auto inplace_op = [](auto &&x, const auto &y) { x += y; };
  constexpr auto op = overloaded{
      transform_flags::expect_in_variance_if_out_variance, inplace_op};
  EXPECT_THROW(transform_in_place<std::tuple<double>>(
                   var_with_variance, var_no_variance, op, name),
               except::VariancesError);
  EXPECT_THROW(transform_in_place<std::tuple<double>>(
                   var_no_variance, var_with_variance, op, name),
               except::VariancesError);
  EXPECT_NO_THROW(transform_in_place<std::tuple<double>>(
      var_no_variance, var_no_variance, op, name));
  EXPECT_NO_THROW(transform_in_place<std::tuple<double>>(
      var_with_variance, var_with_variance, op, name));
}

TEST(TransformFlagsTest, expect_all_or_none_have_variance) {
  auto var_with_variance = makeVariable<double>(Values{1}, Variances{1});
  auto var_no_variance = makeVariable<double>(Values{1});
  constexpr auto binary_op = [](auto x, auto y) { return x + y; };
  constexpr auto op_has_flags = scipp::overloaded{
      transform_flags::expect_all_or_none_have_variance, binary_op};
  Variable out;
  EXPECT_THROW((out = transform<std::tuple<double>>(
                    var_with_variance, var_no_variance, op_has_flags, name)),
               except::VariancesError);
  EXPECT_THROW((out = transform<std::tuple<double>>(
                    var_no_variance, var_with_variance, op_has_flags, name)),
               except::VariancesError);
  EXPECT_NO_THROW((out = transform<std::tuple<double>>(
                       var_no_variance, var_no_variance, op_has_flags, name)));
  EXPECT_NO_THROW(
      (out = transform<std::tuple<double>>(var_with_variance, var_with_variance,
                                           op_has_flags, name)));
}

TEST(TransformFlagsTest, expect_all_or_none_have_variance_in_place) {
  auto var_with_variance = makeVariable<double>(Values{1}, Variances{1});
  auto var_no_variance = makeVariable<double>(Values{1});
  constexpr auto unary_op = [](auto &, auto) {};
  constexpr auto op_has_flags = scipp::overloaded{
      transform_flags::expect_all_or_none_have_variance, unary_op};
  Variable out;
  EXPECT_THROW(transform_in_place<std::tuple<double>>(
                   var_with_variance, var_no_variance, op_has_flags, name),
               except::VariancesError);
  EXPECT_THROW(transform_in_place<std::tuple<double>>(
                   var_no_variance, var_with_variance, op_has_flags, name),
               except::VariancesError);
  EXPECT_NO_THROW(transform_in_place<std::tuple<double>>(
      var_no_variance, var_no_variance, op_has_flags, name));
  EXPECT_NO_THROW(transform_in_place<std::tuple<double>>(
      var_with_variance, var_with_variance, op_has_flags, name));
}

TEST(TransformFlagsTest, expect_no_in_variance_if_out_cannot_have_variance) {
  auto var_with_variance = makeVariable<double>(Values{1}, Variances{1});
  auto var_no_variance = makeVariable<double>(Values{1});
  constexpr auto unary_op = [](const auto) { return false; };
  constexpr auto op_has_flags = scipp::overloaded{
      element::arg_list<double>,
      transform_flags::expect_no_in_variance_if_out_cannot_have_variance,
      unary_op, [](const units::Unit &) { return units::one; }};
  Variable out;
  EXPECT_THROW(out = transform(var_with_variance, op_has_flags, name),
               except::VariancesError);
  EXPECT_NO_THROW(out = transform(var_no_variance, op_has_flags, name));
}

class TransformBinElementsTest : public ::testing::Test {
protected:
  Dimensions dims{Dim::Y, 2};
  Variable indices = makeVariable<std::pair<scipp::index, scipp::index>>(
      dims, Values{std::pair{0, 2}, std::pair{2, 4}});
  Variable buffer =
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  Variable var = make_bins(indices, Dim::X, copy(buffer));
};

TEST_F(TransformBinElementsTest, single_arg_in_place) {
  transform_in_place<double>(
      var,
      scipp::overloaded{transform_flags::expect_no_variance_arg<0>,
                        [](auto &x) { x *= x; }},
      name);
  Variable expected = make_bins(indices, Dim::X, buffer * buffer);
  EXPECT_EQ(var, expected);
}

TEST_F(TransformUnaryTest, drop_variances_when_not_supported_on_out_type) {
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2},
                                  Variances{1.1, 2.2});
  const auto result =
      transform<double>(var,
                        overloaded{[](const units::Unit &unit) { return unit; },
                                   [](const auto) { return true; }},
                        name);
  EXPECT_EQ(result,
            makeVariable<bool>(Dims{Dim::X}, Shape{2}, Values{true, true}));
}
