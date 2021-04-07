// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>
#include <vector>

#include <units/units.hpp>

#include "test_macros.h"

#include "scipp/core/except.h"
#include "scipp/variable/bins.h"
#include "scipp/variable/operations.h"
#include "scipp/variable/shape.h"
#include "scipp/variable/variable.h"

using namespace scipp;

TEST(Variable, construct_default) {
  ASSERT_NO_THROW(Variable{});
  Variable var;
  ASSERT_FALSE(var);
}

TEST(Variable, construct) {
  ASSERT_NO_THROW(makeVariable<double>(Dims{Dim::X}, Shape{2}));
  ASSERT_NO_THROW(makeVariable<double>(Dims{Dim::X}, Shape{2}, Values(2)));
  const auto a = makeVariable<double>(Dims{Dim::X}, Shape{2});
  const auto &data = a.values<double>();
  EXPECT_EQ(data.size(), 2);
}

TEST(Variable, construct_llnl_units_quantity) {
  EXPECT_EQ(Variable(1.2 * llnl::units::precise::meter),
            makeVariable<double>(Values{1.2}, units::m));
  // llnl measurement is always double
  EXPECT_EQ(Variable(1.0f * llnl::units::precise::meter),
            makeVariable<double>(Values{1.0}, units::m));
}

TEST(Variable, construct_fail) {
  ASSERT_ANY_THROW(makeVariable<double>(Dims{}, Shape{}, Values(2)));
  ASSERT_ANY_THROW(makeVariable<double>(Dims{Dim::X}, Shape{1}, Values(2)));
  ASSERT_ANY_THROW(makeVariable<double>(Dims{Dim::X}, Shape{3}, Values(2)));
}

TEST(Variable, move) {
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{2});
  Variable moved(std::move(var));
  EXPECT_FALSE(var);
  EXPECT_NE(moved, var);
}

TEST(Variable, makeVariable_custom_type) {
  auto doubles = makeVariable<double>(Values{double{}});
  auto floats = makeVariable<float>(Values{float{}});

  ASSERT_NO_THROW(doubles.values<double>());
  ASSERT_NO_THROW(floats.values<float>());

  ASSERT_ANY_THROW(doubles.values<float>());
  ASSERT_ANY_THROW(floats.values<double>());

  ASSERT_TRUE((std::is_same<decltype(doubles.values<double>())::element_type,
                            double>::value));
  ASSERT_TRUE((std::is_same<decltype(floats.values<float>())::element_type,
                            float>::value));
}

TEST(Variable, makeVariable_custom_type_initializer_list) {
  auto doubles = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1, 2});
  auto ints = makeVariable<int32_t>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  // Passed ints but uses default type based on tag.
  ASSERT_NO_THROW(doubles.values<double>());
  // Passed doubles but explicit type overrides.
  ASSERT_NO_THROW(ints.values<int32_t>());
}

TEST(Variable, dtype) {
  auto doubles = makeVariable<double>(Values{double{}});
  auto floats = makeVariable<float>(Values{float{}});
  EXPECT_EQ(doubles.dtype(), dtype<double>);
  EXPECT_NE(doubles.dtype(), dtype<float>);
  EXPECT_NE(floats.dtype(), dtype<double>);
  EXPECT_EQ(floats.dtype(), dtype<float>);
  EXPECT_EQ(doubles.dtype(), doubles.dtype());
  EXPECT_EQ(floats.dtype(), floats.dtype());
  EXPECT_NE(doubles.dtype(), floats.dtype());
}

TEST(Variable, span_references_Variable) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2});
  auto observer = a.values<double>();
  // This line does not compile, const-correctness works:
  // observer[0] = 1.0;

  auto span = a.values<double>();

  EXPECT_EQ(span.size(), 2);
  span[0] = 1.0;
  EXPECT_EQ(observer[0], 1.0);
}

class Variable_comparison_operators : public ::testing::Test {
private:
  template <class A, class B>
  void expect_eq_impl(const A &a, const B &b) const {
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(b == a);
    EXPECT_FALSE(a != b);
    EXPECT_FALSE(b != a);
  }
  template <class A, class B>
  void expect_ne_impl(const A &a, const B &b) const {
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(b != a);
    EXPECT_FALSE(a == b);
    EXPECT_FALSE(b == a);
  }

protected:
  void expect_eq(const Variable &a, const Variable &b) const {
    expect_eq_impl(a, VariableConstView(b));
    expect_eq_impl(VariableConstView(a), b);
    expect_eq_impl(VariableConstView(a), VariableConstView(b));
  }
  void expect_ne(const Variable &a, const Variable &b) const {
    expect_ne_impl(a, VariableConstView(b));
    expect_ne_impl(VariableConstView(a), b);
    expect_ne_impl(VariableConstView(a), VariableConstView(b));
  }
};

TEST_F(Variable_comparison_operators, values_0d) {
  const auto base = makeVariable<double>(Values{1.1});
  expect_eq(base, base);
  expect_eq(base, makeVariable<double>(Values{1.1}));
  expect_ne(base, makeVariable<double>(Values{1.2}));
}

TEST_F(Variable_comparison_operators, values_1d) {
  const auto base =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});
  expect_eq(base, base);
  expect_eq(base,
            makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2}));
  expect_ne(base,
            makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.3}));
}

TEST_F(Variable_comparison_operators, values_2d) {
  const auto base =
      makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 1}, Values{1.1, 2.2});
  expect_eq(base, base);
  expect_eq(base, makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 1},
                                       Values{1.1, 2.2}));
  expect_ne(base, makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 1},
                                       Values{1.1, 2.3}));
}

TEST_F(Variable_comparison_operators, variances_0d) {
  const auto base = makeVariable<double>(Values{1.1}, Variances{0.1});
  expect_eq(base, base);
  expect_eq(base, makeVariable<double>(Values{1.1}, Variances{0.1}));
  expect_ne(base, makeVariable<double>(Values{1.1}));
  expect_ne(base, makeVariable<double>(Values{1.1}, Variances{0.2}));
}

TEST_F(Variable_comparison_operators, variances_1d) {
  const auto base = makeVariable<double>(Dims{Dim::X}, Shape{2},
                                         Values{1.1, 2.2}, Variances{0.1, 0.2});
  expect_eq(base, base);
  expect_eq(base, makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2},
                                       Variances{0.1, 0.2}));
  expect_ne(base,
            makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2}));
  expect_ne(base, makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2},
                                       Variances{0.1, 0.3}));
}

TEST_F(Variable_comparison_operators, variances_2d) {
  const auto base = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 1},
                                         Values{1.1, 2.2}, Variances{0.1, 0.2});
  expect_eq(base, base);
  expect_eq(base, makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 1},
                                       Values{1.1, 2.2}, Variances{0.1, 0.2}));
  expect_ne(base, makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 1},
                                       Values{1.1, 2.2}));
  expect_ne(base, makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 1},
                                       Values{1.1, 2.2}, Variances{0.1, 0.3}));
}

TEST_F(Variable_comparison_operators, dimension_mismatch) {
  expect_ne(makeVariable<double>(Values{1.1}),
            makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.1}));
  expect_ne(makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.1}),
            makeVariable<double>(Dims{Dim::Y}, Shape{1}, Values{1.1}));
}

TEST_F(Variable_comparison_operators, dimension_transpose) {
  expect_ne(
      makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{1, 1}, Values{1.1}),
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{1, 1}, Values{1.1}));
}

TEST_F(Variable_comparison_operators, dimension_length) {
  expect_ne(makeVariable<double>(Dims{Dim::X}, Shape{1}),
            makeVariable<double>(Dims{Dim::X}, Shape{2}));
}

TEST_F(Variable_comparison_operators, unit) {
  const auto m =
      makeVariable<double>(Dims{Dim::X}, Shape{1}, units::m, Values{1.1});
  const auto s =
      makeVariable<double>(Dims{Dim::X}, Shape{1}, units::s, Values{1.1});
  expect_eq(m, m);
  expect_ne(m, s);
}

TEST_F(Variable_comparison_operators, dtype) {
  const auto base = makeVariable<double>(Values{1.0});
  expect_ne(base, makeVariable<float>(Values{1.0}));
}

TEST_F(Variable_comparison_operators, dense_events) {
  Dimensions dims{Dim::Y, 2};
  Variable indices = makeVariable<std::pair<scipp::index, scipp::index>>(
      dims, Values{std::pair{0, 2}, std::pair{2, 4}});
  auto buf = makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  auto events = make_bins(indices, Dim::X, buf);
  auto dense = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2l, 0l});
  expect_ne(dense, events);
}

TEST_F(Variable_comparison_operators, events) {
  Dimensions dims{Dim::Y, 2};
  Variable indices = makeVariable<std::pair<scipp::index, scipp::index>>(
      dims, Values{std::pair{0, 2}, std::pair{2, 4}});
  Variable indices2 = makeVariable<std::pair<scipp::index, scipp::index>>(
      dims, Values{std::pair{0, 3}, std::pair{3, 4}});
  auto buf = makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  auto buf_with_vars = makeVariable<double>(Dims{Dim::X}, Shape{4},
                                            Values{1, 2, 3, 4}, Variances{});
  auto a = make_bins(indices, Dim::X, buf);
  auto b = make_bins(indices, Dim::X, buf);
  auto c = make_bins(indices, Dim::X, buf * (2.0 * units::one));
  auto d = make_bins(indices2, Dim::X, buf);
  auto a_with_vars = make_bins(indices, Dim::X, buf_with_vars);

  expect_eq(a, a);
  expect_eq(a, b);
  expect_ne(a, c);
  expect_ne(a, d);
  expect_ne(a, a_with_vars);
}

TEST(VariableTest, copy_and_move) {
  const auto reference =
      makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 1}, units::m,
                           Values{1.1, 2.2}, Variances{0.1, 0.2});
  const auto var =
      makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 1}, units::m,
                           Values{1.1, 2.2}, Variances{0.1, 0.2});

  const auto copy(var);
  EXPECT_EQ(copy, reference);

  const Variable copy_via_slice{VariableConstView(var)};
  EXPECT_EQ(copy_via_slice, reference);

  const auto moved(std::move(var));
  EXPECT_EQ(moved, reference);
}

TEST(Variable, assign_slice) {
  const auto parent = makeVariable<double>(
      Dims{Dim::X, Dim::Y, Dim::Z}, Shape{4, 2, 3},
      Values{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
      Variances{25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48});
  const auto empty = makeVariable<double>(
      Dimensions{{Dim::X, 4}, {Dim::Y, 2}, {Dim::Z, 3}}, Values{}, Variances{});

  auto d = copy(empty);
  EXPECT_NE(parent, d);
  for (const scipp::index index : {0, 1, 2, 3})
    d.slice({Dim::X, index}).assign(parent.slice({Dim::X, index}));
  EXPECT_EQ(parent, d);

  d = copy(empty);
  EXPECT_NE(parent, d);
  for (const scipp::index index : {0, 1})
    d.slice({Dim::Y, index}).assign(parent.slice({Dim::Y, index}));
  EXPECT_EQ(parent, d);

  d = copy(empty);
  EXPECT_NE(parent, d);
  for (const scipp::index index : {0, 1, 2})
    d.slice({Dim::Z, index}).assign(parent.slice({Dim::Z, index}));
  EXPECT_EQ(parent, d);
}

TEST(Variable, assign_slice_unit_checks) {
  const auto parent =
      makeVariable<double>(Dims(), Shape(), units::m, Values{1});
  auto dimensionless = makeVariable<double>(Dims{Dim::X}, Shape{4});
  auto m = makeVariable<double>(Dims{Dim::X}, Shape{4}, units::m);

  EXPECT_THROW(dimensionless.slice({Dim::X, 1}).assign(parent),
               except::UnitError);
  EXPECT_NO_THROW(m.slice({Dim::X, 1}).assign(parent));
}

TEST(Variable, assign_slice_variance_checks) {
  const auto parent_vals = makeVariable<double>(Values{1.0});
  const auto parent_vals_vars =
      makeVariable<double>(Values{1.0}, Variances{2.0});
  auto vals = makeVariable<double>(Dims{Dim::X}, Shape{4});
  auto vals_vars =
      makeVariable<double>(Dimensions{Dim::X, 4}, Values{}, Variances{});

  EXPECT_NO_THROW(vals.slice({Dim::X, 1}).assign(parent_vals));
  EXPECT_NO_THROW(vals_vars.slice({Dim::X, 1}).assign(parent_vals_vars));
  EXPECT_THROW(vals.slice({Dim::X, 1}).assign(parent_vals_vars),
               except::VariancesError);
  EXPECT_THROW(vals_vars.slice({Dim::X, 1}).assign(parent_vals),
               except::VariancesError);
}

class VariableTest_3d : public ::testing::Test {
protected:
  const Variable parent{makeVariable<double>(
      Dims{Dim::X, Dim::Y, Dim::Z}, Shape{4, 2, 3}, units::m,
      Values{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
      Variances{25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48})};
  const std::vector<double> vals_x0{1, 2, 3, 4, 5, 6};
  const std::vector<double> vals_x1{7, 8, 9, 10, 11, 12};
  const std::vector<double> vals_x2{13, 14, 15, 16, 17, 18};
  const std::vector<double> vals_x3{19, 20, 21, 22, 23, 24};
  const std::vector<double> vars_x0{25, 26, 27, 28, 29, 30};
  const std::vector<double> vars_x1{31, 32, 33, 34, 35, 36};
  const std::vector<double> vars_x2{37, 38, 39, 40, 41, 42};
  const std::vector<double> vars_x3{43, 44, 45, 46, 47, 48};

  const std::vector<double> vals_x02{
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
  };
  const std::vector<double> vals_x13{7,  8,  9,  10, 11, 12,
                                     13, 14, 15, 16, 17, 18};
  const std::vector<double> vals_x24{13, 14, 15, 16, 17, 18,
                                     19, 20, 21, 22, 23, 24};
  const std::vector<double> vars_x02{25, 26, 27, 28, 29, 30,
                                     31, 32, 33, 34, 35, 36};
  const std::vector<double> vars_x13{31, 32, 33, 34, 35, 36,
                                     37, 38, 39, 40, 41, 42};
  const std::vector<double> vars_x24{37, 38, 39, 40, 41, 42,
                                     43, 44, 45, 46, 47, 48};

  const std::vector<double> vals_y0{1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21};
  const std::vector<double> vals_y1{4,  5,  6,  10, 11, 12,
                                    16, 17, 18, 22, 23, 24};
  const std::vector<double> vars_y0{25, 26, 27, 31, 32, 33,
                                    37, 38, 39, 43, 44, 45};
  const std::vector<double> vars_y1{28, 29, 30, 34, 35, 36,
                                    40, 41, 42, 46, 47, 48};

  const std::vector<double> vals_z0{1, 4, 7, 10, 13, 16, 19, 22};
  const std::vector<double> vals_z1{2, 5, 8, 11, 14, 17, 20, 23};
  const std::vector<double> vals_z2{3, 6, 9, 12, 15, 18, 21, 24};
  const std::vector<double> vars_z0{25, 28, 31, 34, 37, 40, 43, 46};
  const std::vector<double> vars_z1{26, 29, 32, 35, 38, 41, 44, 47};
  const std::vector<double> vars_z2{27, 30, 33, 36, 39, 42, 45, 48};

  const std::vector<double> vals_z02{1,  2,  4,  5,  7,  8,  10, 11,
                                     13, 14, 16, 17, 19, 20, 22, 23};
  const std::vector<double> vals_z13{2,  3,  5,  6,  8,  9,  11, 12,
                                     14, 15, 17, 18, 20, 21, 23, 24};
  const std::vector<double> vars_z02{25, 26, 28, 29, 31, 32, 34, 35,
                                     37, 38, 40, 41, 43, 44, 46, 47};
  const std::vector<double> vars_z13{26, 27, 29, 30, 32, 33, 35, 36,
                                     38, 39, 41, 42, 44, 45, 47, 48};
};

TEST_F(VariableTest_3d, slice_single) {
  Dimensions dims_no_x{{Dim::Y, 2}, {Dim::Z, 3}};
  EXPECT_EQ(parent.slice({Dim::X, 0}),
            makeVariable<double>(Dimensions(dims_no_x), units::m,
                                 Values(vals_x0.begin(), vals_x0.end()),
                                 Variances(vars_x0.begin(), vars_x0.end())));
  EXPECT_EQ(parent.slice({Dim::X, 1}),
            makeVariable<double>(Dimensions(dims_no_x), units::m,
                                 Values(vals_x1.begin(), vals_x1.end()),
                                 Variances(vars_x1.begin(), vars_x1.end())));
  EXPECT_EQ(parent.slice({Dim::X, 2}),
            makeVariable<double>(Dimensions(dims_no_x), units::m,
                                 Values(vals_x2.begin(), vals_x2.end()),
                                 Variances(vars_x2.begin(), vars_x2.end())));
  EXPECT_EQ(parent.slice({Dim::X, 3}),
            makeVariable<double>(Dimensions(dims_no_x), units::m,
                                 Values(vals_x3.begin(), vals_x3.end()),
                                 Variances(vars_x3.begin(), vars_x3.end())));

  Dimensions dims_no_y{{Dim::X, 4}, {Dim::Z, 3}};
  EXPECT_EQ(parent.slice({Dim::Y, 0}),
            makeVariable<double>(Dimensions(dims_no_y), units::m,
                                 Values(vals_y0.begin(), vals_y0.end()),
                                 Variances(vars_y0.begin(), vars_y0.end())));
  EXPECT_EQ(parent.slice({Dim::Y, 1}),
            makeVariable<double>(Dimensions(dims_no_y), units::m,
                                 Values(vals_y1.begin(), vals_y1.end()),
                                 Variances(vars_y1.begin(), vars_y1.end())));

  Dimensions dims_no_z{{Dim::X, 4}, {Dim::Y, 2}};
  EXPECT_EQ(parent.slice({Dim::Z, 0}),
            makeVariable<double>(Dimensions(dims_no_z), units::m,
                                 Values(vals_z0.begin(), vals_z0.end()),
                                 Variances(vars_z0.begin(), vars_z0.end())));
  EXPECT_EQ(parent.slice({Dim::Z, 1}),
            makeVariable<double>(Dimensions(dims_no_z), units::m,
                                 Values(vals_z1.begin(), vals_z1.end()),
                                 Variances(vars_z1.begin(), vars_z1.end())));
  EXPECT_EQ(parent.slice({Dim::Z, 2}),
            makeVariable<double>(Dimensions(dims_no_z), units::m,
                                 Values(vals_z2.begin(), vals_z2.end()),
                                 Variances(vars_z2.begin(), vars_z2.end())));
}

TEST_F(VariableTest_3d, slice_range) {
  // Length 1 slice
  Dimensions dims_x1{{Dim::X, 1}, {Dim::Y, 2}, {Dim::Z, 3}};
  EXPECT_EQ(parent.slice({Dim::X, 0, 1}),
            makeVariable<double>(Dimensions(dims_x1), units::m,
                                 Values(vals_x0.begin(), vals_x0.end()),
                                 Variances(vars_x0.begin(), vars_x0.end())));
  EXPECT_EQ(parent.slice({Dim::X, 1, 2}),
            makeVariable<double>(Dimensions(dims_x1), units::m,
                                 Values(vals_x1.begin(), vals_x1.end()),
                                 Variances(vars_x1.begin(), vars_x1.end())));
  EXPECT_EQ(parent.slice({Dim::X, 2, 3}),
            makeVariable<double>(Dimensions(dims_x1), units::m,
                                 Values(vals_x2.begin(), vals_x2.end()),
                                 Variances(vars_x2.begin(), vars_x2.end())));
  EXPECT_EQ(parent.slice({Dim::X, 3, 4}),
            makeVariable<double>(Dimensions(dims_x1), units::m,
                                 Values(vals_x3.begin(), vals_x3.end()),
                                 Variances(vars_x3.begin(), vars_x3.end())));

  Dimensions dims_y1{{Dim::X, 4}, {Dim::Y, 1}, {Dim::Z, 3}};
  EXPECT_EQ(parent.slice({Dim::Y, 0, 1}),
            makeVariable<double>(Dimensions(dims_y1), units::m,
                                 Values(vals_y0.begin(), vals_y0.end()),
                                 Variances(vars_y0.begin(), vars_y0.end())));
  EXPECT_EQ(parent.slice({Dim::Y, 1, 2}),
            makeVariable<double>(Dimensions(dims_y1), units::m,
                                 Values(vals_y1.begin(), vals_y1.end()),
                                 Variances(vars_y1.begin(), vars_y1.end())));

  Dimensions dims_z1{{Dim::X, 4}, {Dim::Y, 2}, {Dim::Z, 1}};
  EXPECT_EQ(parent.slice({Dim::Z, 0, 1}),
            makeVariable<double>(Dimensions(dims_z1), units::m,
                                 Values(vals_z0.begin(), vals_z0.end()),
                                 Variances(vars_z0.begin(), vars_z0.end())));
  EXPECT_EQ(parent.slice({Dim::Z, 1, 2}),
            makeVariable<double>(Dimensions(dims_z1), units::m,
                                 Values(vals_z1.begin(), vals_z1.end()),
                                 Variances(vars_z1.begin(), vars_z1.end())));
  EXPECT_EQ(parent.slice({Dim::Z, 2, 3}),
            makeVariable<double>(Dimensions(dims_z1), units::m,
                                 Values(vals_z2.begin(), vals_z2.end()),
                                 Variances(vars_z2.begin(), vars_z2.end())));

  // Length 2 slice
  Dimensions dims_x2{{Dim::X, 2}, {Dim::Y, 2}, {Dim::Z, 3}};
  EXPECT_EQ(parent.slice({Dim::X, 0, 2}),
            makeVariable<double>(Dimensions(dims_x2), units::m,
                                 Values(vals_x02.begin(), vals_x02.end()),
                                 Variances(vars_x02.begin(), vars_x02.end())));
  EXPECT_EQ(parent.slice({Dim::X, 1, 3}),
            makeVariable<double>(Dimensions(dims_x2), units::m,
                                 Values(vals_x13.begin(), vals_x13.end()),
                                 Variances(vars_x13.begin(), vars_x13.end())));
  EXPECT_EQ(parent.slice({Dim::X, 2, 4}),
            makeVariable<double>(Dimensions(dims_x2), units::m,
                                 Values(vals_x24.begin(), vals_x24.end()),
                                 Variances(vars_x24.begin(), vars_x24.end())));

  EXPECT_EQ(parent.slice({Dim::Y, 0, 2}), parent);

  Dimensions dims_z2{{Dim::X, 4}, {Dim::Y, 2}, {Dim::Z, 2}};
  EXPECT_EQ(parent.slice({Dim::Z, 0, 2}),
            makeVariable<double>(Dimensions(dims_z2), units::m,
                                 Values(vals_z02.begin(), vals_z02.end()),
                                 Variances(vars_z02.begin(), vars_z02.end())));
  EXPECT_EQ(parent.slice({Dim::Z, 1, 3}),
            makeVariable<double>(Dimensions(dims_z2), units::m,
                                 Values(vals_z13.begin(), vals_z13.end()),
                                 Variances(vars_z13.begin(), vars_z13.end())));
}

TEST(VariableView, full_const_view) {
  const auto var =
      makeVariable<double>(Dimensions{Dim::X, 3}, Values{}, Variances{});
  VariableConstView view(var);
  EXPECT_EQ(&var.values<double>()[0], &view.values<double>()[0]);
  EXPECT_EQ(&var.variances<double>()[0], &view.variances<double>()[0]);
}

TEST(VariableView, full_mutable_view) {
  auto var = makeVariable<double>(Dimensions{Dim::X, 3}, Values{}, Variances{});
  VariableView view(var);
  EXPECT_EQ(&var.values<double>()[0], &view.values<double>()[0]);
  EXPECT_EQ(&var.variances<double>()[0], &view.variances<double>()[0]);
}

TEST(VariableView, strides) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
  EXPECT_EQ(var.slice({Dim::X, 0}).strides(), (std::vector<scipp::index>{3}));
  EXPECT_EQ(var.slice({Dim::X, 1}).strides(), (std::vector<scipp::index>{3}));
  EXPECT_EQ(var.slice({Dim::Y, 0}).strides(), (std::vector<scipp::index>{1}));
  EXPECT_EQ(var.slice({Dim::Y, 1}).strides(), (std::vector<scipp::index>{1}));
  EXPECT_EQ(var.slice({Dim::X, 0, 1}).strides(),
            (std::vector<scipp::index>{3, 1}));
  EXPECT_EQ(var.slice({Dim::X, 1, 2}).strides(),
            (std::vector<scipp::index>{3, 1}));
  EXPECT_EQ(var.slice({Dim::Y, 0, 1}).strides(),
            (std::vector<scipp::index>{3, 1}));
  EXPECT_EQ(var.slice({Dim::Y, 1, 2}).strides(),
            (std::vector<scipp::index>{3, 1}));
  EXPECT_EQ(var.slice({Dim::X, 0, 2}).strides(),
            (std::vector<scipp::index>{3, 1}));
  EXPECT_EQ(var.slice({Dim::X, 1, 3}).strides(),
            (std::vector<scipp::index>{3, 1}));
  EXPECT_EQ(var.slice({Dim::Y, 0, 2}).strides(),
            (std::vector<scipp::index>{3, 1}));
  EXPECT_EQ(var.slice({Dim::Y, 1, 3}).strides(),
            (std::vector<scipp::index>{3, 1}));

  EXPECT_EQ(var.slice({Dim::X, 0, 1}).slice({Dim::Y, 0, 1}).strides(),
            (std::vector<scipp::index>{3, 1}));

  auto var3D =
      makeVariable<double>(Dims{Dim::Z, Dim::Y, Dim::X}, Shape{4, 3, 2});
  EXPECT_EQ(var3D.slice({Dim::X, 0, 1}).slice({Dim::Z, 0, 1}).strides(),
            (std::vector<scipp::index>{6, 2, 1}));
}

TEST(VariableView, values_and_variances) {
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3},
                                        Variances{4, 5, 6});
  const auto view = var.slice({Dim::X, 1, 2});
  EXPECT_EQ(view.values<double>().size(), 1);
  EXPECT_EQ(view.values<double>()[0], 2.0);
  EXPECT_EQ(view.variances<double>().size(), 1);
  EXPECT_EQ(view.variances<double>()[0], 5.0);
}

TEST(VariableView, slicing_does_not_transpose) {
  auto var = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{3, 3});
  Dimensions expected{{Dim::X, 1}, {Dim::Y, 1}};
  EXPECT_EQ(var.slice({Dim::X, 1, 2}).slice({Dim::Y, 1, 2}).dims(), expected);
  EXPECT_EQ(var.slice({Dim::Y, 1, 2}).slice({Dim::X, 1, 2}).dims(), expected);
}

TEST(VariableView, variable_copy_from_slice) {
  const auto source =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3}, units::m,
                           Values{11, 12, 13, 21, 22, 23, 31, 32, 33},
                           Variances{44, 45, 46, 54, 55, 56, 64, 65, 66});

  const Dimensions dims({{Dim::Y, 2}, {Dim::X, 2}});
  EXPECT_EQ(source.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2}),
            makeVariable<double>(Dimensions(dims), units::m,
                                 Values{11, 12, 21, 22},
                                 Variances{44, 45, 54, 55}));

  EXPECT_EQ(source.slice({Dim::X, 1, 3}).slice({Dim::Y, 0, 2}),
            makeVariable<double>(Dimensions(dims), units::m,
                                 Values{12, 13, 22, 23},
                                 Variances{45, 46, 55, 56}));

  EXPECT_EQ(source.slice({Dim::X, 0, 2}).slice({Dim::Y, 1, 3}),
            makeVariable<double>(Dimensions(dims), units::m,
                                 Values{21, 22, 31, 32},
                                 Variances{54, 55, 64, 65}));

  EXPECT_EQ(source.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3}),
            makeVariable<double>(Dimensions(dims), units::m,
                                 Values{22, 23, 32, 33},
                                 Variances{55, 56, 65, 66}));
}

TEST(VariableView, variable_assign_from_slice) {
  const Dimensions dims({{Dim::Y, 2}, {Dim::X, 2}});
  // Unit is dimensionless and no variance
  auto target = makeVariable<double>(Dimensions(dims), Values{1, 2, 3, 4});
  const auto source =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3}, units::m,
                           Values{11, 12, 13, 21, 22, 23, 31, 32, 33},
                           Variances{44, 45, 46, 54, 55, 56, 64, 65, 66});

  target = Variable(source.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2}));
  EXPECT_EQ(target, makeVariable<double>(Dimensions(dims), units::m,
                                         Values{11, 12, 21, 22},
                                         Variances{44, 45, 54, 55}));

  target = Variable(source.slice({Dim::X, 1, 3}).slice({Dim::Y, 0, 2}));
  EXPECT_EQ(target, makeVariable<double>(Dimensions(dims), units::m,
                                         Values{12, 13, 22, 23},
                                         Variances{45, 46, 55, 56}));

  target = Variable(source.slice({Dim::X, 0, 2}).slice({Dim::Y, 1, 3}));
  EXPECT_EQ(target, makeVariable<double>(Dimensions(dims), units::m,
                                         Values{21, 22, 31, 32},
                                         Variances{54, 55, 64, 65}));

  target = Variable(source.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3}));
  EXPECT_EQ(target, makeVariable<double>(Dimensions(dims), units::m,
                                         Values{22, 23, 32, 33},
                                         Variances{55, 56, 65, 66}));
}

TEST(VariableView, variable_assign_from_slice_clears_variances) {
  const Dimensions dims({{Dim::Y, 2}, {Dim::X, 2}});
  auto target = makeVariable<double>(Dimensions(dims), Values{1, 2, 3, 4},
                                     Variances{5, 6, 7, 8});
  const auto source =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3}, units::m,
                           Values{11, 12, 13, 21, 22, 23, 31, 32, 33});

  target = Variable(source.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2}));
  EXPECT_EQ(target, makeVariable<double>(Dimensions(dims), units::m,
                                         Values{11, 12, 21, 22}));
}

TEST(VariableView, slice_assign_from_variable_broadcast) {
  const auto source = makeVariable<double>(Values{2});
  auto target = makeVariable<double>(Dims{Dim::X}, Shape{3});
  target.slice({Dim::X, 1, 3}).assign(source);
  EXPECT_EQ(target, makeVariable<double>(target.dims(), Values{0, 2, 2}));
}

TEST(VariableView, variable_self_assign_via_slice) {
  auto target =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3},
                           Values{11, 12, 13, 21, 22, 23, 31, 32, 33},
                           Variances{44, 45, 46, 54, 55, 56, 64, 65, 66});

  target = Variable(target.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3}));
  // Note: This test does not actually fail if self-assignment is broken. Had to
  // run address sanitizer to see that it is reading from free'ed memory.
  EXPECT_EQ(target, makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                         Values{22, 23, 32, 33},
                                         Variances{55, 56, 65, 66}));
}

TEST(VariableView, slice_assign_from_variable_unit_fail) {
  const auto source = makeVariable<double>(Dims{Dim::X}, Shape{1}, units::m);
  auto target = makeVariable<double>(Dims{Dim::X}, Shape{2});
  EXPECT_THROW(target.slice({Dim::X, 1, 2}).assign(source), except::UnitError);
  target = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m);
  EXPECT_NO_THROW(target.slice({Dim::X, 1, 2}).assign(source));
}

TEST(VariableView, slice_assign_from_variable_dimension_fail) {
  const auto source = makeVariable<double>(Dims{Dim::Y}, Shape{1});
  auto target = makeVariable<double>(Dims{Dim::X}, Shape{2});
  EXPECT_THROW(target.slice({Dim::X, 1, 2}).assign(source),
               except::NotFoundError);
}

TEST(VariableView, slice_assign_from_variable_full_slice_can_change_unit) {
  const auto source = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m);
  auto target = makeVariable<double>(Dims{Dim::X}, Shape{2});
  target.slice({Dim::X, 0, 2}).assign(source);
  EXPECT_NO_THROW(target.slice({Dim::X, 0, 2}).assign(source));
}

TEST(VariableView, slice_assign_from_variable_variance_fail) {
  const auto vals = makeVariable<double>(Dims{Dim::X}, Shape{1});
  const auto vals_vars =
      makeVariable<double>(Dimensions{Dim::X, 1}, Values{}, Variances{});

  auto target = makeVariable<double>(Dims{Dim::X}, Shape{2});
  EXPECT_THROW(target.slice({Dim::X, 1, 2}).assign(vals_vars),
               except::VariancesError);
  EXPECT_NO_THROW(target.slice({Dim::X, 1, 2}).assign(vals));

  target = makeVariable<double>(Dimensions{Dim::X, 2}, Values{}, Variances{});
  EXPECT_THROW(target.slice({Dim::X, 1, 2}).assign(vals),
               except::VariancesError);
  EXPECT_NO_THROW(target.slice({Dim::X, 1, 2}).assign(vals_vars));
}

TEST(VariableView, slice_assign_from_variable) {
  const auto source =
      makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                           Values{11, 12, 21, 22}, Variances{33, 34, 43, 44});

  // We might want to mimick Python's __setitem__, but operator= would (and
  // should!?) assign the view contents, not the data.
  const Dimensions dims({{Dim::Y, 3}, {Dim::X, 3}});
  auto target = makeVariable<double>(Dimensions{dims}, Values{}, Variances{});
  target.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2}).assign(source);
  EXPECT_EQ(target, makeVariable<double>(
                        Dimensions(dims), Values{11, 12, 0, 21, 22, 0, 0, 0, 0},
                        Variances{33, 34, 0, 43, 44, 0, 0, 0, 0}));

  target = makeVariable<double>(Dimensions{dims}, Values{}, Variances{});
  target.slice({Dim::X, 1, 3}).slice({Dim::Y, 0, 2}).assign(source);
  EXPECT_EQ(target, makeVariable<double>(
                        Dimensions(dims), Values{0, 11, 12, 0, 21, 22, 0, 0, 0},
                        Variances{0, 33, 34, 0, 43, 44, 0, 0, 0}));

  target = makeVariable<double>(Dimensions{dims}, Values{}, Variances{});
  target.slice({Dim::X, 0, 2}).slice({Dim::Y, 1, 3}).assign(source);
  EXPECT_EQ(target, makeVariable<double>(
                        Dimensions(dims), Values{0, 0, 0, 11, 12, 0, 21, 22, 0},
                        Variances{0, 0, 0, 33, 34, 0, 43, 44, 0}));

  target = makeVariable<double>(Dimensions{dims}, Values{}, Variances{});
  target.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3}).assign(source);
  EXPECT_EQ(target, makeVariable<double>(
                        Dimensions(dims), Values{0, 0, 0, 0, 11, 12, 0, 21, 22},
                        Variances{0, 0, 0, 0, 33, 34, 0, 43, 44}));
}

TEST(VariableTest, rename) {
  auto var = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 3},
                                  Values{1, 2, 3, 4, 5, 6},
                                  Variances{7, 8, 9, 10, 11, 12});
  const Variable expected(reshape(var, {{Dim::X, 2}, {Dim::Z, 3}}));

  VariableConstView view(var);
  view.rename(Dim::Y, Dim::Z);
  ASSERT_EQ(view, expected);
  ASSERT_EQ(view.slice({Dim::X, 1}), expected.slice({Dim::X, 1}));
  ASSERT_EQ(view.slice({Dim::Z, 1}), expected.slice({Dim::Z, 1}));
  ASSERT_NE(var, expected);

  var.rename(Dim::Y, Dim::Z);
  ASSERT_EQ(view, expected);
  ASSERT_EQ(view.slice({Dim::X, 1}), expected.slice({Dim::X, 1}));
  ASSERT_EQ(view.slice({Dim::Z, 1}), expected.slice({Dim::Z, 1}));
  ASSERT_EQ(var, expected);
}

TEST(VariableTest, create_with_variance) {
  ASSERT_NO_THROW(makeVariable<double>(Values{1.0}, Variances{0.1}));
  ASSERT_NO_THROW(makeVariable<double>(Dims(), Shape(), units::m, Values{1.0},
                                       Variances{0.1}));
}

TEST(VariableTest, hasVariances) {
  ASSERT_FALSE(makeVariable<double>(Values{double{}}).hasVariances());
  ASSERT_FALSE(makeVariable<double>(Values{1.0}).hasVariances());
  ASSERT_TRUE(makeVariable<double>(Values{1.0}, Variances{0.1}).hasVariances());
  ASSERT_TRUE(makeVariable<double>(Dims(), Shape(), units::m, Values{1.0},
                                   Variances{0.1})
                  .hasVariances());
}

TEST(VariableTest, values_variances) {
  const auto var = makeVariable<double>(Values{1.0}, Variances{0.1});
  ASSERT_NO_THROW(var.values<double>());
  ASSERT_NO_THROW(var.variances<double>());
  ASSERT_TRUE(equals(var.values<double>(), {1.0}));
  ASSERT_TRUE(equals(var.variances<double>(), {0.1}));
}

template <typename Var> void test_set_variances(Var &var) {
  const auto v = var * (2.0 * units::one);
  var.setVariances(Variable(var));
  ASSERT_TRUE(equals(var.template variances<double>(), {1.0, 2.0, 3.0}));
  // Fail because `var` has variances (setVariances uses only the values)
  EXPECT_THROW(var.setVariances(var * (2.0 * units::one)),
               except::VariancesError);
  var.setVariances(v);
  ASSERT_TRUE(equals(var.template variances<double>(), {2.0, 4.0, 6.0}));

  Variable bad_dims = copy(v);
  bad_dims.rename(Dim::X, Dim::Y);
  EXPECT_THROW(var.setVariances(bad_dims), except::DimensionError);

  Variable bad_unit = copy(v);
  bad_unit.setUnit(units::s);
  EXPECT_THROW(var.setVariances(bad_unit), except::UnitError);

  EXPECT_THROW(var.setVariances(astype(v, dtype<float>)), except::TypeError);
}

TEST(VariableTest, set_variances) {
  Variable var = makeVariable<double>(Dims{Dim::X}, Shape{3}, units::m,
                                      Values{1.0, 2.0, 3.0});
  test_set_variances(var);
}

TEST(VariableTest, set_variances_remove) {
  Variable var =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{}, Variances{});
  EXPECT_TRUE(var.hasVariances());
  EXPECT_NO_THROW(var.setVariances(Variable()));
  EXPECT_FALSE(var.hasVariances());
}

TEST(VariableViewTest, set_variances) {
  Variable var = makeVariable<double>(Dims{Dim::X}, Shape{3}, units::m,
                                      Values{1.0, 2.0, 3.0});
  auto view = VariableView(var);
  test_set_variances(view);
  EXPECT_THROW(
      var.slice({Dim::X, 0}).setVariances(Variable(var.slice({Dim::X, 0}))),
      except::VariancesError);
}

TEST(VariableViewTest, set_variances_slice_fail) {
  Variable var = makeVariable<double>(Dims{Dim::X}, Shape{3});
  EXPECT_THROW(
      var.slice({Dim::X, 0}).setVariances(Variable(var.slice({Dim::X, 0}))),
      except::VariancesError);
}

TEST(VariableViewTest, create_with_variance) {
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{2},
                                        Values{1.0, 2.0}, Variances{0.1, 0.2});
  ASSERT_NO_THROW(var.slice({Dim::X, 1, 2}));
  const auto slice = var.slice({Dim::X, 1, 2});
  ASSERT_TRUE(slice.hasVariances());
  ASSERT_EQ(slice.variances<double>().size(), 1);
  ASSERT_EQ(slice.variances<double>()[0], 0.2);
  const auto reference =
      makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{2.0}, Variances{0.2});
  ASSERT_EQ(slice, reference);
}

TEST(VariableTest, variances_unsupported_type_fail) {
  ASSERT_NO_THROW(
      makeVariable<std::string>(Dims{Dim::X}, Shape{1}, Values{"a"}));
  ASSERT_THROW(makeVariable<std::string>(Dims{Dim::X}, Shape{1}, Values{"a"},
                                         Variances{"variances"}),
               except::VariancesError);
}

TEST(VariableTest, construct_view_dims) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2l, 3l});
  Variable vv(var.slice({Dim::X, 0, 2}));
  ASSERT_NO_THROW(Variable(var.slice({Dim::X, 0, 2}), Dimensions(Dim::Y, 2)));
}

TEST(VariableTest, construct_mult_dev_unit) {
  Variable refDiv =
      makeVariable<float>(Dims(), Shape(), units::one / units::m, Values{1.0f});
  Variable refMult =
      makeVariable<int32_t>(Dims(), Shape(), units::kg, Values{1});
  EXPECT_EQ(1.0f / units::m, refDiv);
  EXPECT_EQ(int32_t(1) * units::kg, refMult);
}

TEST(VariableTest, datetime_dtype) {
  auto dt =
      makeVariable<scipp::core::time_point>(Values{scipp::core::time_point{}});
  EXPECT_EQ(dt.dtype(), dtype<scipp::core::time_point>);
}

TEST(VariableTest, construct_time_unit) {
  Variable refMult =
      makeVariable<int64_t>(Dims(), Shape(), units::ns, Values{1000});
  EXPECT_EQ(int64_t(1000) * units::ns, refMult);
}

template <class T> class AsTypeTest : public ::testing::Test {};

using type_pairs =
    ::testing::Types<std::pair<float, double>, std::pair<double, float>,
                     std::pair<int32_t, float>>;
TYPED_TEST_SUITE(AsTypeTest, type_pairs);

TYPED_TEST(AsTypeTest, variable_astype) {
  using T1 = typename TypeParam::first_type;
  using T2 = typename TypeParam::second_type;
  Variable var1;
  Variable var2;
  if constexpr (core::canHaveVariances<T1>() && core::canHaveVariances<T2>()) {
    var1 = makeVariable<T1>(Values{1}, Variances{1});
    var2 = makeVariable<T2>(Values{1}, Variances{1});
    ASSERT_EQ(astype(var1, core::dtype<T2>), var2);
  }

  var1 = makeVariable<T1>(Values{1});
  var2 = makeVariable<T2>(Values{1});
  ASSERT_EQ(astype(var1, core::dtype<T2>), var2);
  var1 =
      makeVariable<T1>(Dims{Dim::X}, Shape{3}, units::m, Values{1.0, 2.0, 3.0});
  var2 =
      makeVariable<T2>(Dims{Dim::X}, Shape{3}, units::m, Values{1.0, 2.0, 3.0});
  ASSERT_EQ(astype(var1, core::dtype<T2>), var2);
}

TEST(TransposeTest, make_transposed_2d) {
  auto var = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{3, 2},
                                  Values{1, 2, 3, 4, 5, 6},
                                  Variances{11, 12, 13, 14, 15, 16});

  const auto constVar = Variable(var);

  auto ref = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                  Values{1, 3, 5, 2, 4, 6},
                                  Variances{11, 13, 15, 12, 14, 16});
  EXPECT_EQ(transpose(var, {Dim::Y, Dim::X}), ref);
  EXPECT_EQ(transpose(constVar, {Dim::Y, Dim::X}), ref);

  EXPECT_THROW_DISCARD(transpose(constVar, {Dim::Y, Dim::Z}),
                       except::DimensionError);
  EXPECT_THROW_DISCARD(transpose(constVar, {Dim::Y}), except::DimensionError);
  EXPECT_THROW_DISCARD(transpose(constVar, {Dim::Y, Dim::Z}),
                       except::DimensionError);
  EXPECT_THROW_DISCARD(transpose(var, {Dim::Z}), except::DimensionError);
}

TEST(TransposeTest, make_transposed_multiple_d) {
  auto var = makeVariable<double>(Dims{Dim::X, Dim::Y, Dim::Z}, Shape{3, 2, 1},
                                  Values{1, 2, 3, 4, 5, 6},
                                  Variances{11, 12, 13, 14, 15, 16});

  const auto constVar = Variable(var);

  auto ref = makeVariable<double>(Dims{Dim::Y, Dim::Z, Dim::X}, Shape{2, 1, 3},
                                  Values{1, 3, 5, 2, 4, 6},
                                  Variances{11, 13, 15, 12, 14, 16});
  EXPECT_EQ(transpose(var, {Dim::Y, Dim::Z, Dim::X}), ref);
  EXPECT_EQ(transpose(constVar, {Dim::Y, Dim::Z, Dim::X}), ref);

  EXPECT_THROW_DISCARD(transpose(constVar, {Dim::Y, Dim::Z}),
                       except::DimensionError);
  EXPECT_THROW_DISCARD(transpose(constVar, {Dim::Y}), except::DimensionError);
  EXPECT_THROW_DISCARD(transpose(var, {Dim::Y, Dim::Z}),
                       except::DimensionError);
  EXPECT_THROW_DISCARD(transpose(var, {Dim::Z}), except::DimensionError);
}

TEST(TransposeTest, reverse) {
  Variable var = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{3, 2},
                                      Values{1, 2, 3, 4, 5, 6},
                                      Variances{11, 12, 13, 14, 15, 16});
  const Variable constVar = makeVariable<double>(
      Dims{Dim::X, Dim::Y}, Shape{3, 2}, Values{1, 2, 3, 4, 5, 6},
      Variances{11, 12, 13, 14, 15, 16});
  auto ref = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                  Values{1, 3, 5, 2, 4, 6},
                                  Variances{11, 13, 15, 12, 14, 16});
  auto tvar = transpose(var);
  auto tconstVar = transpose(constVar);
  static_assert(
      std::is_same_v<VariableConstView, std::decay_t<decltype(tconstVar)>>);
  static_assert(std::is_same_v<VariableView, decltype(tvar)>);
  EXPECT_EQ(tvar, ref);
  EXPECT_EQ(tconstVar, ref);
  auto tview = transpose(VariableView(var));
  auto tconstView = transpose(VariableConstView(constVar));
  static_assert(
      std::is_same_v<VariableConstView, std::decay_t<decltype(tconstView)>>);
  static_assert(std::is_same_v<VariableView, decltype(tview)>);
  EXPECT_EQ(tconstView, ref);
  EXPECT_EQ(tview, ref);
  auto v = transpose(makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{3, 2},
                                          Values{1, 2, 3, 4, 5, 6},
                                          Variances{11, 12, 13, 14, 15, 16}));
  static_assert(std::is_same_v<Variable, decltype(v)>);
  EXPECT_EQ(v, ref);

  EXPECT_EQ(transpose(transpose(var)), var);
  EXPECT_EQ(transpose(transpose(constVar)), var);

  Variable dummy = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 1},
                                        Values{0, 0}, Variances{1, 1});
  EXPECT_NO_THROW(tvar.slice({Dim::X, 0, 1}).assign(dummy));
}

TEST(VariableTest, array_params) {
  const auto parent =
      makeVariable<double>(Dims{Dim::X, Dim::Y, Dim::Z}, Shape{4, 2, 3});

  const Dimensions yz({Dim::Y, Dim::Z}, {8, 3});
  const Dimensions xz({Dim::X, Dim::Z}, {4, 6});
  Dimensions xy = parent.dims();
  xy.relabel(2, Dim::Invalid);
  EXPECT_EQ(parent.array_params().dataDims(), parent.dims());
  EXPECT_EQ(parent.slice({Dim::X, 1}).array_params().dataDims(), yz);
  EXPECT_EQ(parent.slice({Dim::Y, 1}).array_params().dataDims(), xz);
  EXPECT_EQ(parent.slice({Dim::Z, 1}).array_params().dataDims(), xy);

  const auto empty_1d = makeVariable<double>(Dims{Dim::X}, Shape{0});
  EXPECT_EQ(empty_1d.array_params().dataDims(), empty_1d.dims());
  const auto empty_2d = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 0});
  EXPECT_EQ(empty_2d.array_params().dataDims(), empty_2d.dims());
}
