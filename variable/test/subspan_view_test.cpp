// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "test_macros.h"

#include "scipp/core/except.h"
#include "scipp/variable/subspan_view.h"

using namespace scipp;

class SubspanViewTest : public ::testing::Test {
protected:
  Variable var{makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3}, units::m,
                                    Values{1, 2, 3, 4, 5, 6})};
  Variable var_with_errors{makeVariable<double>(
      Dims{Dim::Y, Dim::X}, Shape{2, 3}, units::m, Values{1, 2, 3, 4, 5, 6},
      Variances{7, 8, 9, 10, 11, 12})};
};

TEST_F(SubspanViewTest, fail_not_inner) {
  EXPECT_THROW([[maybe_unused]] auto view = subspan_view(var, Dim::Y),
               except::DimensionError);
}

TEST_F(SubspanViewTest, values) {
  auto view = subspan_view(var, Dim::X);
  EXPECT_EQ(view.dims(), Dimensions({Dim::Y, 2}));
  EXPECT_EQ(view.unit(), units::m);
  EXPECT_TRUE(equals(view.values<span<double>>()[0], {1, 2, 3}));
  EXPECT_TRUE(equals(view.values<span<double>>()[1], {4, 5, 6}));
  EXPECT_FALSE(view.hasVariances());
}

TEST_F(SubspanViewTest, values_length_0) {
  auto view = subspan_view(var.slice({Dim::X, 0, 0}), Dim::X);
  EXPECT_EQ(view.dims(), Dimensions({Dim::Y, 2}));
  EXPECT_EQ(view.unit(), units::m);
  // Note the `const` here: Temporary returned by `slice()` uses `const Variable
  // &` overload.
  EXPECT_TRUE(view.values<span<const double>>()[0].empty());
  EXPECT_TRUE(view.values<span<const double>>()[1].empty());
  EXPECT_FALSE(view.hasVariances());
}

TEST_F(SubspanViewTest, values_and_errors) {
  auto view = subspan_view(var_with_errors, Dim::X);
  EXPECT_EQ(view.dims(), Dimensions({Dim::Y, 2}));
  EXPECT_EQ(view.unit(), units::m);
  EXPECT_TRUE(equals(view.values<span<double>>()[0], {1, 2, 3}));
  EXPECT_TRUE(equals(view.values<span<double>>()[1], {4, 5, 6}));
  EXPECT_TRUE(equals(view.variances<span<double>>()[0], {7, 8, 9}));
  EXPECT_TRUE(equals(view.variances<span<double>>()[1], {10, 11, 12}));
}

TEST_F(SubspanViewTest, values_and_errors_length_0) {
  auto view = subspan_view(var_with_errors.slice({Dim::X, 0, 0}), Dim::X);
  EXPECT_EQ(view.dims(), Dimensions({Dim::Y, 2}));
  EXPECT_EQ(view.unit(), units::m);
  EXPECT_TRUE(view.values<span<const double>>()[0].empty());
  EXPECT_TRUE(view.values<span<const double>>()[1].empty());
  EXPECT_TRUE(view.variances<span<const double>>()[0].empty());
  EXPECT_TRUE(view.variances<span<const double>>()[1].empty());
}

TEST_F(SubspanViewTest, view_of_const) {
  const auto &const_var = var;
  auto view = subspan_view(const_var, Dim::X);
  EXPECT_NO_THROW(view.values<span<const double>>());
}
