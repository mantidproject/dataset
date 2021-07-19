// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/variable/convolution.h"
#include "scipp/variable/variable.h"

using namespace scipp;

TEST(ConvolveTest, 1d) {
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{6}, units::m,
                                        Values{1, 2, 3, 4, 5, 6});
  const auto kernel = makeVariable<double>(Dims{Dim::X}, Shape{2},
                                           units::one / units::s, Values{1, 1});
  EXPECT_EQ(convolve(var, kernel),
            makeVariable<double>(Dims{Dim::X}, Shape{5}, units::m / units::s,
                                 Values{3, 5, 7, 9, 11}));
}

TEST(ConvolveTest, 2d) {
  // 123
  // 456
  // 789
  const auto var = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{3, 3},
                                        Values{1, 2, 3, 4, 5, 6, 7, 8, 9});
  const auto kernel = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 2},
                                           Values{1, 1, 1, 1});
  EXPECT_EQ(convolve(var, kernel),
            makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 2},
                                 Values{12, 16, 24, 28}));
}
