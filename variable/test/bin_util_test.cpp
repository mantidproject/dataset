// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/variable/bin_util.h"

using namespace scipp;
using namespace scipp::variable;

TEST(BinUtilTest, left_edge_right_edge) {
  const auto edges =
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  EXPECT_EQ(left_edge(edges),
            makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  EXPECT_EQ(right_edge(edges),
            makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{2, 3, 4}));
  EXPECT_THROW(left_edge(edges.slice({Dim::X, 0, 0})), except::BinEdgeError);
  EXPECT_THROW(left_edge(edges.slice({Dim::X, 0, 1})), except::BinEdgeError);
  EXPECT_THROW(right_edge(edges.slice({Dim::X, 0, 0})), except::BinEdgeError);
  EXPECT_THROW(right_edge(edges.slice({Dim::X, 0, 1})), except::BinEdgeError);
}
