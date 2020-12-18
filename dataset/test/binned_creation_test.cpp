// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/dataset/bins.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/creation.h"

using namespace scipp;

class BinnedCreationTest : public ::testing::Test {
protected:
  Variable m_indices = makeVariable<scipp::index_pair>(
      Dims{Dim::X}, Shape{2}, Values{std::pair{0, 2}, std::pair{2, 5}});
  Variable m_data =
      makeVariable<double>(Dims{Dim::Event}, Shape{5}, Values{1, 2, 3, 4, 5});
  DataArray m_buffer = DataArray(m_data, {{Dim::X, m_data}}, {{"mask", m_data}},
                                 {{Dim("attr"), 1.2 * units::m}});
  Variable m_var = make_bins(m_indices, Dim::Event, m_buffer);
};

TEST_F(BinnedCreationTest, empty_like_default_shape) {
  const auto empty = empty_like(m_var);
  EXPECT_EQ(empty.dims(), m_var.dims());
  const auto [indices, dim, buf] = empty.constituents<core::bin<DataArray>>();
  EXPECT_EQ(indices, m_indices);
}

TEST_F(BinnedCreationTest, empty_like_slice_default_shape) {
  const auto empty = empty_like(m_var.slice({Dim::X, 1}));
  EXPECT_EQ(empty.dims(), m_var.slice({Dim::X, 1}).dims());
  const auto [indices, dim, buf] = empty.constituents<core::bin<DataArray>>();
  EXPECT_EQ(indices, makeVariable<scipp::index_pair>(Values{std::pair{0, 3}}));
}

TEST_F(BinnedCreationTest, empty_like) {
  Variable shape = makeVariable<scipp::index>(Dims{Dim::X, Dim::Y}, Shape{2, 3},
                                              Values{1, 2, 5, 6, 3, 4});
  const auto empty = empty_like(m_var, {}, shape);
  EXPECT_EQ(empty.dims(), shape.dims());
  const auto [indices, dim, buf] = empty.constituents<core::bin<DataArray>>();
  EXPECT_EQ(buf.dims(), Dimensions(Dim::Event, 21));
  EXPECT_EQ(buf.attrs(), m_buffer.attrs()); // scalar, so copied, not resized
  EXPECT_TRUE(buf.masks().contains("mask"));
  EXPECT_TRUE(buf.coords().contains(Dim::X));
  scipp::index i = 0;
  for (const auto n : {1, 2, 5, 6, 3, 4}) {
    EXPECT_EQ(empty.values<core::bin<DataArray>>()[i++].dims()[Dim::Event], n);
  }
}
