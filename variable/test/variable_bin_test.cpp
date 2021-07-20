// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/variable/bins.h"
#include "scipp/variable/operations.h"

using namespace scipp;

class VariableBinsTest : public ::testing::Test {
protected:
  Dimensions dims{Dim::Y, 2};
  Variable indices = makeVariable<scipp::index_pair>(
      dims, Values{std::pair{0, 2}, std::pair{2, 4}});
  Variable buffer =
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  Variable var = make_bins(indices, Dim::X, buffer);
};

TEST_F(VariableBinsTest, make_bins_from_slice) {
  // Sharing indices or not yields equivalent results.
  EXPECT_EQ(make_bins(indices.slice({Dim::Y, 1}), Dim::X, buffer),
            make_bins(copy(indices.slice({Dim::Y, 1})), Dim::X, buffer));
}

TEST_F(VariableBinsTest, make_bins_shares_indices_and_buffer) {
  auto binned = make_bins(indices, Dim::X, buffer);
  EXPECT_EQ(binned.bin_indices().values<scipp::index_pair>().data(),
            indices.values<scipp::index_pair>().data());
  EXPECT_EQ(binned.values<bucket<Variable>>().front().values<double>().data(),
            buffer.values<double>().data());
}

TEST_F(VariableBinsTest, make_bins_from_slice_shares_indices_and_buffer) {
  auto binned = make_bins(indices.slice({Dim::Y, 1}), Dim::X, buffer);
  EXPECT_EQ(binned.bin_indices().values<scipp::index_pair>().data(),
            indices.values<scipp::index_pair>().data() + 1);
  EXPECT_EQ(binned.values<bucket<Variable>>().front().values<double>().data(),
            buffer.values<double>().data() + 2);
}

TEST_F(VariableBinsTest, comparison) {
  EXPECT_TRUE(var == var);
  EXPECT_FALSE(var != var);
  auto var2 = make_bins(copy(indices), Dim::X, copy(buffer));
  EXPECT_TRUE(var == var2);
}

TEST_F(VariableBinsTest, copy) {
  auto copied = copy(var);
  EXPECT_EQ(copied, var);
  copied.bin_indices().values<scipp::index_pair>()[0].first += 1;
  EXPECT_NE(copied, var); // indices gets copied
  copied = copy(var);
  EXPECT_EQ(copied, var);
  buffer.values<double>()[0] += 1;
  EXPECT_NE(copied, var); // buffer gets copied
}

TEST_F(VariableBinsTest, assign) {
  Variable copy = variable::copy(var);
  var.values<bucket<Variable>>()[0] += var.values<bucket<Variable>>()[1];
  EXPECT_NE(copy, var);
  variable::copy(var, copy);
  EXPECT_EQ(copy, var);
}

TEST_F(VariableBinsTest, copy_slice) {
  EXPECT_EQ(copy(var.slice({Dim::Y, 0, 2})), var);
  EXPECT_EQ(copy(var.slice({Dim::Y, 0, 1})), var.slice({Dim::Y, 0, 1}));
  EXPECT_EQ(copy(var.slice({Dim::Y, 1, 2})), var.slice({Dim::Y, 1, 2}));
}

TEST_F(VariableBinsTest, cannot_set_unit) {
  EXPECT_EQ(var.unit(), units::one);
  EXPECT_THROW(var.setUnit(units::m), except::UnitError);
  EXPECT_EQ(var.unit(), units::one);
}

TEST_F(VariableBinsTest, basics) {
  EXPECT_EQ(var.unit(), units::one);
  EXPECT_EQ(var.dims(), dims);
  const auto vals = var.values<bucket<Variable>>();
  EXPECT_EQ(vals.size(), 2);
  EXPECT_EQ(vals[0], buffer.slice({Dim::X, 0, 2}));
  EXPECT_EQ(vals[1], buffer.slice({Dim::X, 2, 4}));
  EXPECT_EQ(vals.front(), buffer.slice({Dim::X, 0, 2}));
  EXPECT_EQ(vals.back(), buffer.slice({Dim::X, 2, 4}));
  EXPECT_EQ(*vals.begin(), buffer.slice({Dim::X, 0, 2}));
  const auto &const_var = var;
  EXPECT_EQ(const_var.values<bucket<Variable>>()[0],
            buffer.slice({Dim::X, 0, 2}));
}

TEST_F(VariableBinsTest, view) {
  Variable view(var);
  EXPECT_EQ(view.values<bucket<Variable>>(), var.values<bucket<Variable>>());
  view = var.slice({Dim::Y, 1});
  const auto vals = view.values<bucket<Variable>>();
  EXPECT_EQ(vals.size(), 1);
  EXPECT_EQ(vals[0], buffer.slice({Dim::X, 2, 4}));
}

TEST_F(VariableBinsTest, construct_from_view) {
  Variable copy{Variable(var)};
  EXPECT_EQ(copy, var);
}

TEST_F(VariableBinsTest, unary_operation) {
  const auto expected = make_bins(indices, Dim::X, sqrt(buffer));
  EXPECT_EQ(sqrt(var), expected);
  EXPECT_EQ(sqrt(var.slice({Dim::Y, 1})), expected.slice({Dim::Y, 1}));
}

TEST_F(VariableBinsTest, binary_operation) {
  const auto expected = make_bins(indices, Dim::X, buffer + buffer);
  EXPECT_EQ(var + var, expected);
  EXPECT_EQ(var.slice({Dim::Y, 1}) + var.slice({Dim::Y, 1}),
            expected.slice({Dim::Y, 1}));
}

TEST_F(VariableBinsTest, binary_operation_with_dense) {
  Variable dense = makeVariable<double>(var.dims(), Values{0.1, 0.2});
  Variable expected_buffer =
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1.1, 2.1, 3.2, 4.2});
  const auto expected = make_bins(indices, Dim::X, expected_buffer);
  EXPECT_EQ(var + dense, expected);
  EXPECT_EQ(var.slice({Dim::Y, 1}) + dense.slice({Dim::Y, 1}),
            expected.slice({Dim::Y, 1}));
}

TEST_F(VariableBinsTest, binary_operation_with_dense_broadcast) {
  Variable dense =
      makeVariable<double>(Dims{Dim::Z}, Shape{2}, Values{0.1, 0.2});
  Variable expected_buffer = makeVariable<double>(
      Dims{Dim::X}, Shape{8}, Values{1.1, 2.1, 1.2, 2.2, 3.1, 4.1, 3.2, 4.2});
  Variable expected_indices =
      makeVariable<std::pair<scipp::index, scipp::index>>(
          Dims{Dim::Y, Dim::Z}, Shape{2, 2},
          Values{std::pair{0, 2}, std::pair{2, 4}, std::pair{4, 6},
                 std::pair{6, 8}});
  const auto expected = make_bins(expected_indices, Dim::X, expected_buffer);
  EXPECT_EQ(var + dense, expected);
  EXPECT_EQ(var.slice({Dim::Y, 1}) + dense, expected.slice({Dim::Y, 1}));
  EXPECT_EQ(dense + var, transpose(expected));
}

TEST_F(VariableBinsTest, binary_operation_with_dense_2d_bins) {
  auto indices_2d = makeVariable<scipp::index_pair>(
      Dims{Dim::X}, Shape{3},
      Values{std::pair{0, 1}, std::pair{1, 1}, std::pair{1, 4}});
  auto dense = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{2, 3, 4});

  // Bin in outer dim.
  auto buffer_2d = makeVariable<double>(Dims{Dim::Event, Dim::Y}, Shape{4, 2},
                                        Values{0, 1, 2, 3, 4, 5, 6, 7});
  auto binned = make_bins(indices_2d, Dim::Event, buffer_2d);
  EXPECT_EQ(
      binned * dense,
      make_bins(indices_2d, Dim::Event,
                makeVariable<double>(Dims{Dim::Event, Dim::Y}, Shape{4, 2},
                                     Values{0, 2, 8, 12, 16, 20, 24, 28})));

  // Bin in inner dim.
  buffer_2d = makeVariable<double>(Dims{Dim::Y, Dim::Event}, Shape{2, 4},
                                   Values{0, 1, 2, 3, 4, 5, 6, 7});
  binned = make_bins(indices_2d, Dim::Event, buffer_2d);
  EXPECT_EQ(binned * dense, make_bins(indices_2d, Dim::Event,
                                      makeVariable<double>(
                                          Dims{Dim::Y, Dim::Event}, Shape{2, 4},
                                          Values{0, 4, 8, 12, 8, 20, 24, 28})));
}

TEST_F(VariableBinsTest, binary_operation_strided) {
  Variable big_buffer = makeVariable<double>(Dimensions{Dim::X, 8},
                                             Values{1, 2, 3, 4, 5, 6, 7, 8});
  Variable indices_2d =
      makeVariable<scipp::index_pair>(Dimensions{{Dim::Y, 2}, {Dim::Z, 2}},
                                      Values{std::pair{0, 2}, std::pair{2, 4},
                                             std::pair{4, 6}, std::pair{6, 8}});
  Variable complete = make_bins(indices_2d, Dim::X, big_buffer);
  Variable sliced = complete.slice(Slice{Dim::Z, 0, 1});

  Variable expected_buffer =
      makeVariable<double>(Dimensions{Dim::X, 4}, Values{2, 4, 10, 12});
  Variable expected_indices =
      makeVariable<scipp::index_pair>(Dimensions{{Dim::Y, 2}, {Dim::Z, 1}},
                                      Values{std::pair{0, 2}, std::pair{2, 4}});
  Variable expected = make_bins(expected_indices, Dim::X, expected_buffer);
  EXPECT_EQ(sliced * makeVariable<double>(Dims{}, Values{2}), expected);
}

TEST_F(VariableBinsTest, to_constituents) {
  auto [idx0, dim0, buf0] = var.constituents<Variable>();
  static_cast<void>(dim0);
  auto idx_ptr = idx0.values<std::pair<scipp::index, scipp::index>>().data();
  auto buf_ptr = buf0.values<double>().data();
  auto [idx1, dim1, buf1] = var.to_constituents<Variable>();
  EXPECT_FALSE(var.is_valid());
  EXPECT_EQ((idx1.values<std::pair<scipp::index, scipp::index>>().data()),
            idx_ptr);
  EXPECT_EQ(buf1.values<double>().data(), buf_ptr);
  EXPECT_EQ(idx1, indices);
  EXPECT_EQ(dim1, Dim::X);
  EXPECT_EQ(buf1, buffer);
}

TEST_F(VariableBinsTest, setSlice) {
  const auto dense = makeVariable<double>(indices.dims(), Values{1.1, 2.2});
  var.setSlice({}, dense);
  auto expected = make_bins(
      indices, Dim::X,
      makeVariable<double>(buffer.dims(), Values{1.1, 1.1, 2.2, 2.2}));
  EXPECT_EQ(var, expected);
  var.setSlice({Dim::Y, 1}, dense.slice({Dim::Y, 0}));
  expected = make_bins(
      indices, Dim::X,
      makeVariable<double>(buffer.dims(), Values{1.1, 1.1, 1.1, 1.1}));
  EXPECT_EQ(var, expected);
}
