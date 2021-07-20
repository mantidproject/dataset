// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include "dataset_test_common.h"
#include "test_macros.h"
#include <gtest/gtest-matchers.h>
#include <gtest/gtest.h>

#include "scipp/dataset/bin.h"
#include "scipp/dataset/bins.h"
#include "scipp/dataset/dataset.h"
#include "scipp/dataset/histogram.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/comparison.h"
#include "scipp/variable/shape.h"

using namespace scipp;
using namespace scipp::dataset;

struct HistogramHelpersTest : public ::testing::Test {
protected:
  Variable dataX = makeVariable<double>(Dims{Dim::X}, Shape{2});
  Variable dataY = makeVariable<double>(Dims{Dim::Y}, Shape{2});
  Variable dataXY = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 3});
  Variable edgesX = makeVariable<double>(Dims{Dim::X}, Shape{3});
  Variable edgesY = makeVariable<double>(Dims{Dim::Y}, Shape{4});
  Variable coordX = makeVariable<double>(Dims{Dim::X}, Shape{2});
  Variable coordY = makeVariable<double>(Dims{Dim::Y}, Shape{3});
};

TEST_F(HistogramHelpersTest, edge_dimension) {
  const auto histX = DataArray(dataX, {{Dim::X, edgesX}});
  EXPECT_EQ(edge_dimension(histX), Dim::X);

  const auto histX2d = DataArray(dataXY, {{Dim::X, edgesX}});
  EXPECT_EQ(edge_dimension(histX2d), Dim::X);

  const auto histY2d = DataArray(dataXY, {{Dim::X, coordX}, {Dim::Y, edgesY}});
  EXPECT_EQ(edge_dimension(histY2d), Dim::Y);

  const auto hist2d = DataArray(dataXY, {{Dim::X, edgesX}, {Dim::Y, edgesY}});
  EXPECT_THROW(edge_dimension(hist2d), except::BinEdgeError);

  EXPECT_THROW(edge_dimension(DataArray(dataX, {{Dim::X, coordX}})),
               except::BinEdgeError);
  EXPECT_THROW(edge_dimension(DataArray(dataX, {{Dim::Y, coordX}})),
               except::BinEdgeError);

  // Coord length X is 2 and data does not depend on X, but this is *not*
  // interpreted as a single-bin histogram.
  EXPECT_THROW(edge_dimension(DataArray(dataY, {{Dim::X, coordX}})),
               except::BinEdgeError);
}

TEST_F(HistogramHelpersTest, is_histogram) {
  const auto histX = DataArray(dataX, {{Dim::X, edgesX}});
  EXPECT_TRUE(is_histogram(histX, Dim::X));
  EXPECT_FALSE(is_histogram(histX, Dim::Y));
  // Also for Dataset
  const auto ds_histX = Dataset{histX};
  EXPECT_TRUE(is_histogram(ds_histX, Dim::X));
  EXPECT_FALSE(is_histogram(ds_histX, Dim::Y));

  const auto histX2d = DataArray(dataXY, {{Dim::X, edgesX}});
  EXPECT_TRUE(is_histogram(histX2d, Dim::X));
  EXPECT_FALSE(is_histogram(histX2d, Dim::Y));

  const auto histY2d = DataArray(dataXY, {{Dim::X, coordX}, {Dim::Y, edgesY}});
  EXPECT_FALSE(is_histogram(histY2d, Dim::X));
  EXPECT_TRUE(is_histogram(histY2d, Dim::Y));

  EXPECT_FALSE(is_histogram(DataArray(dataX, {{Dim::X, coordX}}), Dim::X));
  EXPECT_FALSE(is_histogram(DataArray(dataX, {{Dim::Y, coordX}}), Dim::X));

  // Coord length X is 2 and data does not depend on X, but this is *not*
  // interpreted as a single-bin histogram.
  EXPECT_FALSE(is_histogram(DataArray(dataY, {{Dim::X, coordX}}), Dim::X));
}

DataArray make_1d_events_default_weights() {
  const auto y = makeVariable<double>(
      Dims{Dim::Event}, Shape{22},
      Values{1.5, 2.5, 3.5, 4.5, 5.5, 3.5, 4.5, 5.5, 6.5, 7.5, -1.0,
             0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 6.0});
  const auto weights = copy(broadcast(
      makeVariable<double>(units::counts, Values{1}, Variances{1}), y.dims()));
  const DataArray table(weights, {{Dim::Y, y}});
  const auto indices = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::X}, Shape{3},
      Values{std::pair{0, 5}, std::pair{5, 10}, std::pair{10, 22}});
  auto var = make_bins(indices, Dim::Event, table);
  return DataArray(var, {});
}

TEST(HistogramTest, fail_edges_not_sorted) {
  auto events = make_1d_events_default_weights();
  ASSERT_THROW(dataset::histogram(
                   events, makeVariable<double>(Dims{Dim::Y}, Shape{6},
                                                Values{1, 3, 2, 4, 5, 6})),
               except::BinEdgeError);
}

auto make_single_events() {
  const auto x =
      makeVariable<double>(Dims{Dim::Event}, Shape{5}, Values{0, 1, 1, 2, 3});
  const auto weights = copy(broadcast(
      makeVariable<double>(units::counts, Values{1}, Variances{1}), x.dims()));
  const DataArray table(weights, {{Dim::X, x}});
  const auto indices = makeVariable<std::pair<scipp::index, scipp::index>>(
      Values{std::pair{0, 5}});
  Dataset events;
  events.setData("events", make_bins(indices, Dim::Event, table));
  return events;
}

DataArray make_expected(const Variable &var, const Variable &edges) {
  auto dim = var.dims().inner();
  std::unordered_map<Dim, Variable> coords = {{dim, edges}};
  auto expected = DataArray(var, coords, {}, {}, "events");
  return expected;
}

TEST(HistogramTest, below) {
  const auto events = make_single_events();
  auto edges =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{-2.0, -1.0, 0.0});
  auto hist = dataset::histogram(events["events"], edges);
  std::map<Dim, Variable> coords = {{Dim::X, edges}};
  auto expected =
      make_expected(makeVariable<double>(Dims{Dim::X}, Shape{2}, units::counts,
                                         Values{0, 0}, Variances{0, 0}),
                    edges);
  EXPECT_EQ(hist, expected);
}

TEST(HistogramTest, between) {
  const auto events = make_single_events();
  auto edges =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1.5, 1.6, 1.7});
  auto hist = dataset::histogram(events["events"], edges);
  auto expected =
      make_expected(makeVariable<double>(Dims{Dim::X}, Shape{2}, units::counts,
                                         Values{0, 0}, Variances{0, 0}),
                    edges);
  EXPECT_EQ(hist, expected);
}

TEST(HistogramTest, above) {
  const auto events = make_single_events();
  auto edges =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{3.5, 4.5, 5.5});
  auto hist = dataset::histogram(events["events"], edges);
  auto expected =
      make_expected(makeVariable<double>(Dims{Dim::X}, Shape{2}, units::counts,
                                         Values{0, 0}, Variances{0, 0}),
                    edges);
  EXPECT_EQ(hist, expected);
}

TEST(HistogramTest, data_view) {
  auto events = make_1d_events_default_weights();
  std::vector<double> ref{1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2, 3, 0, 3, 0};
  auto edges =
      makeVariable<double>(Dims{Dim::Y}, Shape{6}, Values{1, 2, 3, 4, 5, 6});
  auto hist = dataset::histogram(events, edges);
  auto expected = make_expected(
      makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{3, 5}, units::counts,
                           Values(ref.begin(), ref.end()),
                           Variances(ref.begin(), ref.end())),
      edges);

  EXPECT_EQ(hist, expected);
}

TEST(HistogramTest, dense) {
  auto events = make_1d_events_default_weights();
  auto edges1 =
      makeVariable<double>(Dims{Dim::Y}, Shape{6}, Values{1, 2, 3, 4, 5, 6});
  auto edgesY = makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{1, 3, 6});
  auto edgesZ = makeVariable<double>(Dims{Dim::Z}, Shape{3}, Values{1, 3, 6});
  auto expected = dataset::histogram(events, edgesY);
  auto dense = dataset::histogram(events, edges1);
  EXPECT_THROW(dataset::histogram(dense, edgesY), except::BinEdgeError);
  // dense depends on Y, histogram by Y coord into Y-dependent histogram
  EXPECT_TRUE(dense.dims().contains(edgesY.dims().inner()));
  dense.coords().erase(Dim::Y);
  dense.coords().set(Dim::Y,
                     makeVariable<double>(Dims{Dim::Y}, Shape{5},
                                          Values{1.5, 2.5, 3.5, 4.5, 5.5}));
  EXPECT_EQ(dataset::histogram(dense, edgesY), expected);
  // dense depends on Y, histogram by Z coord into Z-dependent histogram
  EXPECT_FALSE(dense.dims().contains(edgesZ.dims().inner()));
  dense.coords().set(Dim::Z,
                     makeVariable<double>(Dims{Dim::Y}, Shape{5},
                                          Values{1.5, 2.5, 3.5, 4.5, 5.5}));
  expected.rename(Dim::Y, Dim::Z);
  expected.coords().set(Dim::Z, expected.coords().extract(Dim::Y));
  EXPECT_EQ(dataset::histogram(dense, edgesZ), expected);
}

TEST(HistogramTest, keeps_scalar_coords) {
  auto events = make_1d_events_default_weights();
  events.coords().set(Dim("scalar"), makeVariable<double>(Values{1.2}));
  auto edges = makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{1, 6});
  auto hist = dataset::histogram(events, edges);
  EXPECT_TRUE(hist.coords().contains(Dim("scalar")));
}

DataArray make_1d_events() {
  const auto y = makeVariable<double>(
      Dims{Dim::Event}, Shape{22},
      Values{1.5, 2.5, 3.5, 4.5, 5.5, 3.5, 4.5, 5.5, 6.5, 7.5, -1.0,
             0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 6.0});
  const auto weights = makeVariable<double>(
      y.dims(), units::counts,
      Values{1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      Variances{1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  const DataArray table(weights, {{Dim::Y, y}});
  const auto indices = makeVariable<std::pair<scipp::index, scipp::index>>(
      Dims{Dim::X}, Shape{3},
      Values{std::pair{0, 5}, std::pair{5, 10}, std::pair{10, 22}});
  auto var = make_bins(indices, Dim::Event, table);
  return DataArray(var, {});
}

TEST(HistogramTest, weight_lists) {
  const auto events = make_1d_events();
  auto edges =
      makeVariable<double>(Dims{Dim::Y}, Shape{6}, Values{1, 2, 3, 4, 5, 6});
  std::vector<double> ref{1, 1, 1, 2, 2, 0, 0, 2, 2, 2, 2, 3, 0, 3, 0};
  auto expected = make_expected(
      makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{3, 5}, units::counts,
                           Values(ref.begin(), ref.end()),
                           Variances(ref.begin(), ref.end())),
      edges);
  EXPECT_EQ(dataset::histogram(events, edges), expected);
}

TEST(HistogramTest, dense_vs_binned) {
  using testdata::make_table;
  auto table_no_variance = make_table(100);
  table_no_variance.data().setVariances(Variable{});
  for (const auto &table :
       {make_table(0), make_table(100), make_table(1000), table_no_variance}) {
    const auto binned_x =
        bin(table, {makeVariable<double>(Dims{Dim::X}, Shape{5},
                                         Values{-2, -1, 0, 1, 2})});
    auto binned_y = bin(
        table, {makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{-2, 2})});
    binned_y.coords().erase(Dim::Y);
    const auto edges =
        makeVariable<double>(Dims{Dim::X}, Shape{8},
                             Values{-2.0, -1.5, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0});
    EXPECT_EQ(histogram(table, edges), histogram(binned_x, edges));
    EXPECT_EQ(histogram(table, edges),
              histogram(binned_y.slice({Dim::Y, 0}), edges));
  }
}

struct Histogram1DTest : public ::testing::Test {
protected:
  Histogram1DTest() {
    data = makeVariable<double>(Dims{Dim::X}, Shape{10},
                                Values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    coord = makeVariable<double>(Dims{Dim::X}, Shape{10},
                                 Values{1, 2, 1, 2, 3, 4, 3, 2, 1, 1});
    mask = less(data, 4.0 * units::one);
  }
  Variable data;
  Variable coord;
  Variable mask;
};

TEST_F(Histogram1DTest, coord_name_matches_dim) {
  DataArray da(data, {{Dim::X, coord}}, {{"mask", mask}});
  const auto edges =
      makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  EXPECT_EQ(histogram(da, edges).data(),
            makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{19, 12, 12}));
}

TEST_F(Histogram1DTest, coord_name_differs_dim) {
  // Ensure `histogram` considers masks that depend on Dim::X rather than Dim::Y
  DataArray da(data, {{Dim::Y, coord}}, {{"mask", mask}});
  const auto edges =
      makeVariable<double>(Dims{Dim::Y}, Shape{4}, Values{1, 2, 3, 4});
  EXPECT_EQ(histogram(da, edges).data(),
            makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{19, 12, 12}));
}

struct Histogram2DTest : public ::testing::Test {
protected:
  Histogram2DTest() {
    data = makeVariable<double>(
        Dims{Dim::Y, Dim::X}, Shape{3, 4},
        Values{11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34});
    coord = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 4},
                                 Values{1, 2, 1, 2, 3, 4, 3, 2, 1, 1, 2, 3});
  }
  Variable data;
  Variable coord;
};

TEST_F(Histogram2DTest, outer_1d_coord) {
  DataArray da(data, {{Dim::Y, coord.slice({Dim::X, 0})}});
  // data:
  // 11, 12, 13, 14
  // 21, 22, 23, 24
  // 31, 32, 33, 34
  // coord: 1, 3, 1 => [sum of rows 1 and 3, row 2]
  const auto edges =
      makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{1.0, 2.5, 5.0});
  EXPECT_EQ(histogram(da, edges).data(),
            makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{4, 2},
                                 Values{42, 21, 44, 22, 46, 23, 48, 24}));
}

TEST_F(Histogram2DTest, outer_2d_coord) {
  DataArray da(data, {{Dim::Y, coord}});
  // data:
  // 11, 12, 13, 14
  // 21, 22, 23, 24
  // 31, 32, 33, 34
  // coord:
  // 1, 2, 1, 2
  // 3, 4, 3, 2
  // 1, 1, 2, 3
  const auto edges =
      makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{1.0, 2.5, 5.0});
  EXPECT_EQ(histogram(da, edges).data(),
            makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{4, 2},
                                 Values{42, 21, 44, 22, 46, 23, 38, 34}));
}

TEST_F(Histogram2DTest, outer_2d_coord_tranposed) {
  // Histogramming dim is outer dim of data but inner dim of coord in `da2`
  DataArray da1(data, {{Dim::Y, coord}});
  DataArray da2(data, {{Dim::Y, copy(transpose(coord))}});
  const auto edges =
      makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{1.0, 2.5, 5.0});
  EXPECT_EQ(histogram(da1, edges), histogram(da2, edges));
}

TEST_F(Histogram2DTest, noncontiguous_slice) {
  DataArray da(data, {{Dim::Y, coord}});
  const auto edges =
      makeVariable<double>(Dims{Dim::Y}, Shape{3}, Values{1.0, 2.5, 5.0});
  // 1d histogram but along Dim::Y which has stride 4 since based on slice
  const auto slice = da.slice({Dim::X, 0});
  EXPECT_EQ(histogram(slice, edges), histogram(copy(slice), edges));
}
