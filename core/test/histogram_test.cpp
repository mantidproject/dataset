// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
#include "test_macros.h"
#include <gtest/gtest-matchers.h>
#include <gtest/gtest.h>

#include "scipp/core/dataset.h"
#include "scipp/core/histogram.h"

using namespace scipp;
using namespace scipp::core;

TEST(HistogramTest, is_histogram) {
  const auto dataX = createVariable<double>(Dims{Dim::X}, Shape{2});
  const auto dataY = createVariable<double>(Dims{Dim::Y}, Shape{2});
  const auto dataXY = createVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 3});
  const auto edgesX = createVariable<double>(Dims{Dim::X}, Shape{3});
  const auto edgesY = createVariable<double>(Dims{Dim::Y}, Shape{4});
  const auto coordX = createVariable<double>(Dims{Dim::X}, Shape{2});
  const auto coordY = createVariable<double>(Dims{Dim::Y}, Shape{3});

  const auto histX = DataArray(dataX, {{Dim::X, edgesX}});
  EXPECT_TRUE(is_histogram(histX, Dim::X));
  EXPECT_FALSE(is_histogram(histX, Dim::Y));

  const auto histX2d = DataArray(dataXY, {{Dim::X, edgesX}});
  EXPECT_TRUE(is_histogram(histX2d, Dim::X));
  EXPECT_FALSE(is_histogram(histX2d, Dim::Y));

  const auto histY2d = DataArray(dataXY, {{Dim::X, coordX}, {Dim::Y, edgesY}});
  EXPECT_FALSE(is_histogram(histY2d, Dim::X));
  EXPECT_TRUE(is_histogram(histY2d, Dim::Y));

  EXPECT_FALSE(is_histogram(DataArray(dataX, {{Dim::X, coordX}}), Dim::X));
  EXPECT_FALSE(is_histogram(DataArray(dataX, {{Dim::X, coordY}}), Dim::X));
  EXPECT_FALSE(is_histogram(DataArray(dataX, {{Dim::Y, coordX}}), Dim::X));
  EXPECT_FALSE(is_histogram(DataArray(dataX, {{Dim::Y, coordY}}), Dim::X));

  // Coord length X is 2 and data does not depend on X, but this is *not*
  // interpreted as a single-bin histogram.
  EXPECT_FALSE(is_histogram(DataArray(dataY, {{Dim::X, coordX}}), Dim::X));

  const auto sparse =
      createVariable<double>(Dims{Dim::X}, Shape{Dimensions::Sparse});
  EXPECT_FALSE(is_histogram(DataArray(sparse, {{Dim::X, coordX}}), Dim::X));
}

Dataset make_2d_sparse_coord_only(const std::string &name) {
  Dataset sparse;
  auto var = createVariable<double>(Dims{Dim::X, Dim::Y},
                                    Shape{3l, Dimensions::Sparse});
  var.sparseValues<double>()[0] = {1.5, 2.5, 3.5, 4.5, 5.5};
  var.sparseValues<double>()[1] = {3.5, 4.5, 5.5, 6.5, 7.5};
  var.sparseValues<double>()[2] = {-1, 0, 0, 1, 1, 2, 2, 2, 4, 4, 4, 6};
  sparse.setSparseCoord(name, var);
  return sparse;
}

TEST(HistogramTest, fail_edges_not_sorted) {
  auto sparse = make_2d_sparse_coord_only("sparse");
  ASSERT_THROW(core::histogram(sparse["sparse"], createVariable<double>(
                                                     Dims{Dim::Y}, Shape{6},
                                                     Values{1, 3, 2, 4, 5, 6})),
               except::BinEdgeError);
}

auto make_single_sparse() {
  Dataset sparse;
  sparse.setSparseCoord("sparse", createVariable<double>(
                                      Dims{Dim::X}, Shape{Dimensions::Sparse}));
  sparse["sparse"].coords()[Dim::X].sparseValues<double>()[0] = {0, 1, 1, 2, 3};
  return sparse;
}

DataArray make_expected(const Variable &var, const Variable &edges) {
  auto dim = var.dims().inner();
  std::map<Dim, Variable> coords = {{dim, edges}};
  auto expected = DataArray(var, coords, {}, {}, {}, "sparse");
  return expected;
}

TEST(HistogramTest, below) {
  const auto sparse = make_single_sparse();
  auto edges =
      createVariable<double>(Dims{Dim::X}, Shape{3}, Values{-2.0, -1.0, 0.0});
  auto hist = core::histogram(sparse["sparse"], edges);
  std::map<Dim, Variable> coords = {{Dim::X, edges}};
  auto expected = make_expected(
      createVariable<double>(Dims{Dim::X}, Shape{2}, units::Unit(units::counts),
                             Values{0, 0}, Variances{0, 0}),
      edges);
  EXPECT_EQ(hist, expected);
}

TEST(HistogramTest, between) {
  const auto sparse = make_single_sparse();
  auto edges =
      createVariable<double>(Dims{Dim::X}, Shape{3}, Values{1.5, 1.6, 1.7});
  auto hist = core::histogram(sparse["sparse"], edges);
  auto expected = make_expected(
      createVariable<double>(Dims{Dim::X}, Shape{2}, units::Unit(units::counts),
                             Values{0, 0}, Variances{0, 0}),
      edges);
  EXPECT_EQ(hist, expected);
}

TEST(HistogramTest, above) {
  const auto sparse = make_single_sparse();
  auto edges =
      createVariable<double>(Dims{Dim::X}, Shape{3}, Values{3.5, 4.5, 5.5});
  auto hist = core::histogram(sparse["sparse"], edges);
  auto expected = make_expected(
      createVariable<double>(Dims{Dim::X}, Shape{2}, units::Unit(units::counts),
                             Values{0, 0}, Variances{0, 0}),
      edges);
  EXPECT_EQ(hist, expected);
}

TEST(HistogramTest, data_proxy) {
  auto sparse = make_2d_sparse_coord_only("sparse");
  std::vector<double> ref{1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2, 3, 0, 3, 0};
  auto edges =
      createVariable<double>(Dims{Dim::Y}, Shape{6}, Values{1, 2, 3, 4, 5, 6});
  auto hist = core::histogram(sparse["sparse"], edges);
  auto expected =
      make_expected(createVariable<double>(Dims{Dim::X, Dim::Y}, Shape{3, 5},
                                           units::Unit(units::counts),
                                           Values(ref.begin(), ref.end()),
                                           Variances(ref.begin(), ref.end())),
                    edges);

  EXPECT_EQ(hist, expected);
}

TEST(HistogramTest, with_data) {
  auto sparse = make_2d_sparse_coord_only("sparse");
  Variable data = makeVariableWithVariances<double>(
      {{Dim::X, 3}, {Dim::Y, Dimensions::Sparse}});
  data.sparseValues<double>()[0] = {1, 1, 1, 2, 2};
  data.sparseValues<double>()[1] = {2, 2, 2, 2, 2};
  data.sparseValues<double>()[2] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  data.sparseVariances<double>()[0] = {1, 1, 1, 2, 2};
  data.sparseVariances<double>()[1] = {2, 2, 2, 2, 2};
  data.sparseVariances<double>()[2] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  data.setUnit(units::counts);
  sparse.setData("sparse", data);
  auto edges =
      createVariable<double>(Dims{Dim::Y}, Shape{6}, Values{1, 2, 3, 4, 5, 6});
  std::vector<double> ref{1, 1, 1, 2, 2, 0, 0, 2, 2, 2, 2, 3, 0, 3, 0};
  auto expected =
      make_expected(createVariable<double>(Dims{Dim::X, Dim::Y}, Shape{3, 5},
                                           units::Unit(units::counts),
                                           Values(ref.begin(), ref.end()),
                                           Variances(ref.begin(), ref.end())),
                    edges);

  EXPECT_EQ(core::histogram(sparse["sparse"], edges), expected);
}

TEST(HistogramTest, dataset) {
  auto sparse = make_2d_sparse_coord_only("a");
  sparse.setData("b", sparse["a"]);
  sparse["b"].coords()[Dim::Y] += 1.0;
  std::vector<double> a{1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2, 3, 0, 3, 0};
  std::vector<double> b{0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2, 3, 0, 3};
  const auto coord =
      createVariable<double>(Dims{Dim::Y}, Shape{6}, Values{1, 2, 3, 4, 5, 6});
  Dataset expected;
  expected.setCoord(Dim::Y, coord);
  expected.setData(
      "a", createVariable<double>(
               Dims{Dim::X, Dim::Y}, Shape{3, 5}, units::Unit(units::counts),
               Values(a.begin(), a.end()), Variances(a.begin(), a.end())));
  expected.setData(
      "b", createVariable<double>(
               Dims{Dim::X, Dim::Y}, Shape{3, 5}, units::Unit(units::counts),
               Values(b.begin(), b.end()), Variances(b.begin(), b.end())));

  EXPECT_EQ(core::histogram(sparse, coord), expected);
}

TEST(HistogramTest, dataset_own_coord) {
  auto sparse = make_2d_sparse_coord_only("a");
  sparse.setData("b", sparse["a"]);
  sparse["b"].coords()[Dim::Y] += 1.0;
  const auto coord =
      createVariable<double>(Dims{Dim::Y}, Shape{6}, Values{1, 2, 3, 4, 5, 6});
  const auto expected = core::histogram(sparse, coord);

  sparse.setCoord(Dim::Y, coord);
  EXPECT_EQ(core::histogram(sparse, Dim::Y), expected);
}
