// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/dataset/data_array.h"
#include "scipp/dataset/except.h"
#include "scipp/dataset/util.h"
#include "scipp/variable/operations.h"

#include "dataset_test_common.h"
#include "test_macros.h"

using namespace scipp;
using namespace scipp::dataset;

TEST(DataArrayTest, construct) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();

  DataArray array(dataset["data_xyz"]);
  EXPECT_EQ(array, dataset["data_xyz"]);
  // Comparison ignores the name, so this is tested separately.
  EXPECT_EQ(array.name(), "data_xyz");
}

TEST(DataArrayTest, construct_fail) {
  // Invalid data
  EXPECT_THROW(DataArray(Variable{}), std::runtime_error);
}

TEST(DataArrayTest, setName) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  DataArray array(dataset["data_xyz"]);

  array.setName("newname");
  EXPECT_EQ(array.name(), "newname");
}

TEST(DataArrayTest, erase_coord) {
  const auto var = makeVariable<double>(Values{1});
  DataArray a(var);
  a.coords().set(Dim::X, var);
  EXPECT_THROW(a.attrs().erase(Dim::X), except::NotFoundError);
  EXPECT_NO_THROW(a.coords().erase(Dim::X));
  a.attrs().set(Dim::X, var);
  EXPECT_NO_THROW(a.attrs().erase(Dim::X));
  a.attrs().set(Dim::X, var);
  EXPECT_THROW(a.coords().erase(Dim::X), except::NotFoundError);
}

TEST(DataArrayTest, shadow_attr) {
  const auto var1 = 1.0 * units::m;
  const auto var2 = 2.0 * units::m;
  DataArray a(0.0 * units::m);
  a.coords().set(Dim::X, var1);
  a.attrs().set(Dim::X, var2);
  EXPECT_EQ(a.coords()[Dim::X], var1);
  EXPECT_EQ(a.attrs()[Dim::X], var2);
  EXPECT_THROW_DISCARD(a.meta(), except::DataArrayError);
  a.attrs().erase(Dim::X);
  EXPECT_EQ(a.meta()[Dim::X], var1);
}

TEST(DataArrayTest, sum_dataset_columns_via_DataArray) {
  DatasetFactory3D factory;
  auto dataset = factory.make();

  DataArray array(dataset["data_zyx"]);
  auto sum = array + dataset["data_xyz"];

  dataset["data_zyx"] += dataset["data_xyz"];

  // This would fail if the data items had attributes, since += preserves them
  // but + does not.
  EXPECT_EQ(sum, dataset["data_zyx"]);
}

TEST(DataArrayTest, fail_op_non_matching_coords) {
  auto coord_1 = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3});
  auto coord_2 = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 4});
  auto data = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 4});
  DataArray da_1(data, {{Dim::X, coord_1}, {Dim::Y, data}});
  DataArray da_2(data, {{Dim::X, coord_2}, {Dim::Y, data}});
  EXPECT_THROW_DISCARD(da_1 + da_2, except::CoordMismatchError);
  EXPECT_THROW_DISCARD(da_1 - da_2, except::CoordMismatchError);
}

TEST(DataArrayTest, astype) {
  DataArray a(
      makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}),
      {{Dim::X, makeVariable<int>(Dims{Dim::X}, Shape{3}, Values{4, 5, 6})}});
  const auto x = astype(a, dtype<double>);
  EXPECT_EQ(x.data(),
            makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1., 2., 3.}));
}
