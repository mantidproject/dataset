// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include "scipp/common/index.h"
#include "scipp/core/except.h"
#include "test_macros.h"
#include <gmock/gmock.h>
#include <gtest/gtest-matchers.h>
#include <gtest/gtest.h>

#include <numeric>
#include <set>

#include "scipp/core/dimensions.h"
#include "scipp/dataset/dataset.h"
#include "scipp/dataset/except.h"
#include "scipp/variable/operations.h"

#include "dataset_test_common.h"
#include "test_data_arrays.h"

using namespace scipp;

// Any dataset functionality that is also available for Dataset(Const)View is
// to be tested in dataset_view_test.cpp, not here!

TEST(DatasetTest, construct_default) { ASSERT_NO_THROW(Dataset d); }

TEST(DatasetTest, clear) {
  DatasetFactory3D factory;
  auto dataset = factory.make();

  ASSERT_FALSE(dataset.empty());
  ASSERT_FALSE(dataset.coords().empty());

  ASSERT_NO_THROW(dataset.clear());

  ASSERT_TRUE(dataset.empty());
  ASSERT_FALSE(dataset.coords().empty());
}

TEST(DatasetTest, erase_non_existant) {
  Dataset d;
  ASSERT_THROW(d.erase("not an item"), except::NotFoundError);
  ASSERT_THROW_DISCARD(d.extract("not an item"), except::NotFoundError);
}

TEST(DatasetTest, erase) {
  DatasetFactory3D factory;
  auto dataset = factory.make();
  ASSERT_NO_THROW(dataset.erase("data_xyz"));
  ASSERT_FALSE(dataset.contains("data_xyz"));
}

TEST(DatasetTest, extract) {
  DatasetFactory3D factory;
  auto dataset = factory.make();
  auto reference = copy(dataset);

  auto ptr = dataset["data_xyz"].values<double>().data();
  auto array = dataset.extract("data_xyz");
  EXPECT_EQ(array.values<double>().data(), ptr);

  ASSERT_FALSE(dataset.contains("data_xyz"));
  EXPECT_EQ(array, reference["data_xyz"]);
  reference.erase("data_xyz");
  EXPECT_EQ(dataset, reference);
}

TEST(DatasetTest, erase_extents_rebuild) {
  Dataset d;

  d.setData("a", makeVariable<double>(Dims{Dim::X}, Shape{10}));
  ASSERT_TRUE(d.contains("a"));

  ASSERT_NO_THROW(d.erase("a"));
  ASSERT_FALSE(d.contains("a"));

  ASSERT_NO_THROW(
      d.setData("a", makeVariable<double>(Dims{Dim::X}, Shape{15})));
  ASSERT_TRUE(d.contains("a"));
}

TEST(DatasetTest, extract_extents_rebuild) {
  Dataset d;

  d.setData("a", makeVariable<double>(Dims{Dim::X}, Shape{10}));
  ASSERT_TRUE(d.contains("a"));

  ASSERT_NO_THROW(auto _ = d.extract("a"));
  ASSERT_FALSE(d.contains("a"));

  ASSERT_NO_THROW(
      d.setData("a", makeVariable<double>(Dims{Dim::X}, Shape{15})));
  ASSERT_TRUE(d.contains("a"));
}

TEST(DatasetTest, setCoord) {
  Dataset d;
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{3});

  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.coords().size(), 0);

  ASSERT_NO_THROW(d.setCoord(Dim::X, var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.coords().size(), 1);

  ASSERT_NO_THROW(d.setCoord(Dim::Y, var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.coords().size(), 2);

  ASSERT_NO_THROW(d.setCoord(Dim::X, var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.coords().size(), 2);
}

TEST(DatasetTest, setCoord_grow) {
  const auto var3 = makeVariable<double>(Dims{Dim::X}, Shape{3});
  const auto var4 = makeVariable<double>(Dims{Dim::X}, Shape{4});
  Dataset d;
  ASSERT_NO_THROW(d.setCoord(Dim::X, var3));
  ASSERT_NO_THROW(d.setCoord(Dim::Y, var4));
}

TEST(DatasetTest, setCoord_shrink) {
  const auto var3 = makeVariable<double>(Dims{Dim::X}, Shape{3});
  const auto var4 = makeVariable<double>(Dims{Dim::X}, Shape{4});
  Dataset d;
  ASSERT_NO_THROW(d.setCoord(Dim::X, var4));
  ASSERT_THROW(d.setCoord(Dim::Y, var3), except::DimensionError);
}

TEST(DatasetTest, set_item_mask) {
  Dataset d;
  d.setData("x", makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3}));
  d.setData("scalar", 1.2 * units::one);
  const auto var =
      makeVariable<bool>(Dims{Dim::X}, Shape{3}, Values{false, true, false});
  d["x"].masks().set("mask", var);
  EXPECT_TRUE(d["x"].masks().contains("mask"));
}

TEST(DatasetTest, setData_with_and_without_variances) {
  Dataset d;
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{3});

  ASSERT_NO_THROW(d.setData("a", var));
  ASSERT_EQ(d.size(), 1);

  ASSERT_NO_THROW(d.setData("b", var));
  ASSERT_EQ(d.size(), 2);

  ASSERT_NO_THROW(d.setData("a", var));
  ASSERT_EQ(d.size(), 2);

  ASSERT_NO_THROW(d.setData("a", makeVariable<double>(Dims{Dim::X}, Shape{3},
                                                      Values{1, 1, 1},
                                                      Variances{0, 0, 0})));
  ASSERT_EQ(d.size(), 2);
}

void check_array_shared(Dataset &ds, const std::string &name,
                        const DataArray &array,
                        const bool shared_coord = true) {
  EXPECT_EQ(ds[name], array);
  // Data and meta data are shared
  EXPECT_TRUE(ds[name].data().is_same(array.data()));
  EXPECT_EQ(ds[name].coords()[Dim::X].is_same(array.coords()[Dim::X]),
            shared_coord);
  EXPECT_TRUE(ds[name].masks()["mask"].is_same(array.masks()["mask"]));
  EXPECT_TRUE(
      ds[name].attrs()[Dim("attr")].is_same(array.attrs()[Dim("attr")]));
  // Metadata *dicts* are not shared
  ds.coords().erase(Dim::X);
  EXPECT_NE(ds[name].coords(), array.coords());
  EXPECT_TRUE(array.coords().contains(Dim::X));
  ds[name].masks().erase("mask");
  EXPECT_NE(ds[name].masks(), array.masks());
  EXPECT_TRUE(array.masks().contains("mask"));
  ds[name].attrs().erase(Dim("attr"));
  EXPECT_NE(ds[name].attrs(), array.attrs());
  EXPECT_TRUE(array.attrs().contains(Dim("attr")));
}

TEST(DatasetTest, setData_from_DataArray) {
  const auto array = make_data_array_1d();
  Dataset ds;
  ds.setData("a", array);
  check_array_shared(ds, "a", array);
}

TEST(DatasetTest, setData_from_DataArray_replace) {
  const auto array1 = make_data_array_1d(0);
  const auto array2 = make_data_array_1d(1);
  const auto original = copy(array1);
  Dataset ds;
  ds.setData("a", array1);
  ds.setData("a", array2);
  EXPECT_EQ(array1, original);     // setData does not copy elements
  const bool shared_coord = false; // coord exists in dataset, not replaced
  check_array_shared(ds, "a", array2, shared_coord);
}

TEST(DatasetTest, setData_updates_dimensions) {
  const auto xy = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 3});
  const auto x = makeVariable<double>(Dims{Dim::X}, Shape{2});

  Dataset d;
  d.setData("x", xy);
  d.setData("x", x);

  const auto dims = d.sizes();
  ASSERT_TRUE(dims.contains(Dim::X));
  // Dim::Y should no longer appear in dimensions after item "x" was replaced.
  ASSERT_FALSE(dims.contains(Dim::Y));
}

TEST(DatasetTest, setData_clears_attributes) {
  const auto var = makeVariable<double>(Values{1});
  Dataset d;
  d.setData("x", var);
  d["x"].attrs().set(Dim("attr"), var);

  EXPECT_TRUE(d["x"].attrs().contains(Dim("attr")));
  d.setData("x", var);
  EXPECT_FALSE(d["x"].attrs().contains(Dim("attr")));
}

TEST(DatasetTest, setData_keep_attributes) {
  const auto var = makeVariable<double>(Values{1});
  Dataset d;
  d.setData("x", var);
  d["x"].attrs().set(Dim("attr"), var);

  EXPECT_TRUE(d["x"].attrs().contains(Dim("attr")));
  d.setData("x", var, AttrPolicy::Keep);
  EXPECT_TRUE(d["x"].attrs().contains(Dim("attr")));
}

TEST(DatasetTest, setData_with_mismatched_dims) {
  scipp::index expected_size = 2;
  const auto original =
      makeVariable<double>(Dims{Dim::X}, Shape{expected_size});
  const auto mismatched =
      makeVariable<double>(Dims{Dim::X}, Shape{expected_size + 1});
  Dataset d;

  ASSERT_NO_THROW(d.setData("a", original));
  ASSERT_THROW(d.setData("a", mismatched), except::DimensionError);
}

TEST(DatasetTest, DataArrayView_setData) {
  const auto var = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1, 2});
  Dataset d;
  d.setData("a", var);
  d.setData("b", var);

  EXPECT_THROW(d["a"].setData(makeVariable<double>(Dims{Dim::X}, Shape{4})),
               except::DimensionError);
  EXPECT_EQ(d["a"].data(), var);
  EXPECT_NO_THROW(d["a"].setData(var + var));
  EXPECT_EQ(d["a"].data(), var + var);
}

struct SetDataTest : public ::testing::Test {
protected:
  Variable var = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1, 2});
  Variable y = makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{1, 3});
  DataArray data{var, {{Dim::Y, var}}};
};

TEST(DatasetTest, setCoord_with_name_matching_data_name) {
  Dataset d;
  d.setData("a", makeVariable<double>(Dims{Dim::X}, Shape{3}));
  d.setData("b", makeVariable<double>(Dims{Dim::X}, Shape{3}));

  // It is possible to set labels with a name matching data. However, there is
  // no special meaning attached to this. In particular it is *not* linking the
  // labels to that data item.
  ASSERT_NO_THROW(d.setCoord(Dim("a"), makeVariable<double>(Values{double{}})));
  ASSERT_EQ(d.size(), 2);
  ASSERT_EQ(d.coords().size(), 1);
  ASSERT_EQ(d["a"].coords().size(), 1);
  ASSERT_EQ(d["b"].coords().size(), 1);
}

TEST(DatasetTest, iterators_return_types) {
  Dataset d;
  ASSERT_TRUE((std::is_same_v<decltype(*d.begin()), DataArray>));
  ASSERT_TRUE((std::is_same_v<decltype(*d.end()), DataArray>));
}

TEST(DatasetTest, const_iterators_return_types) {
  const Dataset d;
  ASSERT_TRUE((std::is_same_v<decltype(*d.begin()), DataArray>));
  ASSERT_TRUE((std::is_same_v<decltype(*d.end()), DataArray>));
}

TEST(DatasetTest, iterators) {
  DataArray da1(makeVariable<double>(Dims{Dim::X}, Shape{2}));
  DataArray da2(makeVariable<double>(Dims{Dim::Y}, Shape{2}));
  da2.coords().set(Dim::Y, makeVariable<double>(Dims{Dim::Y}, Shape{2}));
  Dataset d;
  d.setData("data1", da1);
  d.setData("data2", da2);

  const std::vector data_arrays{std::ref(da1), std::ref(da2)};
  for (auto it = d.begin(); it != d.end(); ++it) {
    EXPECT_THAT(data_arrays, ::testing::Contains(*it));
  }

  const std::vector names{"data1", "data2"};
  for (auto it = d.keys_begin(); it != d.keys_end(); ++it) {
    EXPECT_THAT(names, ::testing::Contains(*it));
  }

  for (auto it = d.items_begin(); it != d.items_end(); ++it) {
    EXPECT_THAT(names, ::testing::Contains(it->first));
    EXPECT_THAT(data_arrays, ::testing::Contains(it->second));
  }
}

TEST(DatasetTest, slice_temporary) {
  DatasetFactory3D factory;
  auto dataset = factory.make().slice({Dim::X, 1});
  ASSERT_TRUE((std::is_same_v<decltype(dataset), Dataset>));
}

TEST(DatasetTest, construct_from_slice) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  const auto slice = dataset.slice({Dim::X, 1});
  Dataset from_slice = copy(slice);
  ASSERT_EQ(from_slice, dataset.slice({Dim::X, 1}));
}

TEST(DatasetTest, construct_dataarray_from_slice) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  const auto slice = dataset["data_xyz"].slice({Dim::X, 1});
  DataArray from_slice = copy(slice);
  ASSERT_EQ(from_slice, dataset["data_xyz"].slice({Dim::X, 1}));
}

TEST(DatasetTest, slice_no_data) {
  Dataset d;
  d.setCoord(Dim::X, makeVariable<double>(Dims{Dim::X}, Shape{4}));
  EXPECT_TRUE(d.coords().contains(Dim::X));
  const auto slice = d.slice({Dim::X, 1, 3});
  EXPECT_TRUE(slice.coords().contains(Dim::X));
}

TEST(DatasetTest, slice_validation_simple) {
  Dataset dataset;
  // TODO this fails length 2, since setCoords detects bin edges and does not
  // add dim
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{3}, Values{1, 2, 3});
  dataset.setCoord(Dim::X, var);
  EXPECT_THROW(dataset.slice(Slice{Dim::Y, 0, 1}), except::SliceError);
  EXPECT_THROW(dataset.slice(Slice{Dim::X, 0, 4}), except::SliceError);
  EXPECT_THROW(dataset.slice(Slice{Dim::X, -1, 0}), except::SliceError);
  EXPECT_NO_THROW(dataset.slice(Slice{Dim::X, 0, 1}));
}

TEST(DatasetTest, slice_with_no_coords) {
  Dataset ds;
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  ds.setData("a", var);
  // No dataset coords. slicing should still work.
  auto slice = ds.slice(Slice{Dim::X, 0, 2});
  auto extents = slice["a"].data().dims()[Dim::X];
  EXPECT_EQ(extents, 2);
}

TEST(DatasetTest, slice_validation_complex) {
  Dataset ds;
  auto var1 = makeVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  ds.setCoord(Dim::X, var1);
  auto var2 = makeVariable<double>(Dims{Dim::Y}, Shape{4}, Values{1, 2, 3, 4});
  ds.setCoord(Dim::Y, var2);

  // Slice arguments applied in order.
  EXPECT_NO_THROW(ds.slice(Slice{Dim::X, 0, 3}).slice(Slice{Dim::X, 1, 2}));
  // Reverse order. Invalid slice creation should be caught up front.
  EXPECT_THROW(ds.slice(Slice{Dim::X, 1, 2}).slice(Slice{Dim::X, 0, 3}),
               except::SliceError);
}

TEST(DatasetTest, extract_coord) {
  DatasetFactory3D factory;
  const auto ref = factory.make();
  Dataset ds = copy(ref);
  auto coord = ds.coords()[Dim::X];
  auto ptr = ds.coords()[Dim::X].values<double>().data();
  auto var = ds.coords().extract(Dim::X);
  EXPECT_EQ(var.values<double>().data(), ptr);
  EXPECT_FALSE(ds.coords().contains(Dim::X));
  ds.setCoord(Dim::X, coord);
  EXPECT_EQ(ref, ds);

  ds.coords().erase(Dim::X);
  EXPECT_FALSE(ds.coords().contains(Dim::X));
  ds.setCoord(Dim::X, coord);
  EXPECT_EQ(ref, ds);
}

TEST(DatasetTest, cannot_set_or_erase_item_coord) {
  DatasetFactory3D factory;
  auto ds = factory.make();
  ASSERT_TRUE(ds.contains("data_x"));
  ASSERT_THROW(ds["data_x"].coords().erase(Dim::X), except::DataArrayError);
  ASSERT_TRUE(ds.coords().contains(Dim::X));
  ASSERT_THROW(ds["data_x"].coords().set(Dim("new"), ds.coords()[Dim::X]),
               except::DataArrayError);
  ASSERT_FALSE(ds.coords().contains(Dim("new")));
}

TEST(DatasetTest, item_coord_cannot_change_coord) {
  DatasetFactory3D factory;
  Dataset ds = factory.make();
  const auto original = copy(ds.coords()[Dim::X]);
  ASSERT_THROW(ds["data_x"].coords()[Dim::X] += original,
               except::VariableError);
  ASSERT_EQ(ds.coords()[Dim::X], original);
}

TEST(DatasetTest, extract_labels) {
  DatasetFactory3D factory;
  const auto ref = factory.make();
  Dataset ds = copy(ref);
  auto labels = ds.coords()[Dim("labels_x")];
  ds.coords().extract(Dim("labels_x"));
  EXPECT_FALSE(ds.coords().contains(Dim("labels_x")));
  ds.setCoord(Dim("labels_x"), labels);
  EXPECT_EQ(ref, ds);

  ds.coords().erase(Dim("labels_x"));
  EXPECT_FALSE(ds.coords().contains(Dim("labels_x")));
  ds.setCoord(Dim("labels_x"), labels);
  EXPECT_EQ(ref, ds);
}

TEST(DatasetTest, set_erase_item_attr) {
  DatasetFactory3D factory;
  auto ds = factory.make();
  const auto attr = makeVariable<double>(Values{1.0});
  ds["data_x"].attrs().set(Dim("item-attr"), attr);
  EXPECT_TRUE(ds["data_x"].attrs().contains(Dim("item-attr")));
  ds["data_x"].attrs().erase(Dim("item-attr"));
  EXPECT_FALSE(ds["data_x"].attrs().contains(Dim("item-attr")));
}

TEST(DatasetTest, set_erase_item_mask) {
  DatasetFactory3D factory;
  auto ds = factory.make();
  const auto mask = makeVariable<double>(Values{true});
  ds["data_x"].masks().set("item-mask", mask);
  EXPECT_TRUE(ds["data_x"].masks().contains("item-mask"));
  ds["data_x"].masks().erase("item-mask");
  EXPECT_FALSE(ds["data_x"].masks().contains("item-mask"));
}

TEST(DatasetTest, item_name) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  DataArray array(dataset["data_xyz"]);
  EXPECT_EQ(array, dataset["data_xyz"]);
  // Comparison ignores the name, so this is tested separately.
  EXPECT_EQ(dataset["data_xyz"].name(), "data_xyz");
  EXPECT_EQ(array.name(), "data_xyz");
}

TEST(DatasetTest, self_nesting) {
  const auto make_dset = [](const std::string &name, const Variable &var) {
    Dataset dset;
    dset.setData(name, var);
    return dset;
  };
  auto inner = make_dset(
      "data", makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1, 2}));
  Variable var = makeVariable<Dataset>(Values{inner});

  auto nested_in_data = make_dset("nested", var);
  ASSERT_THROW_DISCARD(var.value<Dataset>() = nested_in_data,
                       std::invalid_argument);

  Dataset nested_in_coord;
  nested_in_coord.coords().set(Dim::X, var);
  ASSERT_THROW_DISCARD(var.value<Dataset>() = nested_in_coord,
                       std::invalid_argument);
}

struct DatasetRenameTest : public ::testing::Test {
  DatasetRenameTest() {
    DatasetFactory3D factory(4, 5, 6, Dim::X);
    factory.seed(0);
    d = factory.make();
    original = d;
  }

protected:
  Dataset d;
  Dataset original;
};

TEST_F(DatasetRenameTest, fail_duplicate_dim) {
  ASSERT_THROW(d.rename(Dim::X, Dim::Y), except::DimensionError);
  ASSERT_EQ(d, original);
}

TEST_F(DatasetRenameTest, existing) {
  ASSERT_NO_THROW(d.rename(Dim::X, Dim::X));
  ASSERT_EQ(d, original);
}

TEST_F(DatasetRenameTest, back_and_forth) {
  d.rename(Dim::X, Dim::Row);
  EXPECT_NE(d, original);
  d.rename(Dim::Row, Dim::X);
  EXPECT_EQ(d, original);
}

TEST_F(DatasetRenameTest, rename) {
  d.rename(Dim::X, Dim::Row);
  DatasetFactory3D factory(4, 5, 6, Dim::Row);
  factory.seed(0);
  d.coords().set(Dim::Row, d.coords().extract(Dim::X));
  EXPECT_EQ(d, factory.make());
}
