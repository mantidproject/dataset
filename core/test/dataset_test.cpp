// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
#include "test_macros.h"
#include <gtest/gtest.h>

#include <numeric>
#include <set>

#include "scipp/core/dataset.h"
#include "scipp/core/dimensions.h"

#include "dataset_test_common.h"

using namespace scipp;
using namespace scipp::core;

// Any dataset functionality that is also available for Dataset(Const)Proxy is
// to be tested in dataset_proxy_test.cpp, not here!

TEST(DatasetTest, construct_default) { ASSERT_NO_THROW(Dataset d); }

TEST(DatasetTest, clear) {
  DatasetFactory3D factory;
  auto dataset = factory.make();

  ASSERT_FALSE(dataset.empty());
  ASSERT_FALSE(dataset.coords().empty());
  ASSERT_FALSE(dataset.labels().empty());
  ASSERT_FALSE(dataset.attrs().empty());
  ASSERT_FALSE(dataset.masks().empty());

  ASSERT_NO_THROW(dataset.clear());

  ASSERT_TRUE(dataset.empty());
  ASSERT_FALSE(dataset.coords().empty());
  ASSERT_FALSE(dataset.labels().empty());
  ASSERT_FALSE(dataset.attrs().empty());
  ASSERT_FALSE(dataset.masks().empty());
}

TEST(DatasetTest, erase_single_non_existant) {
  Dataset d;
  ASSERT_THROW(d.erase("not an item"), except::DatasetError);
}

TEST(DatasetTest, erase_single) {
  DatasetFactory3D factory;
  auto dataset = factory.make();
  ASSERT_NO_THROW(dataset.erase("data_xyz"));
  ASSERT_FALSE(dataset.contains("data_xyz"));
}

TEST(DatasetTest, erase_extents_rebuild) {
  Dataset d;

  d.setData("a", createVariable<double>(Dims{Dim::X}, Shape{10}));
  ASSERT_TRUE(d.contains("a"));

  ASSERT_NO_THROW(d.erase("a"));
  ASSERT_FALSE(d.contains("a"));

  ASSERT_NO_THROW(
      d.setData("a", createVariable<double>(Dims{Dim::X}, Shape{15})));
  ASSERT_TRUE(d.contains("a"));
}

TEST(DatasetTest, setCoord) {
  Dataset d;
  const auto var = createVariable<double>(Dims{Dim::X}, Shape{3});

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

TEST(DatasetTest, setLabels) {
  Dataset d;
  const auto var = createVariable<double>(Dims{Dim::X}, Shape{3});

  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.labels().size(), 0);

  ASSERT_NO_THROW(d.setLabels("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.labels().size(), 1);

  ASSERT_NO_THROW(d.setLabels("b", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.labels().size(), 2);

  ASSERT_NO_THROW(d.setLabels("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.labels().size(), 2);
}

TEST(DatasetTest, setAttr) {
  Dataset d;
  const auto var = createVariable<double>(Dims{Dim::X}, Shape{3});

  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.attrs().size(), 0);

  ASSERT_NO_THROW(d.setAttr("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.attrs().size(), 1);

  ASSERT_NO_THROW(d.setAttr("b", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.attrs().size(), 2);

  ASSERT_NO_THROW(d.setAttr("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.attrs().size(), 2);
}

TEST(DatasetTest, setMask) {
  Dataset d;
  const auto var =
      createVariable<bool>(Dims{Dim::X}, Shape{3}, Values{false, true, false});

  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.masks().size(), 0);

  ASSERT_NO_THROW(d.setMask("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.masks().size(), 1);
  ASSERT_EQ(d.masks()["a"], var);

  ASSERT_NO_THROW(d.setMask("b", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.masks().size(), 2);

  ASSERT_NO_THROW(d.setMask("a", var));
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d.masks().size(), 2);
}

TEST(DatasetTest, setData_with_and_without_variances) {
  Dataset d;
  const auto var = createVariable<double>(Dims{Dim::X}, Shape{3});

  ASSERT_NO_THROW(d.setData("a", var));
  ASSERT_EQ(d.size(), 1);

  ASSERT_NO_THROW(d.setData("b", var));
  ASSERT_EQ(d.size(), 2);

  ASSERT_NO_THROW(d.setData("a", var));
  ASSERT_EQ(d.size(), 2);

  ASSERT_NO_THROW(d.setData("a", createVariable<double>(Dims{Dim::X}, Shape{3},
                                                        Values{1, 1, 1},
                                                        Variances{0, 0, 0})));
  ASSERT_EQ(d.size(), 2);
}

TEST(DatasetTest, setLabels_with_name_matching_data_name) {
  Dataset d;
  d.setData("a", createVariable<double>(Dims{Dim::X}, Shape{3}));
  d.setData("b", createVariable<double>(Dims{Dim::X}, Shape{3}));

  // It is possible to set labels with a name matching data. However, there is
  // no special meaning attached to this. In particular it is *not* linking the
  // labels to that data item.
  ASSERT_NO_THROW(d.setLabels("a", createVariable<double>(Values{double{}})));
  ASSERT_EQ(d.size(), 2);
  ASSERT_EQ(d.labels().size(), 1);
  ASSERT_EQ(d["a"].labels().size(), 1);
  ASSERT_EQ(d["b"].labels().size(), 1);
}

TEST(DatasetTest, setSparseCoord_not_sparse_fail) {
  Dataset d;
  const auto var = createVariable<double>(Dims{Dim::X}, Shape{3});

  ASSERT_ANY_THROW(d.setSparseCoord("a", var));
}

TEST(DatasetTest, setSparseCoord) {
  Dataset d;
  const auto var = createVariable<double>(Dims{Dim::X, Dim::Y},
                                          Shape{3l, Dimensions::Sparse});

  ASSERT_NO_THROW(d.setSparseCoord("a", var));
  ASSERT_EQ(d.size(), 1);
  ASSERT_NO_THROW(d["a"]);
}

TEST(DatasetTest, setSparseLabels_missing_values_or_coord) {
  Dataset d;
  const auto sparse =
      createVariable<double>(Dims{Dim::X}, Shape{Dimensions::Sparse});

  ASSERT_ANY_THROW(d.setSparseLabels("a", "x", sparse));
  d.setSparseCoord("a", sparse);
  ASSERT_NO_THROW(d.setSparseLabels("a", "x", sparse));
}

TEST(DatasetTest, setSparseLabels_not_sparse_fail) {
  Dataset d;
  const auto dense = createVariable<double>(Values{double{}});
  const auto sparse =
      createVariable<double>(Dims{Dim::X}, Shape{Dimensions::Sparse});

  d.setSparseCoord("a", sparse);
  ASSERT_ANY_THROW(d.setSparseLabels("a", "x", dense));
}

TEST(DatasetTest, setSparseLabels) {
  Dataset d;
  const auto sparse =
      createVariable<double>(Dims{Dim::X}, Shape{Dimensions::Sparse});
  d.setSparseCoord("a", sparse);

  ASSERT_NO_THROW(d.setSparseLabels("a", "x", sparse));
  ASSERT_EQ(d.size(), 1);
  ASSERT_NO_THROW(d["a"]);
  ASSERT_EQ(d["a"].labels().size(), 1);
}

TEST(DatasetTest, iterators_return_types) {
  Dataset d;
  ASSERT_TRUE((std::is_same_v<decltype(d.begin()->second), DataProxy>));
  ASSERT_TRUE((std::is_same_v<decltype(d.end()->second), DataProxy>));
}

TEST(DatasetTest, const_iterators_return_types) {
  const Dataset d;
  ASSERT_TRUE((std::is_same_v<decltype(d.begin()->second), DataConstProxy>));
  ASSERT_TRUE((std::is_same_v<decltype(d.end()->second), DataConstProxy>));
}

TEST(DatasetTest, set_dense_data_with_sparse_coord) {
  auto sparse_variable = createVariable<double>(Dims{Dim::Y, Dim::X},
                                                Shape{2l, Dimensions::Sparse});
  auto dense_variable =
      createVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2l, 2l});

  Dataset a;
  a.setData("sparse_coord_and_val", dense_variable);
  ASSERT_THROW(a.setSparseCoord("sparse_coord_and_val", sparse_variable),
               except::DimensionError);

  // Setting coords first yields same response.
  Dataset b;
  b.setSparseCoord("sparse_coord_and_val", sparse_variable);
  ASSERT_THROW(b.setData("sparse_coord_and_val", dense_variable),
               except::DimensionError);
}

TEST(DatasetTest, construct_from_proxy) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  const DatasetConstProxy proxy(dataset);
  Dataset from_proxy(proxy);
  ASSERT_EQ(from_proxy, dataset);
}

TEST(DatasetTest, construct_from_slice) {
  DatasetFactory3D factory;
  const auto dataset = factory.make();
  const auto slice = dataset.slice({Dim::X, 1});
  Dataset from_slice(slice);
  ASSERT_EQ(from_slice, dataset.slice({Dim::X, 1}));
}

TEST(DatasetTest, slice_temporary) {
  DatasetFactory3D factory;
  auto dataset = factory.make().slice({Dim::X, 1});
  ASSERT_TRUE((std::is_same_v<decltype(dataset), Dataset>));
}

template <typename T> void do_test_slice_validation(const T &container) {
  EXPECT_THROW(container.slice(Slice{Dim::Y, 0, 1}), except::SliceError);
  EXPECT_THROW(container.slice(Slice{Dim::X, 0, 3}), except::SliceError);
  EXPECT_THROW(container.slice(Slice{Dim::X, -1, 0}), except::SliceError);
  EXPECT_NO_THROW(container.slice(Slice{Dim::X, 0, 1}));
}

TEST(DatasetTest, slice_validation_simple) {
  Dataset dataset;
  auto var = createVariable<double>(Dims{Dim::X}, Shape{2}, Values{1, 2});
  dataset.setCoord(Dim::X, var);
  do_test_slice_validation(dataset);

  // Make sure correct via const proxies
  DatasetConstProxy constproxy(dataset);
  do_test_slice_validation(constproxy);

  // Make sure correct via proxies
  DatasetProxy proxy(dataset);
  do_test_slice_validation(proxy);
}

TEST(DatasetTest, slice_with_no_coords) {
  Dataset ds;
  auto var = createVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  ds.setData("a", var);
  // No dataset coords. slicing should still work.
  auto slice = ds.slice(Slice{Dim::X, 0, 2});
  auto extents = slice["a"].data().dims()[Dim::X];
  EXPECT_EQ(extents, 2);
}

TEST(DatasetTest, slice_validation_complex) {

  Dataset ds;
  auto var1 =
      createVariable<double>(Dims{Dim::X}, Shape{4}, Values{1, 2, 3, 4});
  ds.setCoord(Dim::X, var1);
  auto var2 =
      createVariable<double>(Dims{Dim::Y}, Shape{4}, Values{1, 2, 3, 4});
  ds.setCoord(Dim::Y, var2);

  // Slice arguments applied in order.
  EXPECT_NO_THROW(ds.slice(Slice{Dim::X, 0, 3}, Slice{Dim::X, 1, 2}));
  // Reverse order. Invalid slice creation should be caught up front.
  EXPECT_THROW(ds.slice(Slice{Dim::X, 1, 2}, Slice{Dim::X, 0, 3}),
               except::SliceError);
}

TEST(DataProxyTest, set_variances) {
  auto d = make_1_values<bool>("a", {Dim::X, 3}, units::m, {true, false, true});
  EXPECT_THROW(d["a"].setVariances(Vector<double>{1, 2, 3}), except::TypeError);

  auto dd = make_1_values<double>("a", {Dim::X, 3}, units::m, {1.0, 1.0, 1.0});
  EXPECT_ANY_THROW(dd["a"].setVariances(Vector<double>{2.0, 2.0, 2.0, 2.0}));
  dd["a"].setVariances(Vector<double>{3.0, 3.0, 3.0});
  EXPECT_EQ(equals(dd["a"].variances<double>(), Vector<double>{3.0, 3.0, 3.0}),
            true);
}

TEST(DatasetTest, sum_and_mean) {
  auto ds = make_1_values_and_variances<float>(
      "a", {Dim::X, 3}, units::dimensionless, {1, 2, 3}, {12, 15, 18});
  EXPECT_EQ(core::sum(ds, Dim::X)["a"].data(),
            createVariable<float>(Values{6}, Variances{45}));
  EXPECT_EQ(core::sum(ds.slice({Dim::X, 0, 2}), Dim::X)["a"].data(),
            createVariable<float>(Values{3}, Variances{27}));

  EXPECT_EQ(core::mean(ds, Dim::X)["a"].data(),
            createVariable<float>(Values{2}, Variances{5.0}));
  EXPECT_EQ(core::mean(ds.slice({Dim::X, 0, 2}), Dim::X)["a"].data(),
            createVariable<float>(Values{1.5}, Variances{6.75}));

  EXPECT_THROW(core::sum(make_sparse_2d({1, 2, 3, 4}, {0, 0}), Dim::X),
               except::DimensionError);
}

TEST(DatasetTest, erase_coord) {
  DatasetFactory3D factory;
  const auto ref = factory.make();
  Dataset ds(ref);
  auto coord = Variable(ds.coords()[Dim::X]);
  ds.eraseCoord(Dim::X);
  EXPECT_FALSE(ds.coords().contains(Dim::X));
  ds.setCoord(Dim::X, coord);
  EXPECT_EQ(ref, ds);

  ds.coords().erase(Dim::X);
  EXPECT_FALSE(ds.coords().contains(Dim::X));
  ds.setCoord(Dim::X, coord);
  EXPECT_EQ(ref, ds);

  const auto var = createVariable<double>(Dims{Dim::X, Dim::Y},
                                          Shape{3l, Dimensions::Sparse});
  ds.setSparseCoord("newCoord", var);
  EXPECT_TRUE(ds["newCoord"].coords().contains(Dim::X));
  ds.eraseSparseCoord("newCoord");
  EXPECT_EQ(ref, ds);

  ds.setSparseCoord("newCoord", var);
  ds.setData("newCoord",
             createVariable<double>(Dims{Dim::X}, Shape{Dimensions::Sparse}));
  EXPECT_TRUE(ds["newCoord"].coords().contains(Dim::X));
  EXPECT_THROW(ds["newCoord"].coords().erase(Dim::Z), except::SparseDataError);
  ds["newCoord"].coords().erase(Dim::X);
  EXPECT_FALSE(ds["newCoord"].coords().contains(Dim::X));
}

TEST(DatasetTest, erase_labels) {
  DatasetFactory3D factory;
  const auto ref = factory.make();
  Dataset ds(ref);
  auto labels = Variable(ds.labels()["labels_x"]);
  ds.eraseLabels("labels_x");
  EXPECT_FALSE(ds.labels().contains("labels_x"));
  ds.setLabels("labels_x", labels);
  EXPECT_EQ(ref, ds);

  ds.labels().erase("labels_x");
  EXPECT_FALSE(ds.labels().contains("labels_x"));
  ds.setLabels("labels_x", labels);
  EXPECT_EQ(ref, ds);

  const auto var = createVariable<double>(Dims{Dim::X, Dim::Y},
                                          Shape{3l, Dimensions::Sparse});
  ds.setSparseCoord("newCoord", var);
  ds.setSparseLabels("newCoord", "labels_sparse", var);
  EXPECT_TRUE(ds["newCoord"].labels().contains("labels_sparse"));
  ds.eraseSparseLabels("newCoord", "labels_sparse");
  EXPECT_FALSE(ds["newCoord"].labels().contains("labels_sparse"));

  ds.setSparseLabels("newCoord", "labels_sparse", var);
  EXPECT_TRUE(ds["newCoord"].labels().contains("labels_sparse"));
  ds["newCoord"].labels().erase("labels_sparse");
  EXPECT_FALSE(ds["newCoord"].labels().contains("labels_sparse"));
}

TEST(DatasetTest, erase_attrs) {
  DatasetFactory3D factory;
  const auto ref = factory.make();
  Dataset ds(ref);
  auto attrs = Variable(ds.attrs()["attr_x"]);
  ds.eraseAttr("attr_x");
  EXPECT_FALSE(ds.attrs().contains("attr_x"));
  ds.setAttr("attr_x", attrs);
  EXPECT_EQ(ref, ds);

  ds.attrs().erase("attr_x");
  EXPECT_FALSE(ds.attrs().contains("attr_x"));
  ds.setAttr("attr_x", attrs);
  EXPECT_EQ(ref, ds);
}

TEST(DatasetTest, erase_masks) {
  DatasetFactory3D factory;
  const auto ref = factory.make();
  Dataset ds(ref);
  auto mask = Variable(ref.masks()["masks_x"]);
  ds.eraseMask("masks_x");
  EXPECT_FALSE(ds.masks().contains("masks_x"));
  ds.setMask("masks_x", mask);
  EXPECT_EQ(ref, ds);

  ds.masks().erase("masks_x");
  EXPECT_FALSE(ds.masks().contains("masks_x"));
  ds.setMask("masks_x", mask);
  EXPECT_EQ(ref, ds);
}
