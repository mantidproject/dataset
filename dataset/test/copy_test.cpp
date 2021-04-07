// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/dataset/dataset.h"
#include "scipp/variable/arithmetic.h"

#include "dataset_test_common.h"

using namespace scipp;
using namespace scipp::dataset;

struct CopyTest : public ::testing::Test {
  CopyTest() : dataset(factory.make()), array(dataset["data_xyz"]) {
    array.attrs().set(Dim("attr"), attr);
  }

protected:
  DatasetFactory3D factory;
  Dataset dataset;
  DataArray array;
  Variable attr = makeVariable<double>(Values{1});
};

TEST_F(CopyTest, data_array) { EXPECT_EQ(copy(array), array); }
TEST_F(CopyTest, dataset) { EXPECT_EQ(copy(dataset), dataset); }

TEST_F(CopyTest, data_array_drop_attrs) {
  auto copied = copy(array, AttrPolicy::Drop);

  EXPECT_NE(copied, array);
  copied.attrs().set(Dim("attr"), attr);
  EXPECT_EQ(copied, array);
}

TEST_F(CopyTest, dataset_drop_attrs) {
  // not implemented yet
  EXPECT_ANY_THROW(auto _ = copy(dataset, AttrPolicy::Drop));
}

struct CopyOutArgTest : public CopyTest {
  CopyOutArgTest() : dataset_copy(copy(dataset)), array_copy(copy(array)) {
    const auto one = 1.0 * units::one;
    array_copy.data() += one;
    array_copy.coords()[Dim::X] += one;
    array_copy.coords()[Dim::Y] += one;
    copy(~array_copy.masks()["masks_x"], array_copy.masks()["masks_x"]);
    array_copy.attrs()[Dim("attr")] += one;
    EXPECT_NE(array_copy, array);
    dataset_copy["data_xyz"].data() += one;
    dataset_copy["data_xyz"].attrs()[Dim("attr")] += one;
    dataset_copy.coords()[Dim::X] += one;
    dataset_copy.coords()[Dim::Y] += one;
    copy(~array_copy.masks()["masks_x"],
         dataset_copy["data_xyz"].masks()["masks_x"]);
    EXPECT_NE(dataset_copy, dataset);
  }

protected:
  Dataset dataset_copy;
  DataArray array_copy;
};

TEST_F(CopyOutArgTest, data_array_out_arg) {
  // copy with out arg also copies coords, masks, and attrs
  EXPECT_EQ(copy(array, array_copy), array);
  EXPECT_EQ(array_copy, array);
}

TEST_F(CopyOutArgTest, dataset_out_arg) {
  // copy with out arg also copies coords, masks, and attrs
  EXPECT_EQ(copy(dataset, dataset_copy), dataset);
  EXPECT_EQ(dataset_copy, dataset);
}

TEST_F(CopyOutArgTest, data_array_out_arg_drop_attrs) {
  copy(array.attrs()[Dim("attr")], array_copy.attrs()[Dim("attr")]);
  // copy with out arg also copies coords, masks, and attrs
  EXPECT_EQ(copy(array, array_copy, AttrPolicy::Drop), array);
  EXPECT_EQ(array_copy, array);
}

TEST_F(CopyOutArgTest, dataset_out_arg_drop_attrs) {
  copy(dataset["data_xyz"].attrs()[Dim("attr")],
       dataset_copy["data_xyz"].attrs()[Dim("attr")]);
  // copy with out arg also copies coords, masks, and attrs
  EXPECT_EQ(copy(dataset, dataset_copy, AttrPolicy::Drop), dataset);
  EXPECT_EQ(dataset_copy, dataset);
}

TEST_F(CopyOutArgTest, data_array_out_arg_drop_attrs_untouched) {
  // copy with out arg leaves items in output that are not in the input
  // untouched. This also applies to dropped attributes.
  EXPECT_NE(copy(array, array_copy, AttrPolicy::Drop), array);
  EXPECT_NE(array_copy, array);
  copy(array.attrs()[Dim("attr")], array_copy.attrs()[Dim("attr")]);
  EXPECT_EQ(array_copy, array);
}

TEST_F(CopyOutArgTest, dataset_out_arg_drop_attrs_untouched) {
  // copy with out arg leaves items in output that are not in the input
  // untouched. This also applies to dropped attributes.
  EXPECT_NE(copy(dataset, dataset_copy, AttrPolicy::Drop), dataset);
  EXPECT_NE(dataset_copy, dataset);
  copy(dataset["data_xyz"].attrs()[Dim("attr")],
       dataset_copy["data_xyz"].attrs()[Dim("attr")]);
  EXPECT_EQ(dataset_copy, dataset);
}
