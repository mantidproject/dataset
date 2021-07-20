// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>
#include <vector>

#include "scipp/core/element_array_view.h"
#include "scipp/core/except.h"
#include "scipp/core/multi_index.h"

using namespace scipp;
using namespace scipp::core;

class MultiIndexTest : public ::testing::Test {
protected:
  template <scipp::index N, class... Indices>
  void check_increment(MultiIndex<N> i,
                       const std::vector<scipp::index> &expected0,
                       const Indices &... expected) const {
    const auto end = i.end();
    for (scipp::index n = 0; n < scipp::size(expected0); ++n) {
      ASSERT_NE(i, end);
      EXPECT_EQ(i.get(), (std::array{expected0[n], expected[n]...}));
      i.increment();
    }
    ASSERT_EQ(i, end);
  }

  template <scipp::index N, class... Indices>
  void check_set_index(MultiIndex<N> i, const scipp::index bin_volume,
                       const std::vector<scipp::index> &expected0,
                       const Indices &... expected) const {
    if (!i.has_bins()) {
      for (scipp::index n = 0; n < scipp::size(expected0); ++n) {
        i.set_index(n);
        auto it = i.begin();
        for (scipp::index k = 0; k < n; ++k) {
          it.increment();
        }
        EXPECT_EQ(i, it); // Checks only for m_coord.
        EXPECT_EQ(i.get(), it.get());
      }
    } else {
      for (scipp::index bin = 0; bin < bin_volume; ++bin) {
        i.set_index(bin);
        scipp::index n = 0;
        // We do not know how many elements there are in each bin.
        // So just increment until we hit the index for the given bin and
        // make sure that we see the correct indices along the way.
        for (auto it = i.begin(); it != i; it.increment(), ++n) {
          ASSERT_EQ(it.get(), (std::array{expected0[n], expected[n]...}));
        }
      }
    }
  }

  template <scipp::index N, class... Indices>
  void check_impl(MultiIndex<N> i, const scipp::index bin_volume,
                  const std::vector<scipp::index> &expected0,
                  const Indices &... expected) const {
    if (scipp::size(expected0) > 0) {
      ASSERT_NE(i.begin(), i.end());
    }
    check_increment(i, expected0, expected...);
    if (i == i.begin()) {
      check_set_index(i, bin_volume, expected0, expected...);
    }
  }
  void check(MultiIndex<1> i, const std::vector<scipp::index> &indices,
             const scipp::index bin_volume = 0) const {
    check_impl(std::move(i), bin_volume, indices);
  }
  void check(MultiIndex<2> i, const std::vector<scipp::index> &indices0,
             const std::vector<scipp::index> &indices1,
             const scipp::index bin_volume = 0) const {
    check_impl(std::move(i), bin_volume, indices0, indices1);
  }
  void check_with_bins(
      const Dimensions &buffer_dims, const Dim slice_dim,
      const std::vector<std::pair<scipp::index, scipp::index>> &indices,
      const Dimensions &iter_dims, const Strides &strides,
      const std::vector<scipp::index> &expected) {
    BucketParams params{slice_dim, buffer_dims, indices.data()};
    MultiIndex<1> index(ElementArrayViewParams{0, iter_dims, strides, params});
    check(index, expected, iter_dims.volume());
  }
  void check_with_bins(
      const Dimensions &buffer_dims0, const Dim slice_dim0,
      const std::vector<std::pair<scipp::index, scipp::index>> &indices0,
      const Dimensions &buffer_dims1, const Dim slice_dim1,
      const std::vector<std::pair<scipp::index, scipp::index>> &indices1,
      const Dimensions &iter_dims, const Strides &strides0,
      const Strides &strides1, const std::vector<scipp::index> &expected0,
      const std::vector<scipp::index> &expected1) {
    BucketParams params0{slice_dim0, buffer_dims0, indices0.data()};
    BucketParams params1{slice_dim1, buffer_dims1, indices1.data()};
    MultiIndex<2> index(
        ElementArrayViewParams{0, iter_dims, strides0, params0},
        ElementArrayViewParams{0, iter_dims, strides1, params1});
    check(index, expected0, expected1, iter_dims.volume());
    // Order of arguments should not matter, in particular this also tests that
    // the dense argument may be the first argument.
    MultiIndex<2> swapped(
        ElementArrayViewParams{0, iter_dims, strides1, params1},
        ElementArrayViewParams{0, iter_dims, strides0, params0});
    check(swapped, expected1, expected0, iter_dims.volume());
  }

  Dimensions x{Dim::X, 2};
  Dimensions y{Dim::Y, 3};
  Dimensions yx{{Dim::Y, Dim::X}, {3, 2}};
  Dimensions xy{{Dim::X, Dim::Y}, {2, 3}};
  Dimensions xz{{Dim::X, Dim::Z}, {2, 4}};
  Dimensions xyz{{Dim::X, Dim::Y, Dim::Z}, {2, 3, 4}};
};

Strides make_strides(const Dimensions &iter_dims, const Dimensions &data_dims) {
  Strides strides;
  scipp::index d = 0;
  for (const auto &dim : iter_dims.labels()) {
    if (data_dims.contains(dim))
      strides[d++] = data_dims.offset(dim);
    else
      strides[d++] = 0;
  }
  return strides;
}

TEST_F(MultiIndexTest, broadcast_inner) {
  check(MultiIndex(xy, make_strides(xy, x)), {0, 0, 0, 1, 1, 1});
}

TEST_F(MultiIndexTest, broadcast_outer) {
  check(MultiIndex(yx, make_strides(yx, x)), {0, 1, 0, 1, 0, 1});
}

TEST_F(MultiIndexTest, slice_inner) {
  check(MultiIndex(x, make_strides(x, xy)), {0, 3});
}

TEST_F(MultiIndexTest, slice_middle) {
  check(MultiIndex(xz, make_strides(xz, xyz)), {0, 1, 2, 3, 12, 13, 14, 15});
}

TEST_F(MultiIndexTest, slice_outer) {
  check(MultiIndex(x, make_strides(x, yx)), {0, 1});
}

TEST_F(MultiIndexTest, 2d) {
  check(MultiIndex(xy, make_strides(xy, xy)), {0, 1, 2, 3, 4, 5});
}

TEST_F(MultiIndexTest, 6d) {
  Dimensions dims({Dim("1"), Dim("2"), Dim("3"), Dim("4"), Dim("5"), Dim("6")},
                  {1, 2, 1, 2, 1, 2});
  MultiIndex i{dims, make_strides(dims, dims)};
  [[maybe_unused]] auto _ = i.end();
  check(i, {0, 1, 2, 3, 4, 5, 6, 7});
}

TEST_F(MultiIndexTest, 2d_transpose) {
  check(MultiIndex<1>(yx, make_strides(yx, xy)), {0, 3, 1, 4, 2, 5});
}

TEST_F(MultiIndexTest, slice_and_broadcast) {
  check(MultiIndex(xz, make_strides(xz, yx)), {0, 0, 0, 0, 1, 1, 1, 1});
  check(MultiIndex(xz, make_strides(xz, xy)), {0, 0, 0, 0, 3, 3, 3, 3});
  check(MultiIndex(yx, make_strides(yx, xz)), {0, 4, 0, 4, 0, 4});
}

TEST_F(MultiIndexTest, multiple_data_indices) {
  check(MultiIndex(yx, make_strides(yx, x), make_strides(yx, y)),
        {0, 1, 0, 1, 0, 1}, {0, 0, 1, 1, 2, 2});
  check(MultiIndex(xy, make_strides(xy, x), make_strides(xy, y)),
        {0, 0, 0, 1, 1, 1}, {0, 1, 2, 0, 1, 2});
  check(MultiIndex(xy, make_strides(xy, yx), make_strides(xy, xy)),
        {0, 2, 4, 1, 3, 5}, {0, 1, 2, 3, 4, 5});
  check(MultiIndex(yx, make_strides(yx, yx), make_strides(yx, xy)),
        {0, 1, 2, 3, 4, 5}, {0, 3, 1, 4, 2, 5});
}

TEST_F(MultiIndexTest, advance_multiple_data_indices) {
  MultiIndex index(yx, make_strides(yx, x), make_strides(yx, y));
  index.set_index(1);
  check(index, {1, 0, 1, 0, 1}, {0, 1, 1, 2, 2});
  index.set_index(2);
  check(index, {0, 1, 0, 1}, {1, 1, 2, 2});
}

TEST_F(MultiIndexTest, advance_slice_middle) {
  MultiIndex index(xz, make_strides(xz, xyz));
  index.set_index(2);
  check(index, {2, 3, 12, 13, 14, 15});
  index.set_index(5);
  check(index, {13, 14, 15});
}

TEST_F(MultiIndexTest, advance_slice_and_broadcast) {
  MultiIndex index(xz, make_strides(xz, xy));
  index.set_index(2);
  check(index, {0, 0, 3, 3, 3, 3});
}

TEST_F(MultiIndexTest, scalar_of_1d_bins) {
  const Dim dim = Dim::Row;
  Dimensions buf{dim, 4};
  check_with_bins(buf, dim, {{0, 4}}, Dimensions{}, Strides{}, {0, 1, 2, 3});
}

TEST_F(MultiIndexTest, 1d_array_of_1d_bins) {
  const Dim dim = Dim::Row;
  Dimensions buf{dim, 7}; // 1d cut into two sections
  // natural order no gaps
  check_with_bins(buf, dim, {{0, 3}, {3, 7}}, x, make_strides(x, x),
                  {0, 1, 2, 3, 4, 5, 6});
  // gap between
  check_with_bins(buf, dim, {{0, 3}, {4, 7}}, x, make_strides(x, x),
                  {0, 1, 2, 4, 5, 6});
  // gap at start
  check_with_bins(buf, dim, {{1, 3}, {3, 7}}, x, make_strides(x, x),
                  {1, 2, 3, 4, 5, 6});
  // out of order
  check_with_bins(buf, dim, {{4, 7}, {0, 4}}, x, make_strides(x, x),
                  {4, 5, 6, 0, 1, 2, 3});
}

TEST_F(MultiIndexTest, 1d_array_of_2d_bins) {
  Dimensions buf{{Dim("a"), Dim("b")}, {2, 3}}; // 2d cut into two sections
  // cut along inner
  check_with_bins(buf, Dim("b"), {{0, 1}, {1, 3}}, x, make_strides(x, x),
                  {0, 3, 1, 2, 4, 5});
  check_with_bins(buf, Dim("b"), {{0, 1}, {2, 3}}, x, make_strides(x, x),
                  {0, 3, 2, 5});
  check_with_bins(buf, Dim("b"), {{1, 2}, {2, 3}}, x, make_strides(x, x),
                  {1, 4, 2, 5});
  check_with_bins(buf, Dim("b"), {{1, 3}, {0, 1}}, x, make_strides(x, x),
                  {1, 2, 4, 5, 0, 3});
  check_with_bins(buf, Dim("b"), {{0, 1}, {1, 1}, {2, 3}}, y,
                  make_strides(y, y), {0, 3, 2, 5});
  // cut along outer
  check_with_bins(buf, Dim("a"), {{0, 1}, {1, 2}}, x, make_strides(x, x),
                  {0, 1, 2, 3, 4, 5});
  check_with_bins(buf, Dim("a"), {{1, 2}, {1, 2}}, x, make_strides(x, x),
                  {3, 4, 5, 3, 4, 5});
  check_with_bins(buf, Dim("a"), {{1, 2}, {0, 1}}, x, make_strides(x, x),
                  {3, 4, 5, 0, 1, 2});
  check_with_bins(buf, Dim("a"), {{0, 1}, {1, 1}, {1, 2}}, y,
                  make_strides(y, y), {0, 1, 2, 3, 4, 5});
}

TEST_F(MultiIndexTest, 2d_array_of_1d_bins) {
  const Dim dim = Dim::Row;
  Dimensions buf{dim, 12}; // 1d cut into xy=2x3 sections
  check_with_bins(buf, dim, {{0, 2}, {2, 4}, {4, 6}, {6, 8}, {8, 10}, {10, 12}},
                  xy, make_strides(xy, xy),
                  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  check_with_bins(buf, dim, {{1, 2}, {2, 4}, {5, 6}, {6, 8}, {8, 10}, {10, 12}},
                  xy, make_strides(xy, xy), {1, 2, 3, 5, 6, 7, 8, 9, 10, 11});
  // transpose
  check_with_bins(buf, dim, {{0, 2}, {2, 4}, {4, 6}, {6, 8}, {8, 10}, {10, 12}},
                  yx, make_strides(yx, xy),
                  {0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11});
  // slice inner
  check_with_bins(buf, dim, {{0, 2}, {2, 4}, {4, 6}, {6, 8}, {8, 10}, {10, 12}},
                  x, make_strides(x, xy), {0, 1, 6, 7});
  // slice outer
  check_with_bins(buf, dim, {{0, 2}, {2, 4}, {4, 6}, {6, 8}, {8, 10}, {10, 12}},
                  y, make_strides(y, xy), {0, 1, 2, 3, 4, 5});
}

TEST_F(MultiIndexTest, 1d_array_of_1d_bins_and_dense) {
  const Dim dim = Dim::Row;
  Dimensions buf{dim, 7}; // 1d cut into two sections
  // natural order no gaps
  check_with_bins(buf, dim, {{0, 3}, {3, 7}}, Dimensions{}, Dim::Invalid, {}, x,
                  make_strides(x, x), make_strides(x, x), {0, 1, 2, 3, 4, 5, 6},
                  {0, 0, 0, 1, 1, 1, 1});
  // gap between
  check_with_bins(buf, dim, {{0, 3}, {4, 7}}, Dimensions{}, Dim::Invalid, {}, x,
                  make_strides(x, x), make_strides(x, x), {0, 1, 2, 4, 5, 6},
                  {0, 0, 0, 1, 1, 1});
  // gap at start
  check_with_bins(buf, dim, {{1, 3}, {3, 7}}, Dimensions{}, Dim::Invalid, {}, x,
                  make_strides(x, x), make_strides(x, x), {1, 2, 3, 4, 5, 6},
                  {0, 0, 1, 1, 1, 1});
  // out of order
  // Note that out of order bin indices is *not* to be confused with
  // reversing a dimension, i.e., we do *not* expect {1,1,1,0,0,0,0} for the
  // dense part.
  check_with_bins(buf, dim, {{4, 7}, {0, 4}}, Dimensions{}, Dim::Invalid, {}, x,
                  make_strides(x, x), make_strides(x, x), {4, 5, 6, 0, 1, 2, 3},
                  {0, 0, 0, 1, 1, 1, 1});
}

TEST_F(MultiIndexTest, 1d_array_of_1d_bins_and_dense_with_empty_bins) {
  const Dim dim = Dim::Row;
  Dimensions buf{dim, 7};
  Dimensions x1{Dim::X, 1};
  check_with_bins(buf, dim, {{0, 0}}, Dimensions{}, Dim::Invalid, {}, x1,
                  make_strides(x1, x1), make_strides(x1, x1), {}, {});
  check_with_bins(buf, dim, {{1, 1}, {0, 0}}, Dimensions{}, Dim::Invalid, {}, x,
                  make_strides(x, x), make_strides(x, x), {}, {});
  check_with_bins(buf, dim, {{0, 0}, {0, 3}}, Dimensions{}, Dim::Invalid, {}, x,
                  make_strides(x, x), make_strides(x, x), {0, 1, 2}, {1, 1, 1});
  check_with_bins(buf, dim, {{0, 0}, {0, 2}, {3, 5}}, Dimensions{},
                  Dim::Invalid, {}, y, make_strides(y, y), make_strides(y, y),
                  {0, 1, 3, 4}, {1, 1, 2, 2});
  check_with_bins(buf, dim, {{0, 2}, {2, 2}, {3, 5}}, Dimensions{},
                  Dim::Invalid, {}, y, make_strides(y, y), make_strides(y, y),
                  {0, 1, 3, 4}, {0, 0, 2, 2});
  check_with_bins(buf, dim, {{0, 2}, {3, 5}, {5, 5}}, Dimensions{},
                  Dim::Invalid, {}, y, make_strides(y, y), make_strides(y, y),
                  {0, 1, 3, 4}, {0, 0, 1, 1});
}

TEST_F(MultiIndexTest, two_1d_arrays_of_1d_bins) {
  const Dim dim = Dim::Row;
  Dimensions buf{dim, 1};
  check_with_bins(buf, dim, {{0, 3}, {3, 7}}, buf, dim, {{4, 7}, {0, 4}}, x,
                  make_strides(x, x), make_strides(x, x), {0, 1, 2, 3, 4, 5, 6},
                  {4, 5, 6, 0, 1, 2, 3});
  // slice inner
  check_with_bins(buf, dim, {{0, 3}, {3, 7}}, buf, dim,
                  {{1, 4}, {5, 9}, {9, 10}, {10, 11}, {11, 12}, {12, 13}}, x,
                  make_strides(x, x), make_strides(x, yx),
                  {0, 1, 2, 3, 4, 5, 6}, {1, 2, 3, 5, 6, 7, 8});
  // slice outer
  check_with_bins(buf, dim, {{0, 3}, {3, 7}}, buf, dim,
                  {{1, 4}, {9, 10}, {10, 11}, {5, 9}, {11, 12}, {12, 13}}, x,
                  make_strides(x, x), make_strides(x, xy),
                  {0, 1, 2, 3, 4, 5, 6}, {1, 2, 3, 5, 6, 7, 8});
  // slice to scalar
  check_with_bins(buf, dim, {{0, 3}}, buf, dim, {{2, 5}, {0, 2}}, Dimensions{},
                  make_strides(Dimensions{}, x), make_strides(Dimensions{}, x),
                  {0, 1, 2}, {2, 3, 4});
}

TEST_F(MultiIndexTest, two_1d_arrays_of_1d_bins_bin_size_mismatch) {
  const Dim dim = Dim::Row;
  Dimensions buf{dim, 7};
  EXPECT_THROW(check_with_bins(buf, dim, {{0, 3}, {3, 7}}, buf, dim,
                               {{0, 4}, {3, 7}}, x, make_strides(x, x),
                               make_strides(x, x), {0, 1, 2, 3, 4, 5, 6},
                               {0, 1, 2, 3, 4, 5, 6}),
               except::BinnedDataError);
  EXPECT_THROW(check_with_bins(buf, dim, {{0, 3}, {3, 7}}, buf, dim,
                               {{0, 3}, {4, 7}}, x, make_strides(x, x),
                               make_strides(x, x), {0, 1, 2, 3, 4, 5, 6},
                               {0, 1, 2, 3, 4, 5, 6}),
               except::BinnedDataError);
}

TEST_F(MultiIndexTest, 2d_empty_dims_array_of_1d_bins) {
  const Dim dim = Dim::Row;
  Dimensions buf{dim, 0}; // 1d cut into dims=0 sections
  Dimensions dims{{Dim::X, 0}};
  check_with_bins(buf, dim, {}, dims, make_strides(dims, dims), {});
}
