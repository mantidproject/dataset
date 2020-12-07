// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
#include "fix_typed_test_suite_warnings.h"
#include "scipp/dataset/reduction.h"
#include <gtest/gtest.h>
#include <scipp/common/overloaded.h>

namespace {
using namespace scipp;
using namespace scipp::dataset;

using MeanTestTypes = testing::Types<int32_t, int64_t, float, double>;
template <typename T> class MeanTest : public ::testing::Test {};
TYPED_TEST_SUITE(MeanTest, MeanTestTypes);

template <class T, class T2>
auto make_1_values_and_variances(const std::string &name,
                                 const Dimensions &dims, const units::Unit unit,
                                 const std::initializer_list<T2> &values,
                                 const std::initializer_list<T2> &variances) {
  auto d = Dataset();
  if constexpr (std::numeric_limits<T>::is_integer)
    d.setData(name, makeVariable<T>(Dimensions(dims), units::Unit(unit),
                                    Values(values)));
  else
    d.setData(name, makeVariable<T>(Dimensions(dims), units::Unit(unit),
                                    Values(values), Variances(variances)));
  return d;
}

template <typename Op> void test_masked_data_array_1_mask(Op op) {
  const auto var = makeVariable<double>(Dimensions{{Dim::Y, 2}, {Dim::X, 2}},
                                        units::m, Values{1.0, 2.0, 3.0, 4.0});
  const auto mask =
      makeVariable<bool>(Dimensions{Dim::X, 2}, Values{false, true});
  DataArray a(var);
  a.masks().set("mask", mask);
  const auto meanX =
      makeVariable<double>(Dimensions{Dim::Y, 2}, units::m, Values{1.0, 3.0});
  const auto meanY =
      makeVariable<double>(Dimensions{Dim::X, 2}, units::m, Values{2.0, 3.0});
  EXPECT_EQ(op(a, Dim::X).data(), meanX);
  EXPECT_EQ(op(a, Dim::Y).data(), meanY);
  EXPECT_FALSE(op(a, Dim::X).masks().contains("mask"));
  EXPECT_TRUE(op(a, Dim::Y).masks().contains("mask"));
}

template <typename Op> void test_masked_data_array_2_masks(Op op) {
  const auto var = makeVariable<double>(Dimensions{{Dim::Y, 2}, {Dim::X, 2}},
                                        units::m, Values{1.0, 2.0, 3.0, 4.0});
  const auto maskX =
      makeVariable<bool>(Dimensions{Dim::X, 2}, Values{false, true});
  const auto maskY =
      makeVariable<bool>(Dimensions{Dim::Y, 2}, Values{false, true});
  DataArray a(var);
  a.masks().set("x", maskX);
  a.masks().set("y", maskY);
  const auto meanX =
      makeVariable<double>(Dimensions{Dim::Y, 2}, units::m, Values{1.0, 3.0});
  const auto meanY =
      makeVariable<double>(Dimensions{Dim::X, 2}, units::m, Values{1.0, 2.0});
  EXPECT_EQ(op(a, Dim::X).data(), meanX);
  EXPECT_EQ(op(a, Dim::Y).data(), meanY);
  EXPECT_FALSE(op(a, Dim::X).masks().contains("x"));
  EXPECT_TRUE(op(a, Dim::X).masks().contains("y"));
  EXPECT_TRUE(op(a, Dim::Y).masks().contains("x"));
  EXPECT_FALSE(op(a, Dim::Y).masks().contains("y"));
}

template <typename Op> void test_masked_data_array_nd_mask(Op op) {
  const auto var = makeVariable<double>(Dimensions{{Dim::Y, 2}, {Dim::X, 2}},
                                        units::m, Values{1.0, 2.0, 3.0, 4.0});
  // Just a single masked element
  const auto mask = makeVariable<bool>(Dimensions{{Dim::Y, 2}, {Dim::X, 2}},
                                       Values{false, true, false, false});
  DataArray a(var);
  a.masks().set("mask", mask);
  const auto meanX =
      makeVariable<double>(Dimensions{Dim::Y, 2}, units::m,
                           Values{(1.0 + 0.0) / 1, (3.0 + 4.0) / 2});
  const auto meanY =
      makeVariable<double>(Dimensions{Dim::X, 2}, units::m,
                           Values{(1.0 + 3.0) / 2, (0.0 + 4.0) / 1});
  const auto mean = makeVariable<double>(units::m, Shape{1},
                                         Values{(1.0 + 0.0 + 3.0 + 4.0) / 3});
  EXPECT_EQ(op(a, Dim::X).data(), meanX);
  EXPECT_EQ(op(a, Dim::Y).data(), meanY);
  EXPECT_EQ(op(a).data(), mean);
  EXPECT_FALSE(op(a, Dim::X).masks().contains("mask"));
  EXPECT_FALSE(op(a, Dim::X).masks().contains("mask"));
  EXPECT_FALSE(op(a).masks().contains("mask"));
}

auto mean_func = overloaded{
    [](const auto &x, const auto &dim) { return mean(x, dim); },
    [](const auto &x, const auto &dim, auto &out) { return mean(x, dim, out); },
    [](const auto &x) { return mean(x); }};
auto nanmean_func =
    overloaded{[](const auto &x, const auto &dim) { return nanmean(x, dim); },
               [](const auto &x, const auto &dim, auto &out) {
                 return nanmean(x, dim, out);
               },
               [](const auto &x) { return nanmean(x); }};
} // namespace

TEST(MeanTest, masked_data_array) {
  test_masked_data_array_1_mask(mean_func);
  test_masked_data_array_1_mask(nanmean_func);
}

TEST(MeanTest, masked_data_array_two_masks) {
  test_masked_data_array_2_masks(mean_func);
  test_masked_data_array_2_masks(nanmean_func);
}

TEST(MeanTest, masked_data_array_md_masks) {
  test_masked_data_array_nd_mask(mean_func);
  test_masked_data_array_nd_mask(nanmean_func);
}

TEST(MeanTest, nanmean_masked_data_with_nans) {
  // Two Nans
  const auto var =
      makeVariable<double>(Dimensions{{Dim::Y, 2}, {Dim::X, 2}}, units::m,
                           Values{double(NAN), double(NAN), 3.0, 4.0});
  // Two masked element
  const auto mask = makeVariable<bool>(Dimensions{{Dim::Y, 2}, {Dim::X, 2}},
                                       Values{false, true, true, false});
  DataArray a(var);
  a.masks().set("mask", mask);
  // First element NaN, second NaN AND masked, third masked, forth non-masked
  // finite number
  const auto mean = makeVariable<double>(units::m, Shape{1},
                                         Values{(0.0 + 0.0 + 0.0 + 4.0) / 1});
  EXPECT_EQ(nanmean(a).data(), mean);
}

TYPED_TEST(MeanTest, mean_over_dim) {
  auto ds = make_1_values_and_variances<TypeParam>(
      "a", {Dim::X, 3}, units::dimensionless, {1, 2, 3}, {12, 15, 18});
  if constexpr (std::numeric_limits<TypeParam>::is_integer) {
    EXPECT_EQ(mean(ds, Dim::X)["a"].data(), makeVariable<double>(Values{2}));
    EXPECT_EQ(mean(ds.slice({Dim::X, 0, 2}), Dim::X)["a"].data(),
              makeVariable<double>(Values{1.5}));
  } else {
    EXPECT_EQ(mean(ds, Dim::X)["a"].data(),
              makeVariable<TypeParam>(Values{2}, Variances{5.0}));
    EXPECT_EQ(mean(ds.slice({Dim::X, 0, 2}), Dim::X)["a"].data(),
              makeVariable<TypeParam>(Values{1.5}, Variances{6.75}));
  }
}

TYPED_TEST(MeanTest, mean_all_dims) {
  DataArray da{makeVariable<TypeParam>(Dims{Dim::X, Dim::Y}, Values{1, 2, 3, 4},
                                       Shape{2, 2})};

  constexpr bool is_integer_t = std::numeric_limits<TypeParam>::is_integer;
  // For all FP input dtypes, dtype is same on output. Integers are converted to
  // double FP precision.
  using T = std::conditional_t<is_integer_t, double, TypeParam>;
  EXPECT_EQ(mean(da).data(), makeVariable<T>(Values{2.5}));
  EXPECT_EQ(mean(da).data(), makeVariable<T>(Values{2.5}));

  Dataset ds{{{"a", da}}};
  EXPECT_EQ(mean(ds)["a"], mean(da));
}

TEST(MeanTest, nanmean_over_dim) {
  auto ds = make_1_values_and_variances<double>(
      "a", {Dim::X, 3}, units::dimensionless, {1.0, 2.0, double(NAN)},
      {12.0, 15.0, 18.0});
  EXPECT_EQ(nanmean(ds, Dim::X)["a"].data(),
            makeVariable<double>(Values{1.5}, Variances{6.75}));
  EXPECT_EQ(nanmean(ds.slice({Dim::X, 0, 2}), Dim::X)["a"].data(),
            makeVariable<double>(Values{1.5}, Variances{6.75}));
}

TEST(MeanTest, nanmean_all_dims) {
  DataArray da{makeVariable<double>(
      Dims{Dim::X, Dim::Y}, Values{1.0, 2.0, 3.0, double(NAN)}, Shape{2, 2})};
  EXPECT_EQ(nanmean(da).data(), makeVariable<double>(Values{2.0}));

  Dataset ds{{{"a", da}}};
  EXPECT_EQ(nanmean(ds)["a"], nanmean(da));

  EXPECT_THROW(nanmean(astype(da, dtype<int64_t>)), except::TypeError);
}

TEST(MeanTest, nanmean_throws_on_int) {
  // Do not support integer type input variables
  auto d = make_1_values_and_variances<int>(
      "a", {Dim::X, 3}, units::dimensionless, {1, 2, 3}, {1, 2, 3});
  EXPECT_THROW(nanmean(d), except::TypeError);
  EXPECT_THROW(nanmean(d, Dim::X), except::TypeError);
  EXPECT_THROW(nanmean(d["a"]), except::TypeError);
  EXPECT_THROW(nanmean(d["a"], Dim::X), except::TypeError);
}
