// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <vector>

#include "fix_typed_test_suite_warnings.h"
#include "test_macros.h"

#include "scipp/core/element/math.h"
#include "scipp/variable/except.h"
#include "scipp/variable/misc_operations.h"
#include "scipp/variable/operations.h"
#include "scipp/variable/variable.h"

using namespace scipp;
using namespace scipp::core;
using namespace scipp::variable;

template <typename T>
class VariableScalarOperatorTest : public ::testing::Test {
public:
  Variable variable{makeVariable<T>(Dims{Dim::X}, Shape{1}, Values{10})};
  const T scalar{2};

  T value() const { return this->variable.template values<T>()[0]; }
};

using ScalarTypes = ::testing::Types<double, float, int64_t, int32_t>;
TYPED_TEST_SUITE(VariableScalarOperatorTest, ScalarTypes);

TYPED_TEST(VariableScalarOperatorTest, plus_equals) {
  this->variable += this->scalar * units::one;
  EXPECT_EQ(this->value(), 12);
}

TYPED_TEST(VariableScalarOperatorTest, minus_equals) {
  this->variable -= this->scalar * units::one;
  EXPECT_EQ(this->value(), 8);
}

TYPED_TEST(VariableScalarOperatorTest, times_equals) {
  this->variable *= this->scalar * units::one;
  EXPECT_EQ(this->value(), 20);
}

TYPED_TEST(VariableScalarOperatorTest, divide_equals) {
  if (this->variable.dtype() == dtype<double> ||
      this->variable.dtype() == dtype<float>) {
    this->variable /= this->scalar * units::one;
    EXPECT_EQ(this->value(), 5);
  } else {
    EXPECT_THROW(this->variable /= this->scalar * units::one,
                 except::TypeError);
  }
}

TEST(Variable, operator_unary_minus) {
  const auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{1.1, 2.2});
  const auto expected = makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m,
                                             Values{-1.1, -2.2});
  auto b = -a;
  EXPECT_EQ(b, expected);
}

TEST(VariableView, unary_minus) {
  const auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{1.1, 2.2});
  const auto expected =
      makeVariable<double>(Dims(), Shape(), units::m, Values{-2.2});
  auto b = -a.slice({Dim::X, 1});
  EXPECT_EQ(b, expected);
}

TEST(Variable, operator_plus_equal) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  ASSERT_NO_THROW(a += a);
  EXPECT_EQ(a.values<double>()[0], 2.2);
  EXPECT_EQ(a.values<double>()[1], 4.4);
}

TEST(Variable, operator_plus_equal_automatic_broadcast_of_rhs) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  auto fewer_dimensions = makeVariable<double>(Values{1.0});

  ASSERT_NO_THROW(a += fewer_dimensions);
  EXPECT_EQ(a.values<double>()[0], 2.1);
  EXPECT_EQ(a.values<double>()[1], 3.2);
}

TEST(Variable, operator_plus_equal_transpose) {
  auto a = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 2}, units::m,
                                Values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  auto transpose =
      makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 3}, units::m,
                           Values{1.0, 3.0, 5.0, 2.0, 4.0, 6.0});

  EXPECT_NO_THROW(a += transpose);
  ASSERT_EQ(a, makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 2}, units::m,
                                    Values{2.0, 4.0, 6.0, 8.0, 10.0, 12.0}));
}

TEST(Variable, operator_plus_equal_different_dimensions) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  auto different_dimensions =
      makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{1.1, 2.2});
  EXPECT_THROW_MSG(a += different_dimensions, std::runtime_error,
                   "Expected {{x, 2}} to contain {{y, 2}}.");
}

TEST(Variable, operator_plus_equal_different_unit) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  auto different_unit = copy(a);
  different_unit.setUnit(units::m);
  EXPECT_THROW(a += different_unit, except::UnitError);
}

TEST(Variable, operator_plus_equal_non_arithmetic_type) {
  auto a = makeVariable<std::string>(Dims{Dim::X}, Shape{1},
                                     Values{std::string("test")});
  EXPECT_THROW(a += a, except::TypeError);
}

TEST(Variable, operator_plus_equal_time_type) {
  using time_point = scipp::core::time_point;
  auto a = makeVariable<time_point>(Shape{1}, units::Unit{units::ns},
                                    Values{time_point{2}});
  EXPECT_THROW(a += static_cast<float>(1.0) * units::ns, except::TypeError);
  EXPECT_NO_THROW(a += static_cast<int64_t>(1) * units::ns);
  EXPECT_NO_THROW(a += static_cast<int32_t>(2) * units::ns);
  EXPECT_EQ(a, makeVariable<time_point>(Shape{1}, units::Unit{units::ns},
                                        Values{time_point{5}}));
}

TEST(Variable, operator_minus_equal_time_type) {
  using time_point = scipp::core::time_point;
  auto a = makeVariable<time_point>(Shape{1}, units::Unit{units::ns},
                                    Values{time_point{10}});
  EXPECT_THROW(a -= static_cast<float>(1.0) * units::ns, except::TypeError);
  EXPECT_NO_THROW(a -= static_cast<int64_t>(1) * units::ns);
  EXPECT_NO_THROW(a -= static_cast<int32_t>(2) * units::ns);
  EXPECT_EQ(a, makeVariable<time_point>(Shape{1}, units::Unit{units::ns},
                                        Values{time_point{7}}));
}

TEST(Variable, operator_plus_equal_different_variables_different_element_type) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.0});
  auto b = makeVariable<int64_t>(Dims{Dim::X}, Shape{1}, Values{2});
  EXPECT_NO_THROW(a += b);
}

TEST(Variable, operator_plus_equal_different_variables_same_element_type) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.0});
  auto b = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{2.0});
  EXPECT_NO_THROW(a += b);
  EXPECT_EQ(a.values<double>()[0], 3.0);
}

TEST(Variable, operator_plus_equal_scalar) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.1, 2.2});

  EXPECT_NO_THROW(a += 1.0 * units::one);
  EXPECT_EQ(a.values<double>()[0], 2.1);
  EXPECT_EQ(a.values<double>()[1], 3.2);
}

TEST(Variable, operator_plus_equal_custom_type) {
  auto a = makeVariable<float>(Dims{Dim::X}, Shape{2}, Values{1.1f, 2.2f});

  EXPECT_NO_THROW(a += a);
  EXPECT_EQ(a.values<float>()[0], 2.2f);
  EXPECT_EQ(a.values<float>()[1], 4.4f);
}

TEST(Variable, operator_plus_unit_fail) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 2.0},
                                Variances{3.0, 4.0});
  a.setUnit(units::m);
  auto b = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 2.0},
                                Variances{3.0, 4.0});
  b.setUnit(units::s);
  ASSERT_ANY_THROW_DISCARD(a + b);
  b.setUnit(units::m);
  ASSERT_NO_THROW_DISCARD(a + b);
}

TEST(Variable, operator_plus_eigen_type) {
  const auto var = makeVariable<Eigen::Vector3d>(
      Dims{Dim::X}, Shape{2},
      Values{Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Vector3d{0.1, 0.2, 0.3}});
  const auto expected = makeVariable<Eigen::Vector3d>(
      Dims(), Shape(), Values{Eigen::Vector3d{1.1, 2.2, 3.3}});

  const auto result = var.slice({Dim::X, 0}) + var.slice({Dim::X, 1});

  EXPECT_EQ(result, expected);
}

TEST(Variable, operator_times_equal) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{2.0, 3.0});

  EXPECT_EQ(a.unit(), units::m);
  EXPECT_NO_THROW(a *= a);
  EXPECT_EQ(a.values<double>()[0], 4.0);
  EXPECT_EQ(a.values<double>()[1], 9.0);
  EXPECT_EQ(a.unit(), units::m * units::m);
}

TEST(Variable, operator_times_equal_scalar) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{2.0, 3.0});

  EXPECT_EQ(a.unit(), units::m);
  EXPECT_NO_THROW(a *= 2.0 * units::one);
  EXPECT_EQ(a.values<double>()[0], 4.0);
  EXPECT_EQ(a.values<double>()[1], 6.0);
  EXPECT_EQ(a.unit(), units::m);
}

TEST(Variable, operator_plus_equal_unit_fail_integrity) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2},
                           units::Unit(units::m * units::m), Values{2.0, 3.0});
  const auto expected(a);

  ASSERT_THROW(a += a * a, std::runtime_error);
  EXPECT_EQ(a, expected);
}

TEST(Variable, operator_times_can_broadcast) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{0.5, 1.5});
  auto b = makeVariable<double>(Dims{Dim::Y}, Shape{2}, Values{2.0, 3.0});

  auto ab = a * b;
  auto reference = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 2},
                                        Values{1.0, 1.5, 3.0, 4.5});
  EXPECT_EQ(ab, reference);
}

TEST(Variable, operator_divide_equal) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{2.0, 3.0});
  auto b = makeVariable<double>(Values{2.0});
  b.setUnit(units::m);

  EXPECT_NO_THROW(a /= b);
  EXPECT_EQ(a.values<double>()[0], 1.0);
  EXPECT_EQ(a.values<double>()[1], 1.5);
  EXPECT_EQ(a.unit(), units::one / units::m);
}

TEST(Variable, operator_divide_equal_self) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{2.0, 3.0});

  EXPECT_EQ(a.unit(), units::m);
  EXPECT_NO_THROW(a /= a);
  EXPECT_EQ(a.values<double>()[0], 1.0);
  EXPECT_EQ(a.values<double>()[1], 1.0);
  EXPECT_EQ(a.unit(), units::one);
}

TEST(Variable, operator_divide_equal_scalar) {
  auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{2.0, 4.0});

  EXPECT_EQ(a.unit(), units::m);
  EXPECT_NO_THROW(a /= 2.0 * units::one);
  EXPECT_EQ(a.values<double>()[0], 1.0);
  EXPECT_EQ(a.values<double>()[1], 2.0);
  EXPECT_EQ(a.unit(), units::m);
}

TEST(Variable, operator_divide_scalar_double) {
  const auto a =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{2.0, 4.0});
  const auto result = 1.111 * units::one / a;
  EXPECT_EQ(result.values<double>()[0], 1.111 / 2.0);
  EXPECT_EQ(result.values<double>()[1], 1.111 / 4.0);
  EXPECT_EQ(result.unit(), units::one / units::m);
}

TEST(Variable, operator_divide_scalar_float) {
  const auto a =
      makeVariable<float>(Dims{Dim::X}, Shape{2}, units::m, Values{2.0, 4.0});
  const auto result = 1.111f * units::one / a;
  EXPECT_EQ(result.values<float>()[0], 1.111f / 2.0f);
  EXPECT_EQ(result.values<float>()[1], 1.111f / 4.0f);
  EXPECT_EQ(result.unit(), units::one / units::m);
}

TEST(Variable, operator_allowed_types) {
  auto i32 = makeVariable<int32_t>(Values{10});
  auto i64 = makeVariable<int64_t>(Values{10});
  auto f = makeVariable<float>(Values{0.5f});
  auto d = makeVariable<double>(Values{0.5});

  /* Can operate on higher precision from lower precision */
  EXPECT_NO_THROW(i64 += i32);
  EXPECT_NO_THROW(d += f);

  /* Can operate on lower precision from higher precision */
  EXPECT_NO_THROW(i32 += i64);
  EXPECT_NO_THROW(f += d);

  /* Expect promotion to double if one parameter is double */
  EXPECT_EQ(dtype<double>, (f + d).dtype());
  EXPECT_EQ(dtype<double>, (d + f).dtype());
}

TEST(Variable, concatenate) {
  Dimensions dims(Dim::X, 1);
  auto a = makeVariable<double>(Dimensions(dims), Values{1.0});
  auto b = makeVariable<double>(Dimensions(dims), Values{2.0});
  a.setUnit(units::m);
  b.setUnit(units::m);
  auto ab = concatenate(a, b, Dim::X);
  ASSERT_EQ(ab.dims().volume(), 2);
  EXPECT_EQ(ab.unit(), units::m);
  const auto &data = ab.values<double>();
  EXPECT_EQ(data[0], 1.0);
  EXPECT_EQ(data[1], 2.0);
  auto ba = concatenate(b, a, Dim::X);
  const auto abba = concatenate(ab, ba, Dim::Y);
  ASSERT_EQ(abba.dims().volume(), 4);
  EXPECT_EQ(abba.dims().shape().size(), 2);
  const auto &data2 = abba.values<double>();
  EXPECT_EQ(data2[0], 1.0);
  EXPECT_EQ(data2[1], 2.0);
  EXPECT_EQ(data2[2], 2.0);
  EXPECT_EQ(data2[3], 1.0);
  const auto ababbaba = concatenate(abba, abba, Dim::X);
  ASSERT_EQ(ababbaba.dims().volume(), 8);
  const auto &data3 = ababbaba.values<double>();
  EXPECT_EQ(data3[0], 1.0);
  EXPECT_EQ(data3[1], 2.0);
  EXPECT_EQ(data3[2], 1.0);
  EXPECT_EQ(data3[3], 2.0);
  EXPECT_EQ(data3[4], 2.0);
  EXPECT_EQ(data3[5], 1.0);
  EXPECT_EQ(data3[6], 2.0);
  EXPECT_EQ(data3[7], 1.0);
  const auto abbaabba = concatenate(abba, abba, Dim::Y);
  ASSERT_EQ(abbaabba.dims().volume(), 8);
  const auto &data4 = abbaabba.values<double>();
  EXPECT_EQ(data4[0], 1.0);
  EXPECT_EQ(data4[1], 2.0);
  EXPECT_EQ(data4[2], 2.0);
  EXPECT_EQ(data4[3], 1.0);
  EXPECT_EQ(data4[4], 1.0);
  EXPECT_EQ(data4[5], 2.0);
  EXPECT_EQ(data4[6], 2.0);
  EXPECT_EQ(data4[7], 1.0);
}

TEST(Variable, concatenate_volume_with_slice) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.0});
  auto aa = concatenate(a, a, Dim::X);
  EXPECT_NO_THROW_DISCARD(concatenate(aa, a, Dim::X));
}

TEST(Variable, concatenate_slice_with_volume) {
  auto a = makeVariable<double>(Dims{Dim::X}, Shape{1}, Values{1.0});
  auto aa = concatenate(a, a, Dim::X);
  EXPECT_NO_THROW_DISCARD(concatenate(a, aa, Dim::X));
}

TEST(Variable, concatenate_fail) {
  Dimensions dims(Dim::X, 1);
  auto a = makeVariable<double>(Dimensions(dims), Values{1.0});
  auto b = makeVariable<double>(Dimensions(dims), Values{2.0});
  auto c = makeVariable<float>(Dimensions(dims), Values{2.0});
  EXPECT_THROW_MSG_DISCARD(
      concatenate(a, c, Dim::X), std::runtime_error,
      "Cannot concatenate Variables: Data types do not match.");
  auto aa = concatenate(a, a, Dim::X);
  EXPECT_THROW_MSG_DISCARD(
      concatenate(a, aa, Dim::Y), std::runtime_error,
      "Cannot concatenate Variables: Dimension extents do not match.");
}

TEST(Variable, concatenate_unit_fail) {
  Dimensions dims(Dim::X, 1);
  auto a = makeVariable<double>(Dimensions(dims), Values{1.0});
  auto b = copy(a);
  EXPECT_NO_THROW_DISCARD(concatenate(a, b, Dim::X));
  a.setUnit(units::m);
  EXPECT_THROW_MSG_DISCARD(concatenate(a, b, Dim::X), std::runtime_error,
                           "Cannot concatenate Variables: Units do not match.");
  b.setUnit(units::m);
  EXPECT_NO_THROW_DISCARD(concatenate(a, b, Dim::X));
}

TEST(Variable, concatenate_from_slices_with_broadcast) {
  auto input_v = {0.0, 0.1, 0.2, 0.3};
  auto var = makeVariable<double>(Dimensions{Dim::X, 4}, Values(input_v),
                                  Variances(input_v));
  auto out = concatenate(var.slice(Slice(Dim::X, 1, 4)),
                         var.slice(Slice(Dim::X, 0, 3)), Dim::Y);
  auto expected = {0.1, 0.2, 0.3, 0.0, 0.1, 0.2};
  EXPECT_EQ(out, makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                      Values(expected), Variances(expected)));
}

TEST(VariableView, minus_equals_failures) {
  auto var = makeVariable<double>(Dims{Dim::X, Dim::Y}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});

  EXPECT_THROW_MSG(var -= var.slice({Dim::X, 0, 1}), std::runtime_error,
                   "Expected {{x, 2}, {y, 2}} to contain {{x, 1}, {y, 2}}.");
}

TEST(VariableView, self_overlapping_view_operation) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});

  var -= var.slice({Dim::Y, 0});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  // This is the critical part: After subtracting for y=0 the view points to
  // data containing 0.0, so subsequently the subtraction would have no effect
  // if self-overlap was not taken into account by the implementation.
  EXPECT_EQ(data[2], 2.0);
  EXPECT_EQ(data[3], 2.0);
}

TEST(VariableView, minus_equals_slice_const_outer) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});
  const auto copy = variable::copy(var);

  var -= copy.slice({Dim::Y, 0});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  EXPECT_EQ(data[2], 2.0);
  EXPECT_EQ(data[3], 2.0);
  var -= copy.slice({Dim::Y, 1});
  EXPECT_EQ(data[0], -3.0);
  EXPECT_EQ(data[1], -4.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], -2.0);
}

TEST(VariableView, minus_equals_slice_outer) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});
  auto copy = variable::copy(var);

  var -= copy.slice({Dim::Y, 0});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  EXPECT_EQ(data[2], 2.0);
  EXPECT_EQ(data[3], 2.0);
  var -= copy.slice({Dim::Y, 1});
  EXPECT_EQ(data[0], -3.0);
  EXPECT_EQ(data[1], -4.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], -2.0);
}

TEST(VariableView, minus_equals_slice_inner) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});
  auto copy = variable::copy(var);

  var -= copy.slice({Dim::X, 0});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 1.0);
  EXPECT_EQ(data[2], 0.0);
  EXPECT_EQ(data[3], 1.0);
  var -= copy.slice({Dim::X, 1});
  EXPECT_EQ(data[0], -2.0);
  EXPECT_EQ(data[1], -1.0);
  EXPECT_EQ(data[2], -4.0);
  EXPECT_EQ(data[3], -3.0);
}

TEST(VariableView, minus_equals_slice_of_slice) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});
  auto copy = variable::copy(var);

  var -= copy.slice({Dim::X, 1}).slice({Dim::Y, 1});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], -3.0);
  EXPECT_EQ(data[1], -2.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], 0.0);
}

TEST(VariableView, minus_equals_nontrivial_slices) {
  auto source = makeVariable<double>(
      Dims{Dim::Y, Dim::X}, Shape{3, 3},
      Values{11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0});
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
    target -= source.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -11.0);
    EXPECT_EQ(data[1], -12.0);
    EXPECT_EQ(data[2], -21.0);
    EXPECT_EQ(data[3], -22.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
    target -= source.slice({Dim::X, 1, 3}).slice({Dim::Y, 0, 2});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -12.0);
    EXPECT_EQ(data[1], -13.0);
    EXPECT_EQ(data[2], -22.0);
    EXPECT_EQ(data[3], -23.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
    target -= source.slice({Dim::X, 0, 2}).slice({Dim::Y, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -21.0);
    EXPECT_EQ(data[1], -22.0);
    EXPECT_EQ(data[2], -31.0);
    EXPECT_EQ(data[3], -32.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
    target -= source.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -22.0);
    EXPECT_EQ(data[1], -23.0);
    EXPECT_EQ(data[2], -32.0);
    EXPECT_EQ(data[3], -33.0);
  }
}

TEST(VariableView, slice_inner_minus_equals) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});

  var.slice({Dim::X, 0}) -= var.slice({Dim::X, 1});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], -1.0);
  EXPECT_EQ(data[1], 2.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], 4.0);
}

TEST(VariableView, slice_outer_minus_equals) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                  Values{1.0, 2.0, 3.0, 4.0});

  var.slice({Dim::Y, 0}) -= var.slice({Dim::Y, 1});
  const auto data = var.values<double>();
  EXPECT_EQ(data[0], -2.0);
  EXPECT_EQ(data[1], -2.0);
  EXPECT_EQ(data[2], 3.0);
  EXPECT_EQ(data[3], 4.0);
}

TEST(VariableView, nontrivial_slice_minus_equals) {
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                       Values{11.0, 12.0, 21.0, 22.0});
    target.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2}) -= source;
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -11.0);
    EXPECT_EQ(data[1], -12.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], -21.0);
    EXPECT_EQ(data[4], -22.0);
    EXPECT_EQ(data[5], 0.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], 0.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                       Values{11.0, 12.0, 21.0, 22.0});
    target.slice({Dim::X, 1, 3}).slice({Dim::Y, 0, 2}) -= source;
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], -11.0);
    EXPECT_EQ(data[2], -12.0);
    EXPECT_EQ(data[3], 0.0);
    EXPECT_EQ(data[4], -21.0);
    EXPECT_EQ(data[5], -22.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], 0.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                       Values{11.0, 12.0, 21.0, 22.0});
    target.slice({Dim::X, 0, 2}).slice({Dim::Y, 1, 3}) -= source;
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], 0.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], -11.0);
    EXPECT_EQ(data[4], -12.0);
    EXPECT_EQ(data[5], 0.0);
    EXPECT_EQ(data[6], -21.0);
    EXPECT_EQ(data[7], -22.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                       Values{11.0, 12.0, 21.0, 22.0});
    target.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3}) -= source;
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], 0.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], 0.0);
    EXPECT_EQ(data[4], -11.0);
    EXPECT_EQ(data[5], -12.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], -21.0);
    EXPECT_EQ(data[8], -22.0);
  }
}

TEST(VariableView, nontrivial_slice_minus_equals_slice) {
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source =
        makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                             Values{666.0, 11.0, 12.0, 666.0, 21.0, 22.0});
    target.slice({Dim::X, 0, 2}).slice({Dim::Y, 0, 2}) -=
        source.slice({Dim::X, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], -11.0);
    EXPECT_EQ(data[1], -12.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], -21.0);
    EXPECT_EQ(data[4], -22.0);
    EXPECT_EQ(data[5], 0.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], 0.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source =
        makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                             Values{666.0, 11.0, 12.0, 666.0, 21.0, 22.0});
    target.slice({Dim::X, 1, 3}).slice({Dim::Y, 0, 2}) -=
        source.slice({Dim::X, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], -11.0);
    EXPECT_EQ(data[2], -12.0);
    EXPECT_EQ(data[3], 0.0);
    EXPECT_EQ(data[4], -21.0);
    EXPECT_EQ(data[5], -22.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], 0.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source =
        makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                             Values{666.0, 11.0, 12.0, 666.0, 21.0, 22.0});
    target.slice({Dim::X, 0, 2}).slice({Dim::Y, 1, 3}) -=
        source.slice({Dim::X, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], 0.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], -11.0);
    EXPECT_EQ(data[4], -12.0);
    EXPECT_EQ(data[5], 0.0);
    EXPECT_EQ(data[6], -21.0);
    EXPECT_EQ(data[7], -22.0);
    EXPECT_EQ(data[8], 0.0);
  }
  {
    auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{3, 3});
    auto source =
        makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                             Values{666.0, 11.0, 12.0, 666.0, 21.0, 22.0});
    target.slice({Dim::X, 1, 3}).slice({Dim::Y, 1, 3}) -=
        source.slice({Dim::X, 1, 3});
    const auto data = target.values<double>();
    EXPECT_EQ(data[0], 0.0);
    EXPECT_EQ(data[1], 0.0);
    EXPECT_EQ(data[2], 0.0);
    EXPECT_EQ(data[3], 0.0);
    EXPECT_EQ(data[4], -11.0);
    EXPECT_EQ(data[5], -12.0);
    EXPECT_EQ(data[6], 0.0);
    EXPECT_EQ(data[7], -21.0);
    EXPECT_EQ(data[8], -22.0);
  }
}

TEST(VariableView, slice_minus_lower_dimensional) {
  auto target = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2});
  auto source = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1.0, 2.0});
  EXPECT_EQ(target.slice({Dim::Y, 1, 2}).dims(),
            (Dimensions{{Dim::Y, 1}, {Dim::X, 2}}));

  target.slice({Dim::Y, 1, 2}) -= source;

  const auto data = target.values<double>();
  EXPECT_EQ(data[0], 0.0);
  EXPECT_EQ(data[1], 0.0);
  EXPECT_EQ(data[2], -1.0);
  EXPECT_EQ(data[3], -2.0);
}

TEST(VariableView, slice_binary_operations) {
  auto v = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 2},
                                Values{1, 2, 3, 4});
  // Note: There does not seem to be a way to test whether this is using the
  // operators that convert the second argument to Variable (it should not), or
  // keep it as a view. See variable_benchmark.cpp for an attempt to verify
  // this.
  auto sum = v.slice({Dim::X, 0}) + v.slice({Dim::X, 1});
  auto difference = v.slice({Dim::X, 0}) - v.slice({Dim::X, 1});
  auto product = v.slice({Dim::X, 0}) * v.slice({Dim::X, 1});
  auto ratio = v.slice({Dim::X, 0}) / v.slice({Dim::X, 1});
  EXPECT_TRUE(equals(sum.values<double>(), {3, 7}));
  EXPECT_TRUE(equals(difference.values<double>(), {-1, -1}));
  EXPECT_TRUE(equals(product.values<double>(), {2, 12}));
  EXPECT_TRUE(equals(ratio.values<double>(), {1.0 / 2.0, 3.0 / 4.0}));
}

TEST(Variable, reverse) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                  Values{1, 2, 3, 4, 5, 6});
  auto reverseX = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                       Values{3, 2, 1, 6, 5, 4});
  auto reverseY = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                       Values{4, 5, 6, 1, 2, 3});

  EXPECT_EQ(reverse(var, Dim::X), reverseX);
  EXPECT_EQ(reverse(var, Dim::Y), reverseY);
}

TEST(Variable, non_in_place_scalar_operations) {
  auto var = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{1, 2});

  auto sum = var + 1.0 * units::one;
  EXPECT_TRUE(equals(sum.values<double>(), {2, 3}));
  sum = 2.0 * units::one + var;
  EXPECT_TRUE(equals(sum.values<double>(), {3, 4}));

  auto diff = var - 1.0 * units::one;
  EXPECT_TRUE(equals(diff.values<double>(), {0, 1}));
  diff = 2.0 * units::one - var;
  EXPECT_TRUE(equals(diff.values<double>(), {1, 0}));

  auto prod = var * (2.0 * units::one);
  EXPECT_TRUE(equals(prod.values<double>(), {2, 4}));
  prod = 3.0 * units::one * var;
  EXPECT_TRUE(equals(prod.values<double>(), {3, 6}));

  auto ratio = var / (2.0 * units::one);
  EXPECT_TRUE(equals(ratio.values<double>(), {1.0 / 2.0, 1.0}));
  ratio = 3.0 * units::one / var;
  EXPECT_TRUE(equals(ratio.values<double>(), {3.0, 1.5}));
}

TEST(VariableView, scalar_operations) {
  auto var = makeVariable<double>(Dims{Dim::Y, Dim::X}, Shape{2, 3},
                                  Values{11, 12, 13, 21, 22, 23});

  var.slice({Dim::X, 0}) += 1.0 * units::one;
  EXPECT_TRUE(equals(var.values<double>(), {12, 12, 13, 22, 22, 23}));
  var.slice({Dim::Y, 1}) += 1.0 * units::one;
  EXPECT_TRUE(equals(var.values<double>(), {12, 12, 13, 23, 23, 24}));
  var.slice({Dim::X, 1, 3}) += 1.0 * units::one;
  EXPECT_TRUE(equals(var.values<double>(), {12, 13, 14, 23, 24, 25}));
  var.slice({Dim::X, 1}) -= 1.0 * units::one;
  EXPECT_TRUE(equals(var.values<double>(), {12, 12, 14, 23, 23, 25}));
  var.slice({Dim::X, 2}) *= 0.0 * units::one;
  EXPECT_TRUE(equals(var.values<double>(), {12, 12, 0, 23, 23, 0}));
  var.slice({Dim::Y, 0}) /= 2.0 * units::one;
  EXPECT_TRUE(equals(var.values<double>(), {6, 6, 0, 23, 23, 0}));
}

TEST(VariableTest, binary_op_with_variance) {
  const auto var = makeVariable<double>(
      Dims{Dim::X, Dim::Y}, Shape{2, 3}, Values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
      Variances{0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
  const auto sum = makeVariable<double>(
      Dims{Dim::X, Dim::Y}, Shape{2, 3}, Values{2.0, 4.0, 6.0, 8.0, 10.0, 12.0},
      Variances{0.2, 0.4, 0.6, 0.8, 1.0, 1.2});
  auto tmp = var + var;
  EXPECT_TRUE(tmp.hasVariances());
  EXPECT_EQ(tmp.variances<double>()[0], 0.2);
  EXPECT_EQ(var + var, sum);

  tmp = var * sum;
  EXPECT_EQ(tmp.variances<double>()[0], 0.1 * 2.0 * 2.0 + 0.2 * 1.0 * 1.0);
}

TEST(VariableTest, divide_with_variance) {
  // Note the 0.0: With a wrong implementation the resulting variance is INF.
  const auto a = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{2.0, 0.0},
                                      Variances{0.1, 0.1});
  const auto b = makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{3.0, 3.0},
                                      Variances{0.2, 0.2});
  const auto expected =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, Values{2.0 / 3.0, 0.0},
                           Variances{(0.1 / (2.0 * 2.0) + 0.2 / (3.0 * 3.0)) *
                                         (2.0 / 3.0) * (2.0 / 3.0),
                                     /* (0.1 / (0.0 * 0.0) + 0.2 / (3.0
                                      * 3.0)) * (0.0 / 3.0) * (0.0 / 3.0)
                                      naively, but if we take the limit... */
                                     0.1 / (3.0 * 3.0)});
  const auto q = a / b;
  EXPECT_DOUBLE_EQ(q.values<double>()[0], expected.values<double>()[0]);
  EXPECT_DOUBLE_EQ(q.values<double>()[1], expected.values<double>()[1]);
  EXPECT_DOUBLE_EQ(q.variances<double>()[0], expected.variances<double>()[0]);
  EXPECT_DOUBLE_EQ(q.variances<double>()[1], expected.variances<double>()[1]);
}

TEST(VariableTest, boolean_or) {
  const auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, true, true, true});

  const auto result = a | b;

  EXPECT_EQ(result, expected);
}

TEST(VariableTest, boolean_or_equals) {
  auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                              Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  a |= b;
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, true, true, true});

  EXPECT_EQ(a, expected);
}

TEST(VariableTest, boolean_and_equals) {
  auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                              Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  a &= b;
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, false, false, true});

  EXPECT_EQ(a, expected);
}

TEST(VariableTest, boolean_and) {
  const auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, false, false, true});

  const auto result = a & b;

  EXPECT_EQ(result, expected);
}

TEST(VariableTest, boolean_xor_equals) {
  auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                              Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  a ^= b;
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, true, true, false});

  EXPECT_EQ(a, expected);
}

TEST(VariableTest, boolean_xor) {
  const auto a = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, true, false, true});
  const auto b = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                    Values{false, false, true, true});
  const auto expected = makeVariable<bool>(Dims{Dim::X}, Shape{4},
                                           Values{false, true, true, false});
  const auto result = a ^ b;

  EXPECT_EQ(result, expected);
}

TEST(VariableTest, zip_positions) {
  const Variable x =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, units::m, Values{1, 2, 3});
  auto positions = geometry::position(x, x, x);
  auto values = positions.values<Eigen::Vector3d>();
  EXPECT_EQ(values.size(), 3);
  EXPECT_EQ(values[0], (Eigen::Vector3d{1, 1, 1}));
  EXPECT_EQ(values[1], (Eigen::Vector3d{2, 2, 2}));
  EXPECT_EQ(values[2], (Eigen::Vector3d{3, 3, 3}));
}
TEST(VariableTest, unzip_x) {
  const Variable pos = makeVariable<Eigen::Vector3d>(
      Dims{Dim::X}, Shape{2}, units::m,
      Values{Eigen::Vector3d{1, 2, 3}, Eigen::Vector3d{4, 5, 6}});
  auto x_ = geometry::x(pos);
  const auto expected_x =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{1.0, 4.0});
  EXPECT_EQ(x_, expected_x);
}

TEST(VariableTest, unzip_y) {
  const Variable pos = makeVariable<Eigen::Vector3d>(
      Dims{Dim::X}, Shape{2}, units::m,
      Values{Eigen::Vector3d{1, 2, 3}, Eigen::Vector3d{4, 5, 6}});
  auto y_ = geometry::y(pos);
  const auto expected_y =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{2.0, 5.0});
  EXPECT_EQ(y_, expected_y);
}

TEST(VariableTest, unzip_z) {
  const Variable pos = makeVariable<Eigen::Vector3d>(
      Dims{Dim::X}, Shape{2}, units::m,
      Values{Eigen::Vector3d{1, 2, 3}, Eigen::Vector3d{4, 5, 6}});
  auto z_ = geometry::z(pos);
  const auto expected_z =
      makeVariable<double>(Dims{Dim::X}, Shape{2}, units::m, Values{3.0, 6.0});
  EXPECT_EQ(z_, expected_z);
}
TEST(VariableTest, zip_unzip_positions) {
  const Variable x_in =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, units::m, Values{1, 2, 3});
  const Variable y_in =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, units::m, Values{4, 5, 6});
  const Variable z_in =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, units::m, Values{7, 8, 9});
  auto positions = geometry::position(x_in, y_in, z_in);
  auto x_out = geometry::x(positions);
  auto y_out = geometry::y(positions);
  auto z_out = geometry::z(positions);
  EXPECT_EQ(x_in, x_out);
  EXPECT_EQ(y_in, y_out);
  EXPECT_EQ(z_in, z_out);
}

TEST(VariableTest, rotate) {
  Eigen::Vector3d vec1(1, 2, 3);
  Eigen::Vector3d vec2(4, 5, 6);
  auto vec = makeVariable<Eigen::Vector3d>(Dims{Dim::X}, Shape{2}, units::m,
                                           Values{vec1, vec2});
  Eigen::Quaterniond rot1(1.1, 2.2, 3.3, 4.4);
  Eigen::Quaterniond rot2(5.5, 6.6, 7.7, 8.8);
  rot1.normalize();
  rot2.normalize();
  const Variable rot = makeVariable<Eigen::Matrix3d>(
      Dims{Dim::X}, Shape{2}, units::one,
      Values{rot1.toRotationMatrix(), rot2.toRotationMatrix()});
  auto vec_new = rot * vec;
  const auto rotated = makeVariable<Eigen::Vector3d>(
      Dims{Dim::X}, Shape{2}, units::m,
      Values{rot1.toRotationMatrix() * vec1, rot2.toRotationMatrix() * vec2});
  EXPECT_EQ(vec_new, rotated);
}

TEST(VariableTest, mul_vector) {
  Eigen::Vector3d vec1(1, 2, 3);
  Eigen::Vector3d vec2(2, 4, 6);
  auto vec = makeVariable<Eigen::Vector3d>(Dims{Dim::X}, Shape{1}, units::m,
                                           Values{vec1});
  auto expected_vec = makeVariable<Eigen::Vector3d>(Dims{Dim::X}, Shape{1},
                                                    units::m, Values{vec2});
  auto scale = makeVariable<double>(Dims{}, Shape{1}, units::one, Values{2.0});
  auto scale_with_variance = makeVariable<double>(Dims{}, Shape{1}, units::one,
                                                  Values{2.0}, Variances{1.0});

  auto left_scaled_vec = scale * vec;
  auto right_scaled_vec = vec * scale;

  EXPECT_THROW_DISCARD(vec * scale_with_variance, except::VariancesError);
  EXPECT_EQ(left_scaled_vec, expected_vec);
  EXPECT_EQ(right_scaled_vec, expected_vec);
}

TEST(VariableTest, divide_vector) {
  Eigen::Vector3d vec1(1, 2, 3);
  Eigen::Vector3d vec2(2, 4, 6);
  auto vec = makeVariable<Eigen::Vector3d>(Dims{Dim::X}, Shape{1}, units::m,
                                           Values{vec2});
  auto expected_vec = makeVariable<Eigen::Vector3d>(Dims{Dim::X}, Shape{1},
                                                    units::m, Values{vec1});
  auto scale = makeVariable<double>(Dims{}, Shape{1}, units::one, Values{2.0});

  auto scaled_vec = vec / scale;

  EXPECT_EQ(scaled_vec, expected_vec);
}

TEST(VariableTest, masked_to_zero) {
  auto var =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, units::m, Values{1, 1, 1});
  auto mask = makeVariable<bool>(Dims{Dim::X}, Shape{3}, units::one,
                                 Values{true, false, true});
  auto expected_var =
      makeVariable<double>(Dims{Dim::X}, Shape{3}, units::m, Values{0, 1, 0});

  auto masked_var = masked_to_zero(var, mask);

  EXPECT_EQ(masked_var, expected_var);
}
