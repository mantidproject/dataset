// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/core/element/arithmetic.h"
#include "scipp/core/transform_common.h"
#include "scipp/units/unit.h"

using namespace scipp;
using namespace scipp::core::element;

class ElementArithmeticTest : public ::testing::Test {
protected:
  const double a = 1.2;
  const double b = 2.3;
  double val = a;
};

TEST_F(ElementArithmeticTest, plus_equals) {
  plus_equals(val, b);
  EXPECT_EQ(val, a + b);
}

TEST_F(ElementArithmeticTest, minus_equals) {
  minus_equals(val, b);
  EXPECT_EQ(val, a - b);
}

TEST_F(ElementArithmeticTest, times_equals) {
  times_equals(val, b);
  EXPECT_EQ(val, a * b);
}

TEST_F(ElementArithmeticTest, divide_equals) {
  divide_equals(val, b);
  EXPECT_EQ(val, a / b);
}

TEST_F(ElementArithmeticTest, non_in_place) {
  EXPECT_EQ(plus(a, b), a + b);
  EXPECT_EQ(minus(a, b), a - b);
  EXPECT_EQ(times(a, b), a * b);
  EXPECT_EQ(divide(a, b), a / b);
}

TEST_F(ElementArithmeticTest, unary_minus) { EXPECT_EQ(unary_minus(a), -a); }

TEST(ElementArithmeticIntegerDivisionTest, truediv_32bit) {
  const int32_t a = 2;
  const int32_t b = 3;
  EXPECT_EQ(divide(a, b), 2.0 / 3.0);
}

TEST(ElementArithmeticIntegerDivisionTest, truediv_64bit) {
  const int64_t a = 2;
  const int64_t b = 3;
  EXPECT_EQ(divide(a, b), 2.0 / 3.0);
}

template <class T> struct int_as_first_arg : std::false_type {};
template <> struct int_as_first_arg<int64_t> : std::true_type {};
template <> struct int_as_first_arg<int32_t> : std::true_type {};
template <class A, class B>
struct int_as_first_arg<std::tuple<A, B>> : int_as_first_arg<A> {};

template <class... Ts> auto no_int_as_first_arg(const std::tuple<Ts...> &) {
  return !(int_as_first_arg<Ts>::value || ...);
}

template <class T> class ElementArithmeticDivisionTest : public testing::Test {
public:
  using Dividend = std::tuple_element_t<0, T>;
  using Divisor = std::tuple_element_t<1, T>;
  // The result is always double if both inputs are integers.
  using Quotient =
      std::conditional_t<std::is_integral_v<Dividend> &&
                             std::is_integral_v<Divisor>,
                         double, std::common_type_t<Dividend, Divisor>>;

  // Helpers to convert values to the correct types.
  constexpr Dividend dividend(Dividend x = Dividend{}) { return x; }
  constexpr Divisor divisor(Divisor x = Divisor{}) { return x; }
  constexpr Quotient quotient(Quotient x = Quotient{}) { return x; }
};

template <class Tuple, size_t... Indices>
auto DivisionTestTypes_impl(std::index_sequence<Indices...>)
    -> ::testing::Types<std::tuple_element_t<Indices, Tuple>...>;

using DivisionTestTypes =
    decltype(DivisionTestTypes_impl<scipp::core::arithmetic_type_pairs>(
        std::make_index_sequence<
            std::tuple_size_v<scipp::core::arithmetic_type_pairs>>()));

TYPED_TEST_SUITE(ElementArithmeticDivisionTest, DivisionTestTypes);

TYPED_TEST(ElementArithmeticDivisionTest, true_divide) {
  if constexpr (std::is_integral_v<typename TestFixture::Dividend> &&
                std::is_integral_v<typename TestFixture::Divisor>) {
    // Division of integers always returns double regardless of input precision.
    EXPECT_TRUE(
        (std::is_same_v<decltype(divide(this->dividend(), this->divisor())),
                        typename TestFixture::Quotient>));
  }

  EXPECT_DOUBLE_EQ(divide(this->dividend(2), this->divisor(1)),
                   this->quotient(2));
  EXPECT_DOUBLE_EQ(divide(this->dividend(1), this->divisor(2)),
                   this->quotient(0.5));
  EXPECT_DOUBLE_EQ(divide(this->dividend(-5), this->divisor(3)),
                   this->quotient(-1.6666666666666667));
  EXPECT_DOUBLE_EQ(divide(this->dividend(3), this->divisor(-4)),
                   this->quotient(-0.75));
}

TEST(ElementArithmeticDivisionTest, true_divide_variance) {
  const ValueAndVariance<double> a(4.2, 0.1);
  const ValueAndVariance<double> b(2.0, 1.2);
  const auto res = divide(a, b);
  EXPECT_DOUBLE_EQ(res.value, 1.1);
  EXPECT_DOUBLE_EQ(res.variance, 1.2609916732476865);
}

TEST(ElementArithmeticDivisionTest, units) {
  EXPECT_EQ(divide(units::m, units::s), units::m / units::s);
  EXPECT_EQ(floor_divide(units::m, units::s), units::m / units::s);
  EXPECT_EQ(mod(units::m, units::s), units::m / units::s);
}

TEST(ElementArithmeticIntegerDivisionTest, inplace_truediv_not_supported) {
  EXPECT_TRUE(no_int_as_first_arg(decltype(divide_equals)::types{}));
}

TEST(ElementArithmeticIntegerDivisionTest, mod) {
  // x mod 0 is not really defined, but numpy returns 0 and prints a warning
  EXPECT_EQ(mod(0, 0), 0);
  EXPECT_EQ(mod(1, 0), 0);
  EXPECT_EQ(mod(-1, 0), 0);

  EXPECT_EQ(mod(0, -2), 0);
  EXPECT_EQ(mod(1, -2), -1);
  EXPECT_EQ(mod(2, -2), 0);
  EXPECT_EQ(mod(3, -2), -1);
  EXPECT_EQ(mod(-1, -2), -1);
  EXPECT_EQ(mod(-2, -2), 0);
  EXPECT_EQ(mod(-3, -2), -1);

  EXPECT_EQ(mod(-4, 3), 2);
  EXPECT_EQ(mod(-3, 3), 0);
  EXPECT_EQ(mod(-2, 3), 1);
  EXPECT_EQ(mod(-1, 3), 2);
  EXPECT_EQ(mod(0, 3), 0);
  EXPECT_EQ(mod(1, 3), 1);
  EXPECT_EQ(mod(2, 3), 2);
  EXPECT_EQ(mod(3, 3), 0);
  EXPECT_EQ(mod(4, 3), 1);
}

TEST(ElementArithmeticIntegerDivisionTest, mod_equals) {
  constexpr auto check_mod = [](auto a, auto b, auto expected) {
    mod_equals(a, b);
    EXPECT_EQ(a, expected);
  };
  check_mod(units::m, units::s, units::m / units::s);

  check_mod(0, 0, 0);
  check_mod(1, 0, 0);
  check_mod(-1, 0, 0);

  check_mod(0, -2, 0);
  check_mod(1, -2, -1);
  check_mod(2, -2, 0);
  check_mod(3, -2, -1);
  check_mod(-1, -2, -1);
  check_mod(-2, -2, 0);
  check_mod(-3, -2, -1);

  check_mod(-4, 3, 2);
  check_mod(-3, 3, 0);
  check_mod(-2, 3, 1);
  check_mod(-1, 3, 2);
  check_mod(0, 3, 0);
  check_mod(1, 3, 1);
  check_mod(2, 3, 2);
  check_mod(3, 3, 0);
  check_mod(4, 3, 1);
}

class ElementNanArithmeticTest : public ::testing::Test {
protected:
  double x = 1.0;
  double y = 2.0;
  double dNaN = std::numeric_limits<double>::quiet_NaN();
};

TEST_F(ElementNanArithmeticTest, plus_equals) {
  auto expected = x + y;
  nan_plus_equals(x, y);
  EXPECT_EQ(expected, x);
}

TEST_F(ElementNanArithmeticTest, plus_equals_with_rhs_nan) {
  auto expected = x + 0;
  nan_plus_equals(x, dNaN);
  EXPECT_EQ(expected, x);
}
TEST_F(ElementNanArithmeticTest, plus_equals_with_lhs_nan) {
  auto expected = y + 0;
  auto lhs = dNaN;
  nan_plus_equals(lhs, y);
  EXPECT_EQ(expected, lhs);
}
TEST_F(ElementNanArithmeticTest, plus_equals_with_both_nan) {
  auto lhs = dNaN;
  nan_plus_equals(lhs, dNaN);
  EXPECT_EQ(0, lhs);
}
TEST_F(ElementNanArithmeticTest, plus_equals_with_rhs_nan_ValueAndVariance) {
  ValueAndVariance<double> asNaN{dNaN, 0};
  ValueAndVariance<double> z{1, 0};
  auto expected = z + ValueAndVariance<double>{0, 0};
  nan_plus_equals(z, asNaN);
  EXPECT_EQ(expected, z);
}
TEST_F(ElementNanArithmeticTest, plus_equals_with_lhs_nan_rhs_int) {
  auto lhs = dNaN;
  nan_plus_equals(lhs, 1);
  EXPECT_EQ(1.0, lhs);
}
TEST_F(ElementNanArithmeticTest, plus_equals_with_rhs_int_lhs_nan) {
  auto lhs = 1;
  nan_plus_equals(lhs, dNaN);
  EXPECT_EQ(1, lhs);
}
TEST_F(ElementNanArithmeticTest, plus_equals_with_rhs_int_lhs_int) {
  auto lhs = 1;
  nan_plus_equals(lhs, 2);
  EXPECT_EQ(3, lhs);
}
