// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <algorithm>
#include <numeric>

#include "scipp/common/except.h"
#include "scipp/core/element/arithmetic.h"
#include "scipp/core/element/cumulative.h"
#include "scipp/core/subbin_sizes.h"

namespace scipp::core {

SubbinSizes::SubbinSizes(const scipp::index offset, container_type &&sizes)
    : m_offset(offset), m_sizes(std::move(sizes)) {
  if (offset < 0)
    throw std::logic_error("Bad offset in class SubbinSizes.");
}

SubbinSizes::SubbinSizes(const scipp::index value)
    : m_offset(0), m_sizes({value}) {}

void SubbinSizes::operator=(const scipp::index value) {
  for (auto &size : m_sizes)
    size = value;
}

SubbinSizes &SubbinSizes::operator+=(const SubbinSizes &other) {
  if (other.offset() < offset()) // avoid realloc if possible
    return *this = *this + other;
  scipp::index current = other.offset() - offset();
  const auto length = current + scipp::size(other.sizes());
  if (length > scipp::size(sizes()))
    m_sizes.resize(length);
  for (const auto &x : other.sizes())
    m_sizes[current++] += x;
  return *this;
}

SubbinSizes &SubbinSizes::operator-=(const SubbinSizes &other) {
  return *this = *this - other;
}

SubbinSizes SubbinSizes::cumsum_exclusive() const {
  auto out = sizes();
  scipp::index sum = 0;
  for (auto &x : out)
    element::exclusive_scan(sum, x);
  return {offset(), std::move(out)};
}

scipp::index SubbinSizes::sum() const {
  return std::accumulate(sizes().begin(), sizes().end(), scipp::index{0});
}

void SubbinSizes::trim_to(const SubbinSizes &other) {
  auto out = other; // TODO in principle this allocation could be avoided, but
                    // initial implementation attempts got messy.
  out = 0;
  // full index begin/end
  const auto begin = std::max(offset(), out.offset());

  const auto end = std::min(offset() + scipp::size(sizes()),
                            out.offset() + scipp::size(out.sizes()));

  const auto ibegin =
      sizes().begin() +
      std::clamp(begin - offset(), scipp::index(0), scipp::size(sizes()));

  const auto obegin =
      out.m_sizes.begin() + std::max(begin - out.offset(), scipp::index(0));

  std::copy_n(ibegin, std::max(scipp::index(0), end - begin), obegin);
  *this = std::move(out);
}

SubbinSizes &SubbinSizes::add_intersection(const SubbinSizes &other) {
  scipp::index delta = other.offset() - offset();
  auto i = std::max(scipp::index(0), delta);
  auto j = std::max(scipp::index(0), -delta);
  for (; j < scipp::size(other.sizes()) && i < scipp::size(sizes()); ++j, ++i) {
    m_sizes[i] += other.sizes()[j];
  }
  return *this;
}

bool operator==(const SubbinSizes &a, const SubbinSizes &b) {
  return (a.offset() == b.offset()) && (a.sizes() == b.sizes());
}

template <class Op>
SubbinSizes binary(const SubbinSizes &a, const SubbinSizes &b, Op op) {
  const auto begin = std::min(a.offset(), b.offset());
  const auto end =
      std::max(a.offset() + a.sizes().size(), b.offset() + b.sizes().size());
  typename SubbinSizes::container_type sizes(end - begin);
  scipp::index current = a.offset() - begin;
  for (const auto &size : a.sizes())
    sizes[current++] += size;
  current = b.offset() - begin;
  for (const auto &size : b.sizes())
    op(sizes[current++], size);
  return {begin, std::move(sizes)};
}

SubbinSizes operator+(const SubbinSizes &a, const SubbinSizes &b) {
  return binary(a, b, element::add_equals);
}

SubbinSizes operator-(const SubbinSizes &a, const SubbinSizes &b) {
  return binary(a, b, element::subtract_equals);
}

} // namespace scipp::core
