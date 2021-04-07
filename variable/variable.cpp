// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <units/units.hpp>

#include "scipp/variable/variable.h"

#include "scipp/core/dtype.h"
#include "scipp/core/except.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/creation.h"
#include "scipp/variable/shape.h"
#include "scipp/variable/variable_concept.h"

namespace scipp::variable {

Variable::Variable(const VariableConstView &slice)
    : Variable(empty_like(slice)) {
  data().copy(slice, *this);
}

/// Construct from parent with same dtype, unit, and hasVariances but new dims.
///
/// In the case of bucket variables the buffer size is set to zero.
Variable::Variable(const Variable &parent, const Dimensions &dims)
    : m_dims(dims), m_strides(dims),
      m_object(parent.data().makeDefaultFromParent(dims.volume())) {}

Variable::Variable(const VariableConstView &parent, const Dimensions &dims)
    : m_dims(dims), m_strides(dims),
      m_object(
          parent.underlying().data().makeDefaultFromParent(dims.volume())) {}

Variable::Variable(const VariableConstView &parent, const Dimensions &dims,
                   VariableConceptHandle data)
    : m_dims(dims), m_strides(dims), m_object(std::move(data)) {}

Variable::Variable(const Dimensions &dims, VariableConceptHandle data)
    : m_dims(dims), m_strides(dims), m_object(std::move(data)) {}

Variable::Variable(const llnl::units::precise_measurement &m)
    : Variable(m.value() * units::Unit(m.units())) {}

VariableConstView::VariableConstView(const Variable &variable,
                                     const Dimensions &dims)
    : m_variable(&variable), m_dims(dims), m_dataDims(dims) {
  // TODO implement reshape differently, not with a special constructor?
  expect_same_volume(m_variable->dims(), dims);
}

VariableConstView::VariableConstView(const VariableConstView &slice,
                                     const Dim dim, const scipp::index begin,
                                     const scipp::index end) {
  *this = slice;
  m_offset += begin * m_dataDims.offset(dim);
  if (end == -1)
    m_dims.erase(dim);
  else
    m_dims.resize(dim, end - begin);
  // See implementation of ViewIndex regarding this relabeling.
  for (const auto &label : m_dataDims.labels())
    if (label != Dim::Invalid && !m_dims.contains(label))
      m_dataDims.relabel(m_dataDims.index(label), Dim::Invalid);
}

VariableView::VariableView(Variable &variable, const Dimensions &dims)
    : VariableConstView(variable, dims), m_mutableVariable(&variable) {}

VariableView::VariableView(const VariableView &slice, const Dim dim,
                           const scipp::index begin, const scipp::index end)
    : VariableConstView(slice, dim, begin, end),
      m_mutableVariable(slice.m_mutableVariable) {}

void Variable::setDims(const Dimensions &dimensions) {
  if (dimensions.volume() == dims().volume()) {
    if (dimensions != dims()) {
      m_dims = dimensions;
      m_strides = Strides(dimensions);
    }
    return;
  }
  m_dims = dimensions;
  m_strides = Strides(dimensions);
  m_object = m_object->makeDefaultFromParent(dimensions.volume());
}

bool Variable::operator==(const VariableConstView &other) const {
  if (!*this || !other)
    return static_cast<bool>(*this) == static_cast<bool>(other);
  // Note: Not comparings strides
  return dims() == other.dims() &&
         data().equals(*this, other);
}

bool Variable::operator!=(const VariableConstView &other) const {
  return !(*this == other);
}

Variable &Variable::assign(const VariableConstView &other) {
  VariableView(*this).assign(other);
  return *this;
}

template <class T> VariableView VariableView::assign(const T &other) const {
  if (*this == VariableConstView(other))
    return *this; // Self-assignment, return early.
  underlying().data().copy(other, *this);
  return *this;
}

template SCIPP_VARIABLE_EXPORT VariableView
VariableView::assign(const Variable &) const;
template SCIPP_VARIABLE_EXPORT VariableView
VariableView::assign(const VariableConstView &) const;
template SCIPP_VARIABLE_EXPORT VariableView
VariableView::assign(const VariableView &) const;

bool VariableConstView::operator==(const VariableConstView &other) const {
  if (!*this || !other)
    return static_cast<bool>(*this) == static_cast<bool>(other);
  // Always use deep comparison (pointer comparison does not make sense since we
  // may be looking at a different section).
  return dims() == other.dims() && underlying().data().equals(*this, other);
}

bool VariableConstView::operator!=(const VariableConstView &other) const {
  return !(*this == other);
}

void VariableView::setUnit(const units::Unit &unit) const {
  expectCanSetUnit(unit);
  m_mutableVariable->setUnit(unit);
}

void VariableView::expectCanSetUnit(const units::Unit &unit) const {
  if ((this->unit() != unit) && (dims() != m_mutableVariable->dims()))
    throw except::UnitError("Partial view on data of variable cannot be used "
                            "to change the unit.");
}

VariableConstView Variable::slice(const Slice slice) const & {
  return {*this, slice.dim(), slice.begin(), slice.end()};
}

Variable Variable::slice(const Slice slice) const && {
  return Variable{this->slice(slice)};
}

VariableView Variable::slice(const Slice slice) & {
  return {*this, slice.dim(), slice.begin(), slice.end()};
}

Variable Variable::slice(const Slice slice) && {
  return Variable{this->slice(slice)};
}

VariableConstView VariableConstView::slice(const Slice slice) const {
  return VariableConstView(*this, slice.dim(), slice.begin(), slice.end());
}

VariableView VariableView::slice(const Slice slice) const {
  return VariableView(*this, slice.dim(), slice.begin(), slice.end());
}

VariableConstView
VariableConstView::transpose(const std::vector<Dim> &order) const {
  auto transposed(*this);
  transposed.m_dims = core::transpose(dims(), order);
  return transposed;
}

VariableView VariableView::transpose(const std::vector<Dim> &order) const {
  auto transposed(*this);
  transposed.m_dims = core::transpose(dims(), order);
  return transposed;
}

std::vector<scipp::index> VariableConstView::strides() const {
  const auto parent = m_variable->dims();
  std::vector<scipp::index> strides;
  for (const auto &label : parent.labels())
    if (dims().contains(label))
      strides.emplace_back(parent.offset(label));
  return strides;
}

bool VariableConstView::is_trivial() const noexcept {
  return m_variable && m_offset == 0 && m_dims == m_variable->dims() &&
         m_dataDims == m_variable->dims();
}

void Variable::rename(const Dim from, const Dim to) {
  if (dims().contains(from))
    m_dims.relabel(dims().index(from), to);
}

/// Rename dims of a view. Does NOT rename dims of the underlying variable.
void VariableConstView::rename(const Dim from, const Dim to) {
  if (dims().contains(from)) {
    m_dims.relabel(m_dims.index(from), to);
    m_dataDims.relabel(m_dataDims.index(from), to);
  }
}

void Variable::setVariances(Variable v) {
  if (v) {
    core::expect::equals(unit(), v.unit());
    core::expect::equals(dims(), v.dims());
  }
  data().setVariances(std::move(v));
}

void VariableView::setVariances(Variable v) const {
  if (m_offset == 0 && m_dims == m_variable->dims() &&
      m_dataDims == m_variable->dims())
    m_mutableVariable->setVariances(std::move(v));
  else
    throw except::VariancesError(
        "Cannot add variances via sliced or reshaped view of Variable.");
}

namespace detail {
void throw_keyword_arg_constructor_bad_dtype(const DType dtype) {
  throw except::TypeError("Can't create the Variable with type " +
                          to_string(dtype) +
                          " with such values and/or variances.");
}

void expect0D(const Dimensions &dims) {
  core::expect::equals(dims, Dimensions());
}

} // namespace detail

VariableConstView Variable::bin_indices() const { return data().bin_indices(); }

VariableConstView VariableConstView::bin_indices() const {
  auto view = *this;
  view.m_variable = &underlying().bin_indices().underlying();
  return view;
}

} // namespace scipp::variable
