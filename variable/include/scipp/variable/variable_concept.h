// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp-variable_export.h"
#include "scipp/common/deep_ptr.h"
#include "scipp/common/index.h"
#include "scipp/common/span.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/dtype.h"
#include "scipp/units/unit.h"

#include <memory>

namespace scipp::variable {

class Variable;
class VariableConstView;
class VariableView;
class VariableConcept;

class SCIPP_VARIABLE_EXPORT VariableConceptHandle
    : public scipp::deep_ptr<VariableConcept> {
public:
  using scipp::deep_ptr<VariableConcept>::deep_ptr;
  template <class T> VariableConceptHandle(T object);
  VariableConceptHandle(VariableConceptHandle &&) = default;
  VariableConceptHandle(const VariableConceptHandle &other);
  VariableConceptHandle &operator=(VariableConceptHandle &&) = default;
  VariableConceptHandle &operator=(const VariableConceptHandle &other);
};

/// Abstract base class for any data that can be held by Variable. This is using
/// so-called concept-based polymorphism, see talks by Sean Parent.
///
/// This is the most generic representation for a multi-dimensional array of
/// data. More operations are supportd by the typed DataModel.
class SCIPP_VARIABLE_EXPORT VariableConcept {
public:
  VariableConcept(const Dimensions &dimensions, const units::Unit &unit);
  virtual ~VariableConcept() = default;

  virtual VariableConceptHandle clone() const = 0;
  virtual VariableConceptHandle
  makeDefaultFromParent(const Dimensions &dims) const = 0;
  virtual VariableConceptHandle
  makeDefaultFromParent(const VariableConstView &shape) const = 0;

  virtual DType dtype() const noexcept = 0;
  const Dimensions &dims() const { return m_dimensions; }
  const units::Unit &unit() const { return m_unit; }

  void setUnit(const units::Unit &unit) { m_unit = unit; }

  virtual bool hasVariances() const noexcept = 0;
  virtual void setVariances(Variable &&variances) = 0;

  virtual bool equals(const VariableConstView &a,
                      const VariableConstView &b) const = 0;
  virtual void copy(const VariableConstView &src,
                    const VariableView &dest) const = 0;
  virtual void assign(const VariableConcept &other) = 0;
  virtual scipp::index dtype_size() const = 0;

  virtual VariableConstView bin_indices() const = 0;

  friend class Variable;

private:
  Dimensions m_dimensions;
  units::Unit m_unit;
};

template <class T>
VariableConceptHandle::VariableConceptHandle(T object)
    : VariableConceptHandle(
          std::unique_ptr<VariableConcept>(std::move(object))) {}

} // namespace scipp::variable
