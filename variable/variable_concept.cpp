#include "scipp/variable/variable_concept.h"
#include "scipp/core/dimensions.h"

namespace scipp::variable {

VariableConceptHandle::VariableConceptHandle(const VariableConceptHandle &other)
    : VariableConceptHandle(other ? other->clone() : VariableConceptHandle()) {}

VariableConceptHandle &
VariableConceptHandle::operator=(const VariableConceptHandle &other) {
  if (*this && other) {
    // Avoid allocation of new element_array if output is of correct shape.
    // This yields a 5x speedup in assignment operations of variables.
    auto &varconcept = **this;
    auto &otherConcept = *other;
    if (varconcept.dtype() == otherConcept.dtype() &&
        varconcept.dims() == otherConcept.dims() &&
        varconcept.hasVariances() == otherConcept.hasVariances()) {
      varconcept.assign(otherConcept);
      return *this;
    }
  }
  return *this = other ? other->clone() : VariableConceptHandle();
}

VariableConcept::VariableConcept(const Dimensions &dimensions,
                                 const units::Unit &unit)
    : m_dimensions(dimensions), m_unit(unit) {}

} // namespace scipp::variable
