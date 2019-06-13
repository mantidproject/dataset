// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#ifndef VARIABLE_H
#define VARIABLE_H

#include <type_traits>
#include <variant>

#include <Eigen/Dense>

#include "dimensions.h"
#include "dtype.h"
#include "index.h"
#include "scipp/units/unit.h"
#include "span.h"
#include "variable_view.h"
#include "vector.h"

namespace scipp::core {

/// Helper for passing slicing parameters.
struct Slice {
  Slice(const Dim dim, const scipp::index begin, const scipp::index end = -1)
      : dim(dim), begin(begin), end(end) {}
  Dim dim;
  scipp::index begin;
  scipp::index end;
};

template <class T>
using sparse_container = boost::container::small_vector<T, 8>;
template <class T> struct is_sparse_container : std::false_type {};
template <class T>
struct is_sparse_container<sparse_container<T>> : std::true_type {};

template <class T> struct is_sparse : std::false_type {};
template <class T> struct is_sparse<sparse_container<T>> : std::true_type {};
template <class T> struct is_sparse<sparse_container<T> &> : std::true_type {};
template <class T>
struct is_sparse<const sparse_container<T> &> : std::true_type {};
template <class T> inline constexpr bool is_sparse_v = is_sparse<T>::value;

// std::vector<bool> may have a packed non-thread-safe implementation which we
// need to avoid. Therefore we use std::vector<Bool> instead.
template <class T> struct underlying_type { using type = T; };
template <> struct underlying_type<bool> { using type = Bool; };
template <class T> using underlying_type_t = typename underlying_type<T>::type;

class Variable;
template <class... Known> class VariableConceptHandle_impl;
// Any item type that is listed here explicitly can be used with the templated
// `transform`, i.e., we can pass arbitrary functors/lambdas to process data.
using VariableConceptHandle = VariableConceptHandle_impl<
    double, float, int64_t, Eigen::Vector3d, sparse_container<double>,
    sparse_container<float>, sparse_container<int64_t>>;

/// Abstract base class for any data that can be held by Variable. Also used to
/// hold views to data by (Const)VariableProxy. This is using so-called
/// concept-based polymorphism, see talks by Sean Parent.
///
/// This is the most generic representation for a multi-dimensional array of
/// data. More operations are supportd by the partially-typed VariableConceptT.
class VariableConcept {
public:
  VariableConcept(const Dimensions &dimensions);
  virtual ~VariableConcept() = default;

  virtual DType dtype(bool sparse = false) const noexcept = 0;

  virtual VariableConceptHandle clone() const = 0;
  virtual VariableConceptHandle clone(const Dimensions &dims) const = 0;
  virtual VariableConceptHandle makeView() const = 0;
  virtual VariableConceptHandle makeView() = 0;
  virtual VariableConceptHandle makeView(const Dim dim,
                                         const scipp::index begin,
                                         const scipp::index end = -1) const = 0;
  virtual VariableConceptHandle makeView(const Dim dim,
                                         const scipp::index begin,
                                         const scipp::index end = -1) = 0;

  virtual VariableConceptHandle reshape(const Dimensions &dims) const = 0;
  virtual VariableConceptHandle reshape(const Dimensions &dims) = 0;

  virtual bool operator==(const VariableConcept &other) const = 0;

  virtual bool isContiguous() const = 0;
  virtual bool isView() const = 0;
  virtual bool isConstView() const = 0;
  virtual bool hasVariances() const noexcept = 0;

  virtual scipp::index size() const = 0;
  virtual void copy(const VariableConcept &other, const Dim dim,
                    const scipp::index offset, const scipp::index otherBegin,
                    const scipp::index otherEnd) = 0;

  const Dimensions &dims() const { return m_dimensions; }

  friend class Variable;

private:
  Dimensions m_dimensions;
};

/// Partially typed implementation of VariableConcept. This is a common base
/// class for DataModel<T> and ViewModel<T>. The former holds data in a
/// contiguous array, whereas the latter is a (potentially non-contiguous) view
/// into the former. This base class implements functionality that is common to
/// both, for a specific T.
template <class T> class VariableConceptT : public VariableConcept {
public:
  using value_type = T;

  VariableConceptT(const Dimensions &dimensions)
      : VariableConcept(dimensions) {}

  DType dtype(bool sparse = false) const noexcept override {
    if (!sparse)
      return scipp::core::dtype<T>;
    if constexpr (is_sparse_container<T>::value)
      return scipp::core::dtype<typename T::value_type>;
    std::terminate();
  }
  static DType static_dtype() noexcept { return scipp::core::dtype<T>; }

  virtual scipp::span<T> values() = 0;
  virtual scipp::span<T> values(const Dim dim, const scipp::index begin,
                                const scipp::index end) = 0;
  virtual scipp::span<const T> values() const = 0;
  virtual scipp::span<const T> values(const Dim dim, const scipp::index begin,
                                      const scipp::index end) const = 0;
  virtual scipp::span<T> variances() = 0;
  virtual scipp::span<T> variances(const Dim dim, const scipp::index begin,
                                   const scipp::index end) = 0;
  virtual scipp::span<const T> variances() const = 0;
  virtual scipp::span<const T> variances(const Dim dim,
                                         const scipp::index begin,
                                         const scipp::index end) const = 0;
  virtual VariableView<T> valuesView(const Dimensions &dims) = 0;
  virtual VariableView<T> valuesView(const Dimensions &dims, const Dim dim,
                                     const scipp::index begin) = 0;
  virtual VariableView<const T> valuesView(const Dimensions &dims) const = 0;
  virtual VariableView<const T> valuesView(const Dimensions &dims,
                                           const Dim dim,
                                           const scipp::index begin) const = 0;
  virtual VariableView<T> variancesView(const Dimensions &dims) = 0;
  virtual VariableView<T> variancesView(const Dimensions &dims, const Dim dim,
                                        const scipp::index begin) = 0;
  virtual VariableView<const T> variancesView(const Dimensions &dims) const = 0;
  virtual VariableView<const T>
  variancesView(const Dimensions &dims, const Dim dim,
                const scipp::index begin) const = 0;
  virtual VariableView<const T>
  valuesReshaped(const Dimensions &dims) const = 0;
  virtual VariableView<T> valuesReshaped(const Dimensions &dims) = 0;
  virtual VariableView<const T>
  variancesReshaped(const Dimensions &dims) const = 0;
  virtual VariableView<T> variancesReshaped(const Dimensions &dims) = 0;

  virtual std::unique_ptr<VariableConceptT> copyT() const = 0;

  VariableConceptHandle makeView() const override;

  VariableConceptHandle makeView() override;

  VariableConceptHandle makeView(const Dim dim, const scipp::index begin,
                                 const scipp::index end) const override;

  VariableConceptHandle makeView(const Dim dim, const scipp::index begin,
                                 const scipp::index end) override;

  VariableConceptHandle reshape(const Dimensions &dims) const override;

  VariableConceptHandle reshape(const Dimensions &dims) override;

  bool operator==(const VariableConcept &other) const override;
  void copy(const VariableConcept &other, const Dim dim,
            const scipp::index offset, const scipp::index otherBegin,
            const scipp::index otherEnd) override;
};

template <class... Known> class VariableConceptHandle_impl {
public:
  VariableConceptHandle_impl()
      : m_object(std::unique_ptr<VariableConcept>(nullptr)) {}
  template <class T> VariableConceptHandle_impl(T object) {
    using value_t = typename T::element_type::value_type;
    if constexpr ((std::is_same_v<value_t, Known> || ...))
      m_object = std::unique_ptr<VariableConceptT<value_t>>(std::move(object));
    else
      m_object = std::unique_ptr<VariableConcept>(std::move(object));
  }
  VariableConceptHandle_impl(VariableConceptHandle_impl &&) = default;
  VariableConceptHandle_impl(const VariableConceptHandle_impl &other)
      : VariableConceptHandle_impl(other ? other->clone()
                                         : VariableConceptHandle_impl()) {}
  VariableConceptHandle_impl &
  operator=(VariableConceptHandle_impl &&) = default;
  VariableConceptHandle_impl &
  operator=(const VariableConceptHandle_impl &other) {
    return *this = other ? other->clone() : VariableConceptHandle_impl();
  }

  explicit operator bool() const noexcept {
    return std::visit([](auto &&ptr) { return bool(ptr); }, m_object);
  }
  VariableConcept &operator*() const {
    return std::visit([](auto &&arg) -> VariableConcept & { return *arg; },
                      m_object);
  }
  VariableConcept *operator->() const {
    return std::visit(
        [](auto &&arg) -> VariableConcept * { return arg.operator->(); },
        m_object);
  }

  const auto &variant() const noexcept { return m_object; }

private:
  std::variant<std::unique_ptr<VariableConcept>,
               std::unique_ptr<VariableConceptT<Known>>...>
      m_object;
};

template <class... Tags> class ZipView;
class VariableConstProxy;
class VariableProxy;

namespace detail {
template <class T> struct default_init {
  static T value() { return T(); }
};
// Eigen does not zero-initialize matrices (vectors), which is a recurrent
// source of bugs. Variable does zero-init instead.
template <class T, int Rows, int Cols>
struct default_init<Eigen::Matrix<T, Rows, Cols>> {
  static Eigen::Matrix<T, Rows, Cols> value() {
    return Eigen::Matrix<T, Rows, Cols>::Zero();
  }
};
} // namespace detail

/// Variable is a type-erased handle to any data structure representing a
/// multi-dimensional array. It has a name, a unit, and a set of named
/// dimensions.
class Variable {
public:
  Variable() = default;
  // Having this non-explicit is convenient when passing (potential)
  // variable slices to functions that do not support slices, but implicit
  // conversion may introduce risks, so there is a trade-of here.
  Variable(const VariableConstProxy &slice);
  Variable(const Variable &parent, const Dimensions &dims);
  Variable(const VariableConstProxy &parent, const Dimensions &dims);
  Variable(const Variable &parent, VariableConceptHandle data);

  template <class T>
  Variable(const units::Unit unit, const Dimensions &dimensions, T object);
  template <class T>
  Variable(const units::Unit unit, const Dimensions &dimensions, T values,
           T variances);
  template <class T>
  Variable(const Dimensions &dimensions, std::initializer_list<T> values)
      : Variable(units::dimensionless, std::move(dimensions),
                 Vector<underlying_type_t<T>>(values.begin(), values.end())) {}

  explicit operator bool() const noexcept { return m_object.operator bool(); }

  units::Unit unit() const { return m_unit; }
  void setUnit(const units::Unit &unit) { m_unit = unit; }

  Dimensions dims() const && { return m_object->dims(); }
  const Dimensions &dims() const & { return m_object->dims(); }
  void setDims(const Dimensions &dimensions);

  DType dtype() const noexcept { return data().dtype(dims().sparse()); }

  bool hasVariances() const noexcept { return data().hasVariances(); }

  template <class T> auto values() const { return scipp::span(cast<T>()); }
  template <class T> auto values() { return scipp::span(cast<T>()); }
  template <class T> auto variances() const {
    return scipp::span(cast<T>(true));
  }
  template <class T> auto variances() { return scipp::span(cast<T>(true)); }
  template <class T> auto sparseValues() const {
    return scipp::span(cast<sparse_container<T>>());
  }
  template <class T> auto sparseValues() {
    return scipp::span(cast<sparse_container<T>>());
  }
  template <class T> auto sparseVariances() const {
    return scipp::span(cast<sparse_container<T>>(true));
  }
  template <class T> auto sparseVariances() {
    return scipp::span(cast<sparse_container<T>>(true));
  }
  template <class T> auto sparseSpan() const { return sparseValues<T>(); }
  template <class T> auto sparseSpan() { return sparseValues<T>(); }

  // ATTENTION: It is really important to avoid any function returning a
  // (Const)VariableProxy for rvalue Variable. Otherwise the resulting slice
  // will point to free'ed memory.
  VariableConstProxy slice(const Slice slice) const &;
  Variable slice(const Slice slice) const &&;
  VariableProxy slice(const Slice slice) &;
  Variable slice(const Slice slice) &&;

  VariableConstProxy reshape(const Dimensions &dims) const &;
  VariableProxy reshape(const Dimensions &dims) &;
  // Note: Do we have to delete the `const &&` version? Consider
  //   const Variable var;
  //   std::move(var).reshape({});
  // This calls `reshape() const &` but in this case it is not a temporary and
  // will not go out of scope, so that is ok (unless someone changes var and
  // expects the reshaped view to be still valid).
  Variable reshape(const Dimensions &dims) &&;

  bool operator==(const Variable &other) const;
  bool operator==(const VariableConstProxy &other) const;
  bool operator!=(const Variable &other) const;
  bool operator!=(const VariableConstProxy &other) const;
  Variable operator-() const;
  Variable &operator+=(const Variable &other) &;
  Variable &operator+=(const VariableConstProxy &other) &;
  Variable &operator+=(const double value) &;
  Variable &operator-=(const Variable &other) &;
  Variable &operator-=(const VariableConstProxy &other) &;
  Variable &operator-=(const double value) &;
  Variable &operator*=(const Variable &other) &;
  Variable &operator*=(const VariableConstProxy &other) &;
  Variable &operator*=(const double value) &;
  template <class T>
  Variable &operator*=(const boost::units::quantity<T> &quantity) & {
    setUnit(unit() * units::Unit(T{}));
    return *this *= quantity.value();
  }
  Variable &operator/=(const Variable &other) &;
  Variable &operator/=(const VariableConstProxy &other) &;
  Variable &operator/=(const double value) &;
  template <class T>
  Variable &operator/=(const boost::units::quantity<T> &quantity) & {
    setUnit(unit() / units::Unit(T{}));
    return *this /= quantity.value();
  }

  const VariableConcept &data() const && = delete;
  const VariableConcept &data() const & { return *m_object; }
  VariableConcept &data() && = delete;
  VariableConcept &data() & { return *m_object; }

  const VariableConceptHandle &dataHandle() const && = delete;
  const VariableConceptHandle &dataHandle() const & { return m_object; }

  template <class... Tags> friend class ZipView;

private:
  template <class T>
  const Vector<underlying_type_t<T>> &cast(const bool variances = false) const;
  template <class T>
  Vector<underlying_type_t<T>> &cast(const bool variances = false);

  // Used by ZipView. Need to find a better way instead of having everyone as
  // friend.
  Dimensions &mutableDimensions() { return m_object->m_dimensions; }

  units::Unit m_unit;
  VariableConceptHandle m_object;
};

template <class T> Variable makeVariable(const Dimensions &dimensions) {
  if (dimensions.sparse())
    return Variable(
        units::dimensionless, std::move(dimensions),
        Vector<sparse_container<underlying_type_t<T>>>(dimensions.volume()));
  else
    return Variable(units::dimensionless, std::move(dimensions),
                    Vector<underlying_type_t<T>>(
                        dimensions.volume(),
                        detail::default_init<underlying_type_t<T>>::value()));
}

template <class T>
Variable makeVariableWithVariances(const Dimensions &dimensions) {
  if (dimensions.sparse())
    return Variable(
        units::dimensionless, std::move(dimensions),
        Vector<sparse_container<underlying_type_t<T>>>(dimensions.volume()),
        Vector<sparse_container<underlying_type_t<T>>>(dimensions.volume()));
  else
    return Variable(units::dimensionless, std::move(dimensions),
                    Vector<underlying_type_t<T>>(
                        dimensions.volume(),
                        detail::default_init<underlying_type_t<T>>::value()),
                    Vector<underlying_type_t<T>>(
                        dimensions.volume(),
                        detail::default_init<underlying_type_t<T>>::value()));
}

template <class T>
Variable makeVariable(const std::initializer_list<Dim> &dims,
                      const std::initializer_list<scipp::index> &shape) {
  return makeVariable<T>(Dimensions(dims, shape));
}

template <class T> Variable makeVariable(T value) {
  return Variable(units::dimensionless, Dimensions{},
                  Vector<underlying_type_t<T>>(1, value));
}

template <class T> Variable makeVariable(T value, T variance) {
  return Variable(units::dimensionless, Dimensions{},
                  Vector<underlying_type_t<T>>(1, value),
                  Vector<underlying_type_t<T>>(1, variance));
}

template <class T, class T2 = T>
Variable makeVariable(const Dimensions &dimensions,
                      std::initializer_list<T2> values,
                      std::initializer_list<T2> variances = {}) {
  if constexpr (is_sparse_v<T2>) {
    return Variable(units::dimensionless, std::move(dimensions),
                    Vector<sparse_container<underlying_type_t<T>>>(
                        values.begin(), values.end()),
                    Vector<sparse_container<underlying_type_t<T>>>(
                        variances.begin(), variances.end()));
  } else {
    return Variable(
        units::dimensionless, std::move(dimensions),
        Vector<underlying_type_t<T>>(values.begin(), values.end()),
        Vector<underlying_type_t<T>>(variances.begin(), variances.end()));
  }
}

template <class T, class T2 = T>
Variable makeVariable(const Dimensions &dimensions, const units::Unit unit,
                      std::initializer_list<T2> values,
                      std::initializer_list<T2> variances = {}) {
  return Variable(
      unit, std::move(dimensions),
      Vector<underlying_type_t<T>>(values.begin(), values.end()),
      Vector<underlying_type_t<T>>(variances.begin(), variances.end()));
}

namespace detail {
template <class... N> struct is_vector : std::false_type {};
template <class N, class A>
struct is_vector<std::vector<N, A>> : std::true_type {};

template <int I, class... Ts> decltype(auto) nth(Ts &&... ts) {
  return std::get<I>(std::forward_as_tuple(ts...));
}

template <int I, class... Ts>
using nth_t = decltype(std::get<I>(std::declval<std::tuple<Ts...>>()));
} // namespace detail

template <class T, class... Args>
Variable makeVariable(const Dimensions &dimensions, Args &&... args) {
  // Note: Using `if constexpr` instead of another overload, since overloading
  // on universal reference arguments is problematic.
  if constexpr (detail::is_vector<std::remove_cv_t<
                    std::remove_reference_t<Args>>...>::value) {
    // Copies to aligned memory.
    return Variable(units::dimensionless, std::move(dimensions),
                    Vector<underlying_type_t<T>>(args.begin(), args.end())...);
  } else if constexpr (sizeof...(Args) == 1 &&
                       (std::is_convertible_v<Args, units::Unit> && ...)) {
    return Variable(args..., std::move(dimensions),
                    Vector<underlying_type_t<T>>(
                        dimensions.volume(),
                        detail::default_init<underlying_type_t<T>>::value()));
  } else if constexpr (sizeof...(Args) == 2) {
    if constexpr (std::is_convertible_v<detail::nth_t<0, Args...>,
                                        std::vector<T>> &&
                  std::is_convertible_v<detail::nth_t<1, Args...>,
                                        std::vector<T>>) {
      return Variable(
          units::dimensionless, std::move(dimensions),
          Vector<underlying_type_t<T>>(detail::nth<0>(args...).begin(),
                                       detail::nth<0>(args...).end()),
          Vector<underlying_type_t<T>>(detail::nth<1>(args...).begin(),
                                       detail::nth<1>(args...).end()));
    } else {
      return Variable(
          units::dimensionless, std::move(dimensions),
          Vector<underlying_type_t<T>>(std::forward<Args>(args)...));
    }
  } else if constexpr (sizeof...(Args) == 3) {
    if constexpr (std::is_convertible_v<detail::nth_t<0, Args...>,
                                        units::Unit> &&
                  std::is_convertible_v<detail::nth_t<1, Args...>,
                                        std::vector<T>> &&
                  std::is_convertible_v<detail::nth_t<2, Args...>,
                                        std::vector<T>>) {
      return Variable(
          detail::nth<0>(args...), std::move(dimensions),
          Vector<underlying_type_t<T>>(detail::nth<1>(args...).begin(),
                                       detail::nth<1>(args...).end()),
          Vector<underlying_type_t<T>>(detail::nth<2>(args...).begin(),
                                       detail::nth<2>(args...).end()));
    } else {
      return Variable(
          units::dimensionless, std::move(dimensions),
          Vector<underlying_type_t<T>>(std::forward<Args>(args)...));
    }
  } else {
    return Variable(units::dimensionless, std::move(dimensions),
                    Vector<underlying_type_t<T>>(std::forward<Args>(args)...));
  }
}

/// Non-mutable view into (a subset of) a Variable.
class VariableConstProxy {
public:
  explicit VariableConstProxy(const Variable &variable)
      : m_variable(&variable) {}
  VariableConstProxy(const Variable &variable, const Dimensions &dims)
      : m_variable(&variable), m_view(variable.data().reshape(dims)) {}
  VariableConstProxy(const VariableConstProxy &other) = default;
  VariableConstProxy(const Variable &variable, const Dim dim,
                     const scipp::index begin, const scipp::index end = -1)
      : m_variable(&variable),
        m_view(variable.data().makeView(dim, begin, end)) {}
  VariableConstProxy(const VariableConstProxy &slice, const Dim dim,
                     const scipp::index begin, const scipp::index end = -1)
      : m_variable(slice.m_variable),
        m_view(slice.data().makeView(dim, begin, end)) {}

  explicit operator bool() const noexcept {
    return m_variable->operator bool();
  }

  VariableConstProxy slice(const Slice slice) const {
    return VariableConstProxy(*this, slice.dim, slice.begin, slice.end);
  }

  // Note the return type. Reshaping a non-contiguous slice cannot return a
  // slice in general so we must return a copy of the data.
  Variable reshape(const Dimensions &dims) const;

  units::Unit unit() const { return m_variable->unit(); }

  // Note: Returning by value to avoid issues with referencing a temporary
  // (VariableProxy is returned by-value from DatasetSlice).
  Dimensions dims() const {
    if (m_view)
      return m_view->dims();
    else
      return m_variable->dims();
  }

  std::vector<scipp::index> strides() const {
    const auto parent = m_variable->dims();
    std::vector<scipp::index> strides;
    for (const auto &label : parent.labels())
      if (dims().contains(label))
        strides.emplace_back(parent.offset(label));
    return strides;
  }

  DType dtype() const noexcept { return m_variable->dtype(); }

  const VariableConcept &data() const && = delete;
  const VariableConcept &data() const & {
    if (m_view)
      return *m_view;
    else
      return m_variable->data();
  }

  const VariableConceptHandle &dataHandle() const && = delete;
  const VariableConceptHandle &dataHandle() const & {
    if (m_view)
      return m_view;
    else
      return m_variable->dataHandle();
  }

  bool hasVariances() const noexcept { return m_variable->hasVariances(); }

  // Note: This return a proxy object (a VariableView) that does reference
  // members owner by *this. Therefore we can support this even for
  // temporaries and we do not need to delete the rvalue overload, unlike for
  // many other methods. The data is owned by the underlying variable so it
  // will not be deleted even if *this is a temporary and gets deleted.
  template <class T> auto values() const { return cast<T>(); }
  template <class T> auto variances() const { return castVariances<T>(); }
  template <class T> auto sparseSpan() const {
    return cast<sparse_container<T>>();
  }

  bool operator==(const Variable &other) const;
  bool operator==(const VariableConstProxy &other) const;
  bool operator!=(const Variable &other) const;
  bool operator!=(const VariableConstProxy &other) const;
  Variable operator-() const;

protected:
  friend class Variable;

  template <class T>
  const VariableView<const underlying_type_t<T>> cast() const;
  template <class T>
  const VariableView<const underlying_type_t<T>> castVariances() const;

  const Variable *m_variable;
  VariableConceptHandle m_view;
};

/** Mutable view into (a subset of) a Variable.
 *
 * By inheriting from VariableConstProxy any code that works for
 * VariableConstProxy will automatically work also for this mutable variant. */
class VariableProxy : public VariableConstProxy {
public:
  explicit VariableProxy(Variable &variable)
      : VariableConstProxy(variable), m_mutableVariable(&variable) {}
  // Note that we use the basic constructor of VariableConstProxy to avoid
  // creation of a const m_view, which would be overwritten immediately.
  VariableProxy(Variable &variable, const Dimensions &dims)
      : VariableConstProxy(variable), m_mutableVariable(&variable) {
    m_view = variable.data().reshape(dims);
  }
  VariableProxy(const VariableProxy &other) = default;
  VariableProxy(Variable &variable, const Dim dim, const scipp::index begin,
                const scipp::index end = -1)
      : VariableConstProxy(variable), m_mutableVariable(&variable) {
    m_view = variable.data().makeView(dim, begin, end);
  }
  VariableProxy(const VariableProxy &slice, const Dim dim,
                const scipp::index begin, const scipp::index end = -1)
      : VariableConstProxy(slice), m_mutableVariable(slice.m_mutableVariable) {
    m_view = slice.data().makeView(dim, begin, end);
  }

  VariableProxy slice(const Slice slice) const {
    return VariableProxy(*this, slice.dim, slice.begin, slice.end);
  }

  using VariableConstProxy::data;

  VariableConcept &data() const && = delete;
  VariableConcept &data() const & {
    if (!m_view)
      return m_mutableVariable->data();
    return *m_view;
  }

  const VariableConceptHandle &dataHandle() const && = delete;
  const VariableConceptHandle &dataHandle() const & {
    if (!m_view)
      return m_mutableVariable->dataHandle();
    return m_view;
  }

  // Note: No need to delete rvalue overloads here, see VariableConstProxy.
  template <class T> auto values() const { return cast<T>(); }
  template <class T> auto variances() const { return castVariances<T>(); }
  template <class T> auto sparseSpan() const {
    return cast<sparse_container<T>>();
  }

  // Note: We want to support things like `var(Dim::X, 0) += var2`, i.e., when
  // the left-hand-side is a temporary. This is ok since data is modified in
  // underlying Variable. However, we do not return the typical `VariableProxy
  // &` from these operations since that could reference a temporary. Due to the
  // way Python implements things like __iadd__ we must return an object
  // referencing the data though. We therefore return by value (this is not for
  // free since it involves a memory allocation but is probably relatively cheap
  // compared to other things). If the return by value turns out to be a
  // performance issue, another option is to have overloads for *this of types
  // `&` and `&&` with distinct return types (by reference in the first case, by
  // value in the second). In principle we may also change the implementation of
  // the Python exports to return `a` after calling `a += b` instead of
  // returning `a += b` but I am not sure how Pybind11 handles object lifetimes
  // (would this suffer from the same issue?).
  template <class T> VariableProxy assign(const T &other) const;
  VariableProxy operator+=(const Variable &other) const;
  VariableProxy operator+=(const VariableConstProxy &other) const;
  VariableProxy operator+=(const double value) const;
  VariableProxy operator-=(const Variable &other) const;
  VariableProxy operator-=(const VariableConstProxy &other) const;
  VariableProxy operator-=(const double value) const;
  VariableProxy operator*=(const Variable &other) const;
  VariableProxy operator*=(const VariableConstProxy &other) const;
  VariableProxy operator*=(const double value) const;
  VariableProxy operator/=(const Variable &other) const;
  VariableProxy operator/=(const VariableConstProxy &other) const;
  VariableProxy operator/=(const double value) const;

  void setUnit(const units::Unit &unit) const;

private:
  friend class Variable;
  template <class... Tags> friend class ZipView;

  template <class T> VariableView<underlying_type_t<T>> cast() const;
  template <class T> VariableView<underlying_type_t<T>> castVariances() const;

  Variable *m_mutableVariable;
};

Variable operator+(const Variable &a, const Variable &b);
Variable operator-(const Variable &a, const Variable &b);
Variable operator*(const Variable &a, const Variable &b);
Variable operator/(const Variable &a, const Variable &b);
Variable operator+(const Variable &a, const VariableConstProxy &b);
Variable operator-(const Variable &a, const VariableConstProxy &b);
Variable operator*(const Variable &a, const VariableConstProxy &b);
Variable operator/(const Variable &a, const VariableConstProxy &b);
Variable operator+(const VariableConstProxy &a, const Variable &b);
Variable operator-(const VariableConstProxy &a, const Variable &b);
Variable operator*(const VariableConstProxy &a, const Variable &b);
Variable operator/(const VariableConstProxy &a, const Variable &b);
Variable operator+(const VariableConstProxy &a, const VariableConstProxy &b);
Variable operator-(const VariableConstProxy &a, const VariableConstProxy &b);
Variable operator*(const VariableConstProxy &a, const VariableConstProxy &b);
Variable operator/(const VariableConstProxy &a, const VariableConstProxy &b);
// Note: If the left-hand-side in an addition is a VariableProxy this simply
// implicitly converts it to a Variable. A copy for the return value is required
// anyway so this is a convenient way to avoid defining more overloads.
Variable operator+(Variable a, const double b);
Variable operator-(Variable a, const double b);
Variable operator*(Variable a, const double b);
Variable operator/(Variable a, const double b);
Variable operator+(const double a, Variable b);
Variable operator-(const double a, Variable b);
Variable operator*(const double a, Variable b);
Variable operator/(const double a, Variable b);
template <class T>
Variable operator*(Variable a, const boost::units::quantity<T> &quantity) {
  return std::move(a *= quantity);
}
template <class T>
Variable operator/(Variable a, const boost::units::quantity<T> &quantity) {
  return std::move(a /= quantity);
}

std::vector<Variable> split(const Variable &var, const Dim dim,
                            const std::vector<scipp::index> &indices);
Variable concatenate(const Variable &a1, const Variable &a2, const Dim dim);
Variable rebin(const Variable &var, const Variable &oldCoord,
               const Variable &newCoord);
Variable permute(const Variable &var, const Dim dim,
                 const std::vector<scipp::index> &indices);
Variable filter(const Variable &var, const Variable &filter);
Variable sum(const Variable &var, const Dim dim);
Variable mean(const Variable &var, const Dim dim);
Variable abs(const Variable &var);
Variable norm(const Variable &var);
// TODO add to dataset and python
Variable sqrt(const Variable &var);
Variable broadcast(Variable var, const Dimensions &dims);
Variable reverse(Variable var, const Dim dim);

template <class T>
VariableView<const T> getView(const Variable &var, const Dimensions &dims);

} // namespace scipp::core

#endif // VARIABLE_H
