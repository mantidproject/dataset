// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once
#include <algorithm>

#include "scipp/core/bucket_array_view.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/except.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/cumulative.h"
#include "scipp/variable/data_model.h"
#include "scipp/variable/except.h"
#include "scipp/variable/reduction.h"
#include "scipp/variable/util.h"

namespace scipp::variable {

template <class Indices> class BinModelBase : public VariableConcept {
public:
  BinModelBase(const VariableConceptHandle &indices, const Dim dim)
      : VariableConcept(units::one), m_indices(indices), m_dim(dim) {}

  scipp::index size() const override { return indices()->size(); }

  bool hasVariances() const noexcept override { return false; }
  void setVariances(const Variable &) override {
    throw except::VariancesError("This data type cannot have variances.");
  }
  const Indices &bin_indices() const override { return indices(); }

  const auto &indices() const { return m_indices; }
  auto &indices() { return m_indices; }
  Dim bin_dim() const noexcept { return m_dim; }

private:
  Indices m_indices;
  Dim m_dim;
};

namespace {
template <class T> auto clone_impl(const DataModel<bucket<T>> &model) {
  return std::make_unique<DataModel<bucket<T>>>(
      model.indices()->clone(), model.bin_dim(), copy(model.buffer()));
}
} // namespace

/// Specialization of DataModel for "binned" data. T could be Variable,
/// DataArray, or Dataset.
///
/// A bin in this context is defined as an element of a variable mapping to a
/// range of data, such as a slice of a DataArray.
template <class T>
class DataModel<bucket<T>> : public BinModelBase<VariableConceptHandle> {
  using Indices = VariableConceptHandle;

public:
  using value_type = bucket<T>;
  using range_type = typename bucket<T>::range_type;

  DataModel(const VariableConceptHandle &indices, const Dim dim, T buffer)
      : BinModelBase<Indices>(validated_indices(indices, dim, buffer), dim),
        m_buffer(std::move(buffer)) {}

  [[nodiscard]] VariableConceptHandle clone() const override {
    return clone_impl(*this);
  }

  bool operator==(const DataModel &other) const noexcept {
    const auto &i1 = requireT<const DataModel<range_type>>(*indices());
    const auto &i2 = requireT<const DataModel<range_type>>(*other.indices());
    return equals_impl(i1.values(), i2.values()) &&
           this->bin_dim() == other.bin_dim() && m_buffer == other.m_buffer;
  }
  bool operator!=(const DataModel &other) const noexcept {
    return !(*this == other);
  }

  [[nodiscard]] VariableConceptHandle
  makeDefaultFromParent(const scipp::index size) const override {
    return std::make_unique<DataModel>(
        makeVariable<range_type>(Dims{Dim::X}, Shape{size}).data_handle(),
        this->bin_dim(), T{m_buffer.slice({this->bin_dim(), 0, 0})});
  }

  [[nodiscard]] VariableConceptHandle
  makeDefaultFromParent(const Variable &shape) const override {
    const auto end = cumsum(shape);
    const auto begin = end - shape;
    const auto size = end.dims().volume() > 0
                          ? end.values<scipp::index>().as_span().back()
                          : 0;
    return std::make_unique<DataModel>(
        zip(begin, begin).data_handle(), this->bin_dim(),
        resize_default_init(m_buffer, this->bin_dim(), size));
  }

  static DType static_dtype() noexcept { return scipp::dtype<bucket<T>>; }
  [[nodiscard]] DType dtype() const noexcept override {
    return scipp::dtype<bucket<T>>;
  }

  [[nodiscard]] bool equals(const Variable &a,
                            const Variable &b) const override;
  void copy(const Variable &src, Variable &dest) const override;
  void copy(const Variable &src, Variable &&dest) const override;
  void assign(const VariableConcept &other) override;

  // TODO Should the mutable version return a view to prevent risk of clients
  // breaking invariants of variable?
  const T &buffer() const noexcept { return m_buffer; }
  T &buffer() noexcept { return m_buffer; }

  ElementArrayView<bucket<T>> values(const core::ElementArrayViewParams &base) {
    return {index_values(base), this->bin_dim(), m_buffer};
  }
  ElementArrayView<const bucket<T>>
  values(const core::ElementArrayViewParams &base) const {
    return {index_values(base), this->bin_dim(), m_buffer};
  }

  [[nodiscard]] scipp::index dtype_size() const override {
    return sizeof(range_type);
  }

private:
  static auto validated_indices(const VariableConceptHandle &indices,
                                [[maybe_unused]] const Dim dim,
                                [[maybe_unused]] const T &buffer) {
    auto copy = requireT<const DataModel<range_type>>(*indices);
    const auto vals = copy.values();
    std::sort(vals.begin(), vals.end());
    if ((!vals.empty() && (vals.begin()->first < 0)) ||
        (!vals.empty() && ((vals.end() - 1)->second > buffer.dims()[dim])))
      throw except::SliceError("Bin indices out of range");
    if (std::adjacent_find(vals.begin(), vals.end(),
                           [](const auto a, const auto b) {
                             return a.second > b.first;
                           }) != vals.end())
      throw except::SliceError("Overlapping bin indices are not allowed.");
    if (std::find_if(vals.begin(), vals.end(), [](const auto x) {
          return x.first > x.second;
        }) != vals.end())
      throw except::SliceError(
          "Bin begin index must be less or equal to its end index.");
    // Sharing indices
    // TODO Move validation out of this class so we can avoid copies and
    // duplicate validation, in particular for bins_view?
    return indices;
  }

  auto index_values(const core::ElementArrayViewParams &base) const {
    return requireT<const DataModel<range_type>>(*this->indices()).values(base);
  }
  T m_buffer;
};

template <class T>
bool DataModel<bucket<T>>::equals(const Variable &a, const Variable &b) const {
  if (a.unit() != b.unit())
    return false;
  if (a.dims() != b.dims())
    return false;
  if (a.dtype() != b.dtype())
    return false;
  if (a.hasVariances() != b.hasVariances())
    return false;
  if (a.dims().volume() == 0 && a.dims() == b.dims())
    return true;
  // TODO This implementation is slow since it creates a view for every bucket.
  return equals_impl(a.values<bucket<T>>(), b.values<bucket<T>>());
}

template <class T>
void DataModel<bucket<T>>::copy(const Variable &src, Variable &dest) const {
  const auto &[indices0, dim0, buffer0] = src.constituents<bucket<T>>();
  auto &&[indices1, dim1, buffer1] = dest.constituents<bucket<T>>();
  static_cast<void>(dim1);
  copy_slices(buffer0, buffer1, dim0, indices0, indices1);
}
template <class T>
void DataModel<bucket<T>>::copy(const Variable &src, Variable &&dest) const {
  copy(src, dest);
}

template <class T>
void DataModel<bucket<T>>::assign(const VariableConcept &other) {
  *this = requireT<const DataModel<bucket<T>>>(other);
}

} // namespace scipp::variable
