// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <unordered_map>

#include "scipp/dataset/dataset.h"
#include "scipp/dataset/except.h"
#include "scipp/variable/arithmetic.h"

namespace scipp::dataset {

constexpr auto unaligned_by_dim_slice = [](const auto &item,
                                           const Slice &params) {
  if (params.end() != -1)
    return false;
  const Dim dim = params.dim();
  const auto &[key, var] = item;
  if constexpr (std::is_same_v<std::decay_t<decltype(item.first)>, Dim>) {
    // TODO why duplicate?
    const bool is_dimension_coord = var.dims().contains(key);
    return var.dims().contains(dim) &&
           (is_dimension_coord ? key == dim : var.dims().inner() == dim);
  } else {
    return false;
  }
};

template <class T1, class T2> auto union_(const T1 &a, const T2 &b) {
  std::unordered_map<typename T1::key_type, typename T1::mapped_type> out;

  for (const auto [key, item] : a)
    out.emplace(key, item);

  for (const auto item : b) {
    if (const auto it = a.find(item.first); it != a.end()) {
      core::expect::equals(item, *it);
    } else
      out.emplace(item.first, item.second);
  }
  return out;
}

/// Return intersection of maps, i.e., all items with matching names that
/// have matching content.
template <class Map> auto intersection(const Map &a, const Map &b) {
  std::unordered_map<typename Map::key_type, Variable> out;
  for (const auto [key, item] : a)
    if (const auto it = b.find(key); it != b.end() && it->second == item)
      out.emplace(key, item);
  return out;
}

/// Return a copy of map-like objects such as CoordView.
template <class T> auto copy_map(const T &map) {
  std::unordered_map<typename T::key_type, typename T::mapped_type> out;
  for (const auto [key, item] : map)
    out.emplace(key, copy(item));
  return out;
}

static inline void expectAlignedCoord(const Dim coord_dim, const Variable &var,
                                      const Dim operation_dim) {
  // Coordinate is 2D, but the dimension associated with the coordinate is
  // different from that of the operation. Note we do not account for the
  // possibility that the coordinates actually align along the operation
  // dimension.
  if (var.dims().ndim() > 1 && var.dims().contains(operation_dim))
    throw except::DimensionError("Coordinate " + to_string(coord_dim) +
                                 " contains the operation dim " +
                                 to_string(operation_dim) +
                                 ", but has more than one dimension. It will "
                                 "thus not be reduced by the operation.");
}

template <bool ApplyToData, class Func, class... Args>
DataArray apply_or_copy_dim_impl(const DataArray &a, Func func, const Dim dim,
                                 Args &&... args) {
  const auto coord_apply_or_copy_dim = [&](auto &coords_, const auto &view,
                                           const bool aligned) {
    // Note the `copy` call, ensuring that the return value of the ternary
    // operator can be moved. Without `copy`, the result of `func` is always
    // copied.
    for (auto &&[d, coord] : view)
      if (coord.dims().ndim() == 0 || dim_of_coord(coord, d) != dim) {
        if (aligned)
          expectAlignedCoord(d, coord, dim);
        if constexpr (ApplyToData) {
          coords_.emplace(d, coord.dims().contains(dim)
                                 ? func(coord, dim, args...)
                                 : copy(coord));
        } else {
          coords_.emplace(d, coord);
        }
      }
  };
  std::unordered_map<Dim, Variable> coords;
  coord_apply_or_copy_dim(coords, a.coords(), true);

  std::unordered_map<Dim, Variable> attrs;
  coord_apply_or_copy_dim(attrs, a.attrs(), false);

  std::unordered_map<std::string, Variable> masks;
  for (auto &&[name, mask] : a.masks())
    if (!mask.dims().contains(dim))
      masks.emplace(name, mask);

  if constexpr (ApplyToData) {
    return DataArray(func(a.data(), dim, args...), std::move(coords),
                     std::move(masks), std::move(attrs), a.name());
  } else {
    return DataArray(func(a, dim, std::forward<Args>(args)...),
                     std::move(coords), std::move(masks), std::move(attrs),
                     a.name());
  }
}

/// Helper for creating operations that return an object with modified data with
/// a dropped dimension or different dimension extent.
///
/// Examples are mostly reduction operations such as `sum` (dropping a
/// dimension), or `resize` (altering a dimension extent). Creates new data
/// array by applying `func` to data and dropping coords/masks/attrs depending
/// on dim. The exception are multi-dimensional coords that depend on `dim`,
/// with two cases: (1) If the coord is a coord for `dim`, `func` is applied to
/// it, (2) if the coords is a coords for a dimension other than `dim`, a
/// CoordMismatchError is thrown.
template <class Func, class... Args>
DataArray apply_to_data_and_drop_dim(const DataArray &a, Func func,
                                     const Dim dim, Args &&... args) {
  return apply_or_copy_dim_impl<true>(a, func, dim,
                                      std::forward<Args>(args)...);
}

/// Helper for creating operations that return an object with a dropped
/// dimension or different dimension extent.
///
/// In contrast to `apply_to_data_and_drop_dim`, `func` is applied to the input
/// array, not just its data. This is useful for more complex operations such as
/// `histogram`, which require access to coords when computing output data.
template <class Func, class... Args>
DataArray apply_and_drop_dim(const DataArray &a, Func func, const Dim dim,
                             Args &&... args) {
  return apply_or_copy_dim_impl<false>(a, func, dim,
                                       std::forward<Args>(args)...);
}

template <class Func, class... Args>
DataArray apply_to_items(const DataArray &d, Func func, Args &&... args) {
  return func(d, std::forward<Args>(args)...);
}

template <class... Args>
bool copy_attr(const Variable &attr, const Dim dim, const Args &...) {
  return !attr.dims().contains(dim);
}
template <class... Args> bool copy_attr(const Variable &, const Args &...) {
  return true;
}

template <class Func, class... Args>
Dataset apply_to_items(const Dataset &d, Func func, Args &&... args) {
  Dataset result;
  for (const auto &data : d)
    result.setData(data.name(), func(data, std::forward<Args>(args)...));
  return result;
}

/// Copy all map items from `a` and insert them into `b`.
template <class A, class B> auto copy_items(const A &a, B &b) {
  for (const auto [key, item] : a)
    b.set(key, item);
}

/// Return a copy of map-like objects such as Coords with `func` applied to each
/// item.
template <class T, class Func> auto transform_map(const T &map, Func func) {
  std::unordered_map<typename T::key_type, typename T::mapped_type> out;
  for (const auto &[key, item] : map)
    out.emplace(key, func(item));
  return out;
}

template <class T, class Func> DataArray transform(const T &a, Func func) {
  return DataArray(func(a.data()), transform_map(a.coords(), func),
                   transform_map(a.masks(), func),
                   transform_map(a.attrs(), func), a.name());
}

void copy_metadata(const DataArray &a, DataArray &b);

// Helpers for reductions for DataArray and Dataset, which include masks.
[[nodiscard]] Variable mean(const Variable &var, const Dim dim,
                            const Masks &masks);
[[nodiscard]] Variable nanmean(const Variable &var, const Dim dim,
                               const Masks &masks);
[[nodiscard]] Variable sum(const Variable &var, const Masks &masks);
[[nodiscard]] Variable sum(const Variable &var, const Dim dim,
                           const Masks &masks);
Variable &sum(const Variable &var, const Dim dim, const Masks &masks,
              Variable &out);
[[nodiscard]] Variable nansum(const Variable &var, const Masks &masks);
[[nodiscard]] Variable nansum(const Variable &var, const Dim dim,
                              const Masks &masks);

/// Helper class for applying irreducible masks along dim.
///
/// If a mask is applied this class keeps ownership of the masked temporary.
/// `Masker` should thus be created in the scope where the masked data is
/// needed. It will be deleted once the masked goes out of scope.
class Masker {
public:
  // TODO this is not required any more with new sharing model
  Masker(const DataArray &array, const Dim dim) {
    const auto mask = irreducible_mask(array.masks(), dim);
    if (mask)
      m_masked = array.data() * ~mask;
    m_data = m_masked ? m_masked : array.data();
  }
  auto data() const noexcept { return m_data; }

private:
  Variable m_masked;
  Variable m_data;
};

} // namespace scipp::dataset
