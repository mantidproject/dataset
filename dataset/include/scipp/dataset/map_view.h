// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <boost/container/small_vector.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "scipp-dataset_export.h"
#include "scipp/core/sizes.h"
#include "scipp/core/slice.h"
#include "scipp/dataset/map_view_forward.h"
#include "scipp/units/dim.h"
#include "scipp/units/unit.h"
#include "scipp/variable/logical.h"
#include "scipp/variable/variable.h"

namespace scipp::dataset {

namespace detail {
using slice_list =
    boost::container::small_vector<std::pair<Slice, scipp::index>, 2>;

template <class T> void do_make_slice(T &slice, const slice_list &slices) {
  for (const auto &[params, extent] : slices) {
    if (slice.dims().contains(params.dim())) {
      if (slice.dims()[params.dim()] == extent) {
        slice = slice.slice(params);
      } else {
        const auto end = params.end() == -1 ? params.begin() + 2
                                            : params.begin() == params.end()
                                                  ? params.end()
                                                  : params.end() + 1;
        slice = slice.slice(Slice{params.dim(), params.begin(), end});
      }
    }
  }
}

template <class Var> auto makeSlice(Var &var, const slice_list &slices) {
  std::conditional_t<std::is_const_v<Var>, typename Var::const_view_type,
                     typename Var::view_type>
      slice(var);
  do_make_slice(slice, slices);
  return slice;
}

static constexpr auto make_key_value = [](auto &&view) {
  using In = decltype(view);
  using View =
      std::conditional_t<std::is_rvalue_reference_v<In>, std::decay_t<In>, In>;
  return std::pair<const std::string &, View>(
      view.name(), std::forward<decltype(view)>(view));
};

static constexpr auto make_key = [](auto &&view) -> decltype(auto) {
    return view.first;
};

static constexpr auto make_value = [](auto &&view) -> decltype(auto) {
  return view.second;
};

} // namespace detail

/// Return the dimension for given coord.
/// @param var Coordinate variable
/// @param key Key of the coordinate in a coord dict
///
/// For dimension-coords, this is the same as the key, for non-dimension-coords
/// (labels) we adopt the convention that they are "label" their inner
/// dimension. Returns Dim::Invalid for 0-D var.
template <class T, class Key> Dim dim_of_coord(const T &var, const Key &key) {
  if (var.dims().ndim() == 0)
    return Dim::Invalid;
  if constexpr (std::is_same_v<Key, Dim>) {
    const bool is_dimension_coord = var.dims().contains(key);
    return is_dimension_coord ? key : var.dims().inner();
  } else
    return var.dims().inner();
}

template <class T>
auto slice_map(const Sizes &sizes, const T &map, const Slice &params) {
  core::expect::validSlice(sizes, params);
  T out;
  for (const auto &[key, value] : map) {
    if (value.dims().contains(params.dim())) {
      if (value.dims()[params.dim()] == sizes[params.dim()]) {
        out[key] = value.slice(params);
      } else { // bin edge
        const auto end = params.end() == -1 ? params.begin() + 2
                                            : params.begin() == params.end()
                                                  ? params.end()
                                                  : params.end() + 1;
        out[key] = value.slice(Slice{params.dim(), params.begin(), end});
      }
    } else {
      out[key] = value;
    }
  }
  return out;
}

/// Common functionality for other const-view classes.
template <class Key, class Value> class Dict {
public:
  using key_type = Key;
  using mapped_type = Value;
  using holder_type = std::unordered_map<key_type, mapped_type>;

  Dict() = default;
  Dict(const Sizes &sizes,
       std::initializer_list<std::pair<const Key, Value>> items)
      : Dict(sizes, holder_type(items)) {}
  Dict(const Sizes &sizes, holder_type items) : m_sizes(sizes) {
    for (auto &&[key, value] : items)
      set(key, std::move(value));
  }

  /// Return the number of coordinates in the view.
  index size() const noexcept { return scipp::size(m_items); }
  /// Return true if there are 0 coordinates in the view.
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  bool contains(const Key &k) const;
  scipp::index count(const Key &k) const;

  const mapped_type &operator[](const Key &key) const;
  const mapped_type &at(const Key &key) const;
  // Note that the non-const versions return by value, to avoid breakage of
  // invariants.
  mapped_type operator[](const Key &key);
  mapped_type at(const Key &key);

  auto find(const Key &k) const noexcept { return m_items.find(k); }
  auto find(const Key &k) noexcept { return m_items.find(k); }

  /// Return const iterator to the beginning of all items.
  auto begin() const noexcept { return m_items.begin(); }
  auto begin() noexcept { return m_items.begin(); }
  /// Return const iterator to the end of all items.
  auto end() const noexcept { return m_items.end(); }
  auto end() noexcept { return m_items.end(); }

  auto items_begin() const && = delete;
  /// Return const iterator to the beginning of all items.
  auto items_begin() const &noexcept { return begin(); }
  auto items_end() const && = delete;
  /// Return const iterator to the end of all items.
  auto items_end() const &noexcept { return end(); }

  auto keys_begin() const && = delete;
  /// Return const iterator to the beginning of all keys.
  auto keys_begin() const &noexcept {
    return boost::make_transform_iterator(begin(), detail::make_key);
  }
  auto keys_end() const && = delete;
  /// Return const iterator to the end of all keys.
  auto keys_end() const &noexcept {
    return boost::make_transform_iterator(end(), detail::make_key);
  }

  auto values_begin() const && = delete;
  /// Return const iterator to the beginning of all values.
  auto values_begin() const &noexcept {
    return boost::make_transform_iterator(begin(), detail::make_value);
  }
  auto values_end() const && = delete;
  /// Return const iterator to the end of all values.
  auto values_end() const &noexcept {
    return boost::make_transform_iterator(end(), detail::make_value);
  }

  bool operator==(const Dict &other) const;
  bool operator!=(const Dict &other) const;

  const Sizes &sizes() const noexcept { return m_sizes; }
  // TODO users should not have access to this, use only internally in Dataset
  Sizes &sizes() noexcept { return m_sizes; }
  const auto &items() const noexcept { return m_items; }
  auto &items() noexcept { return m_items; }

  void set(const key_type &key, mapped_type coord);
  void erase(const key_type &key);
  mapped_type extract(const key_type &key);

  Dict slice(const Slice &params) const;

protected:
  Sizes m_sizes;
  holder_type m_items;
};

/// Returns the union of all masks with irreducible dimension `dim`.
///
/// Irreducible means that a reduction operation must apply these masks since
/// depend on the reduction dimension. Returns an invalid (empty) variable if
/// there is no irreducible mask.
template <class Masks>
[[nodiscard]] Variable irreducible_mask(const Masks &masks, const Dim dim) {
  Variable union_;
  for (const auto &mask : masks)
    if (mask.second.dims().contains(dim))
      union_ = union_ ? union_ | mask.second : Variable(mask.second);
  return union_;
}

SCIPP_DATASET_EXPORT Variable masks_merge_if_contained(const Masks &masks,
                                                       const Dimensions &dims);

} // namespace scipp::dataset
