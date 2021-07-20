// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/dataset/map_view.h"
#include "scipp/dataset/except.h"

namespace scipp::dataset {

namespace {
template <class T> void expectWritable(const T &dict) {
  if (dict.is_readonly())
    throw except::DataArrayError(
        "Read-only flag is set, cannot mutate metadata dict.");
}
} // namespace

template <class Key, class Value>
Dict<Key, Value>::Dict(const Sizes &sizes,
                       std::initializer_list<std::pair<const Key, Value>> items,
                       const bool readonly)
    : Dict(sizes, holder_type(items), readonly) {}

template <class Key, class Value>
Dict<Key, Value>::Dict(const Sizes &sizes, holder_type items,
                       const bool readonly)
    : m_sizes(sizes) {
  for (auto &&[key, value] : items)
    set(key, std::move(value));
  m_readonly = readonly;
}

template <class Key, class Value>
Dict<Key, Value>::Dict(const Dict &other)
    : Dict(other.m_sizes, other.m_items, false) {}

template <class Key, class Value>
Dict<Key, Value>::Dict(Dict &&other)
    : Dict(std::move(other.m_sizes), std::move(other.m_items),
           other.m_readonly) {}

template <class Key, class Value>
Dict<Key, Value> &Dict<Key, Value>::operator=(const Dict &other) = default;

template <class Key, class Value>
Dict<Key, Value> &Dict<Key, Value>::operator=(Dict &&other) = default;

template <class Key, class Value>
bool Dict<Key, Value>::operator==(const Dict &other) const {
  if (size() != other.size())
    return false;
  for (const auto [name, data] : *this)
    if (!other.contains(name) || data != other[name])
      return false;
  return true;
}

template <class Key, class Value>
bool Dict<Key, Value>::operator!=(const Dict &other) const {
  return !operator==(other);
}

/// Returns whether a given key is present in the view.
template <class Key, class Value>
bool Dict<Key, Value>::contains(const Key &k) const {
  return m_items.find(k) != m_items.cend();
}

/// Returns 1 or 0, depending on whether key is present in the view or not.
template <class Key, class Value>
scipp::index Dict<Key, Value>::count(const Key &k) const {
  return static_cast<scipp::index>(contains(k));
}

/// Const reference to the coordinate for given dimension.
template <class Key, class Value>
const Value &Dict<Key, Value>::operator[](const Key &key) const {
  return at(key);
}

/// Const reference to the coordinate for given dimension.
template <class Key, class Value>
const Value &Dict<Key, Value>::at(const Key &key) const {
  scipp::expect::contains(*this, key);
  return m_items.at(key);
}

/// The coordinate for given dimension.
template <class Key, class Value>
Value Dict<Key, Value>::operator[](const Key &key) {
  return std::as_const(*this).at(key);
}

/// The coordinate for given dimension.
template <class Key, class Value> Value Dict<Key, Value>::at(const Key &key) {
  return std::as_const(*this).at(key);
}

template <class Key, class Value>
void Dict<Key, Value>::setSizes(const Sizes &sizes) {
  scipp::expect::includes(sizes, m_sizes);
  m_sizes = sizes;
}

template <class Key, class Value> void Dict<Key, Value>::rebuildSizes() {
  Sizes new_sizes = m_sizes;
  for (const auto &dim : m_sizes) {
    bool erase = true;
    for (const auto &item : *this) {
      if (item.second.dims().contains(dim)) {
        erase = false;
        break;
      }
    }
    if (erase)
      new_sizes.erase(dim);
  }
  m_sizes = std::move(new_sizes);
}

template <class Key, class Value>
void Dict<Key, Value>::set(const key_type &key, mapped_type coord) {
  if (contains(key) && at(key).is_same(coord))
    return;
  expectWritable(*this);
  // Is a good definition for things that are allowed: "would be possible to
  // concat along existing dim or extra dim"?
  if (!m_sizes.includes(coord.dims()) &&
      !is_edges(m_sizes, coord.dims(), dim_of_coord(coord, key)))
    throw except::DimensionError("Cannot add coord exceeding DataArray dims");
  m_items.insert_or_assign(key, std::move(coord));
}

template <class Key, class Value>
void Dict<Key, Value>::erase(const key_type &key) {
  expectWritable(*this);
  scipp::expect::contains(*this, key);
  m_items.erase(key);
}

template <class Key, class Value>
Value Dict<Key, Value>::extract(const key_type &key) {
  auto extracted = at(key);
  erase(key);
  return extracted;
}

template <class Key, class Value>
Dict<Key, Value> Dict<Key, Value>::slice(const Slice &params) const {
  const bool readonly = true;
  return {m_sizes.slice(params), slice_map(m_sizes, m_items, params), readonly};
}

namespace {
constexpr auto unaligned_by_dim_slice = [](const auto &item,
                                           const Slice &params) {
  if (params == Slice{} || params.end() != -1)
    return false;
  const Dim dim = params.dim();
  const auto &[key, var] = item;
  return var.dims().contains(dim) && dim_of_coord(var, key) == dim;
};
}

template <class Key, class Value>
std::tuple<Dict<Key, Value>, Dict<Key, Value>>
Dict<Key, Value>::slice_coords(const Slice &params) const {
  auto coords = slice(params);
  coords.m_readonly = false;
  Dict<Key, Value> attrs(coords.sizes(), {});
  for (auto &coord : *this)
    if (unaligned_by_dim_slice(coord, params))
      attrs.set(coord.first, coords.extract(coord.first));
  coords.m_readonly = true;
  return {std::move(coords), std::move(attrs)};
}

template <class Key, class Value>
void Dict<Key, Value>::validateSlice(const Slice s, const Dict &dict) const {
  using core::to_string;
  using units::to_string;
  for (const auto &[key, item] : dict) {
    const auto it = find(key);
    if (it == end()) {
      throw except::NotFoundError("Cannot insert new meta data '" +
                                  to_string(key) + "' via a slice.");
    } else if ((it->second.is_readonly() ||
                !it->second.dims().contains(s.dim())) &&
               (it->second.dims().contains(s.dim()) ? it->second.slice(s)
                                                    : it->second) != item) {
      throw except::DimensionError("Cannot update meta data '" +
                                   to_string(key) +
                                   "' via slice since it is implicitly "
                                   "broadcast along the slice dimension '" +
                                   to_string(s.dim()) + "'.");
    }
  }
}

template <class Key, class Value>
Dict<Key, Value> &Dict<Key, Value>::setSlice(const Slice s, const Dict &dict) {
  validateSlice(s, dict);
  for (const auto &[key, item] : dict) {
    const auto it = find(key);
    if (it != end() && !it->second.is_readonly() &&
        it->second.dims().contains(s.dim()))
      it->second.setSlice(s, item);
  }
  return *this;
}

template <class Key, class Value>
void Dict<Key, Value>::rename(const Dim from, const Dim to) {
  m_sizes.replace_key(from, to);
  for (auto &item : m_items)
    if (item.second.dims().contains(from))
      item.second.rename(from, to);
}

/// Return true if the dict is readonly. Does not imply that items are readonly.
template <class Key, class Value>
bool Dict<Key, Value>::is_readonly() const noexcept {
  return m_readonly;
}

template <class Key, class Value>
Dict<Key, Value> Dict<Key, Value>::as_const() const {
  holder_type items;
  std::transform(m_items.begin(), m_items.end(),
                 std::inserter(items, items.end()), [](const auto &item) {
                   return std::pair(item.first, item.second.as_const());
                 });
  const bool readonly = true;
  return {sizes(), std::move(items), readonly};
}

template <class Key, class Value>
Dict<Key, Value> Dict<Key, Value>::merge_from(const Dict &other) const {
  using core::to_string;
  using units::to_string;
  auto out(*this);
  out.m_readonly = false;
  for (const auto &[key, value] : other) {
    if (out.contains(key))
      throw except::DataArrayError(
          "Coord '" + to_string(key) +
          "' shadows attr of the same name. Remove the attr if you are slicing "
          "an array or use the `coords` and `attrs` properties instead of "
          "`meta`.");
    out.set(key, value);
  }
  out.m_readonly = m_readonly;
  return out;
}

template <class Key, class Value>
bool Dict<Key, Value>::item_applies_to(const Key &key,
                                       const Dimensions &dims) const {
  const auto &val = m_items.at(key);
  return dims.includes(val.dims()) ||
         (!sizes().includes(val.dims()) &&
          is_edges(Sizes(dims), val.dims(), dim_of_coord(val, key)));
}

template class SCIPP_DATASET_EXPORT Dict<Dim, Variable>;
template class SCIPP_DATASET_EXPORT Dict<std::string, Variable>;

} // namespace scipp::dataset
