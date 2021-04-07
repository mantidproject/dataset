// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <functional>
#include <iosfwd>
#include <map>
#include <string>
#include <unordered_map>

#include <boost/iterator/transform_iterator.hpp>

#include "scipp/dataset/data_array.h"
#include "scipp/dataset/map_view.h"
#include "scipp/variable/variable.h"

namespace scipp::dataset {

enum CoordCategory { Aligned = 1, Unaligned = 2, All = 3 };

namespace detail {

/// Helper for creating iterators of Dataset.
template <class D> struct make_item {
  D *dataset;
  template <class T> auto operator()(T &item) const {
    return item.second.view_with_coords(dataset->coords(), item.first);
  }
};
template <class D> make_item(D *) -> make_item<D>;

} // namespace detail

/// Collection of data arrays.
class SCIPP_DATASET_EXPORT Dataset {
public:
  using key_type = std::string;
  using mapped_type = DataArray;
  using value_type = std::pair<const std::string &, DataArray>;

  Dataset() = default;
  explicit Dataset(const DataArray &data);

  template <class DataMap = const std::map<std::string, DataArray> &,
            class CoordMap = std::unordered_map<Dim, Variable>>
  explicit Dataset(DataMap data,
                   CoordMap coords = std::unordered_map<Dim, Variable>{}) {
    for (auto &&[dim, coord] : coords)
      setCoord(dim, std::move(coord));
    if constexpr (std::is_base_of_v<Dataset, std::decay_t<DataMap>>)
      for (auto &&item : data)
        setData(item.name(), item);
    else
      for (auto &&[name, item] : data)
        setData(std::string(name), std::move(item));
  }

  /// Return the number of data items in the dataset.
  ///
  /// This does not include coordinates or attributes, but only all named
  /// entities (which can consist of various combinations of values, variances,
  /// and events coordinates).
  index size() const noexcept { return scipp::size(m_data); }
  /// Return true if there are 0 data items in the dataset.
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  void clear();

  const Coords &coords() const noexcept;
  Coords &coords() noexcept;

  const Coords &meta() const noexcept;
  Coords &meta() noexcept;

  bool contains(const std::string &name) const noexcept;

  void erase(const std::string &name);
  [[nodiscard]] DataArray extract(const std::string &name);

  auto find() const && = delete;
  auto find() && = delete;
  auto find(const std::string &name) &noexcept {
    return boost::make_transform_iterator(m_data.find(name),
                                          detail::make_item{this});
  }
  auto find(const std::string &name) const &noexcept {
    return boost::make_transform_iterator(m_data.find(name),
                                          detail::make_item{this});
  }

  DataArray operator[](const std::string &name) const;

  auto begin() const && = delete;
  auto begin() && = delete;
  /// Return const iterator to the beginning of all data items.
  auto begin() const &noexcept {
    return boost::make_transform_iterator(m_data.begin(),
                                          detail::make_item{this});
  }
  /// Return iterator to the beginning of all data items.
  auto begin() &noexcept {
    return boost::make_transform_iterator(m_data.begin(),
                                          detail::make_item{this});
  }
  auto end() const && = delete;
  auto end() && = delete;
  /// Return const iterator to the end of all data items.
  auto end() const &noexcept {
    return boost::make_transform_iterator(m_data.end(),
                                          detail::make_item{this});
  }

  /// Return iterator to the end of all data items.
  auto end() &noexcept {
    return boost::make_transform_iterator(m_data.end(),
                                          detail::make_item{this});
  }

  auto items_begin() const && = delete;
  auto items_begin() && = delete;
  auto items_begin() const &noexcept {
    return boost::make_transform_iterator(begin(), detail::make_key_value);
  }
  auto items_begin() &noexcept {
    return boost::make_transform_iterator(begin(), detail::make_key_value);
  }
  auto items_end() const && = delete;
  auto items_end() && = delete;
  auto items_end() const &noexcept {
    return boost::make_transform_iterator(end(), detail::make_key_value);
  }

  auto items_end() &noexcept {
    return boost::make_transform_iterator(end(), detail::make_key_value);
  }

  auto keys_begin() const && = delete;
  auto keys_begin() && = delete;
  auto keys_begin() const &noexcept {
    return boost::make_transform_iterator(m_data.begin(), detail::make_key);
  }
  auto keys_begin() &noexcept {
    return boost::make_transform_iterator(m_data.begin(), detail::make_key);
  }
  auto keys_end() const && = delete;
  auto keys_end() && = delete;
  auto keys_end() const &noexcept {
    return boost::make_transform_iterator(m_data.end(), detail::make_key);
  }

  auto keys_end() &noexcept {
    return boost::make_transform_iterator(m_data.end(), detail::make_key);
  }

  void setCoord(const Dim dim, Variable coord);
  void setData(const std::string &name, Variable data,
               const AttrPolicy attrPolicy = AttrPolicy::Drop);
  void setData(const std::string &name, const DataArray &data);

  Dataset slice(const Slice s) const;

  void rename(const Dim from, const Dim to);

  bool operator==(const Dataset &other) const;
  bool operator!=(const Dataset &other) const;

  Dataset &operator+=(const DataArray &other);
  Dataset &operator-=(const DataArray &other);
  Dataset &operator*=(const DataArray &other);
  Dataset &operator/=(const DataArray &other);
  Dataset &operator+=(const Variable &other);
  Dataset &operator-=(const Variable &other);
  Dataset &operator*=(const Variable &other);
  Dataset &operator/=(const Variable &other);
  Dataset &operator+=(const Dataset &other);
  Dataset &operator-=(const Dataset &other);
  Dataset &operator*=(const Dataset &other);
  Dataset &operator/=(const Dataset &other);

  // TODO dims() required for generic code. Need proper equivalent to class
  // Dimensions that does not imply dimension order.
  std::unordered_map<Dim, scipp::index> dimensions() const;
  std::unordered_map<Dim, scipp::index> dims() const { return dimensions(); }

private:
  template <class T>
  friend Coords make_coords(const T &view, const CoordCategory category,
                            const bool is_item);

  void setExtent(const Dim dim, const scipp::index extent, const bool isCoord);
  void setDims(const Dimensions &dims, const Dim coordDim = Dim::Invalid);
  void rebuildDims();

  Coords m_coords; // aligned coords
  std::unordered_map<std::string, DataArray> m_data;
};

[[nodiscard]] SCIPP_DATASET_EXPORT Dataset
copy(const Dataset &dataset, const AttrPolicy attrPolicy = AttrPolicy::Keep);
[[nodiscard]] SCIPP_DATASET_EXPORT Dataset deepcopy(
    const Dataset &dataset, const AttrPolicy attrPolicy = AttrPolicy::Keep);

[[maybe_unused]] SCIPP_DATASET_EXPORT DataArray &
copy(const DataArray &array, DataArray &out,
     const AttrPolicy attrPolicy = AttrPolicy::Keep);
[[maybe_unused]] SCIPP_DATASET_EXPORT DataArray
copy(const DataArray &array, DataArray &&out,
     const AttrPolicy attrPolicy = AttrPolicy::Keep);
[[maybe_unused]] SCIPP_DATASET_EXPORT Dataset &
copy(const Dataset &dataset, Dataset &out,
     const AttrPolicy attrPolicy = AttrPolicy::Keep);
[[maybe_unused]] SCIPP_DATASET_EXPORT Dataset
copy(const Dataset &dataset, Dataset &&out,
     const AttrPolicy attrPolicy = AttrPolicy::Keep);

SCIPP_DATASET_EXPORT DataArray operator-(const DataArray &a);

SCIPP_DATASET_EXPORT Dataset operator+(const Dataset &lhs, const Dataset &rhs);
SCIPP_DATASET_EXPORT Dataset operator+(const Dataset &lhs,
                                       const DataArray &rhs);
SCIPP_DATASET_EXPORT Dataset operator+(const DataArray &lhs,
                                       const Dataset &rhs);
SCIPP_DATASET_EXPORT Dataset operator+(const Dataset &lhs, const Variable &rhs);
SCIPP_DATASET_EXPORT Dataset operator+(const Variable &lhs, const Dataset &rhs);

SCIPP_DATASET_EXPORT Dataset operator-(const Dataset &lhs, const Dataset &rhs);
SCIPP_DATASET_EXPORT Dataset operator-(const Dataset &lhs,
                                       const DataArray &rhs);
SCIPP_DATASET_EXPORT Dataset operator-(const DataArray &lhs,
                                       const Dataset &rhs);
SCIPP_DATASET_EXPORT Dataset operator-(const Dataset &lhs, const Variable &rhs);
SCIPP_DATASET_EXPORT Dataset operator-(const Variable &lhs, const Dataset &rhs);

SCIPP_DATASET_EXPORT Dataset operator*(const Dataset &lhs, const Dataset &rhs);
SCIPP_DATASET_EXPORT Dataset operator*(const Dataset &lhs,
                                       const DataArray &rhs);
SCIPP_DATASET_EXPORT Dataset operator*(const DataArray &lhs,
                                       const Dataset &rhs);
SCIPP_DATASET_EXPORT Dataset operator*(const Dataset &lhs, const Variable &rhs);
SCIPP_DATASET_EXPORT Dataset operator*(const Variable &lhs, const Dataset &rhs);

SCIPP_DATASET_EXPORT Dataset operator/(const Dataset &lhs, const Dataset &rhs);
SCIPP_DATASET_EXPORT Dataset operator/(const Dataset &lhs,
                                       const DataArray &rhs);
SCIPP_DATASET_EXPORT Dataset operator/(const DataArray &lhs,
                                       const Dataset &rhs);
SCIPP_DATASET_EXPORT Dataset operator/(const Dataset &lhs, const Variable &rhs);
SCIPP_DATASET_EXPORT Dataset operator/(const Variable &lhs, const Dataset &rhs);

SCIPP_DATASET_EXPORT Dataset merge(const Dataset &a, const Dataset &b);

/// Union the masks of the two proxies.
/// If any of the masks repeat they are OR'ed.
/// The result is stored in a new map
SCIPP_DATASET_EXPORT
std::unordered_map<typename Masks::key_type, typename Masks::mapped_type>
union_or(const Masks &currentMasks, const Masks &otherMasks);

/// Union the masks of the two proxies.
/// If any of the masks repeat they are OR'ed.
/// The result is stored in the first view.
SCIPP_DATASET_EXPORT void union_or_in_place(Masks &currentMasks,
                                            const Masks &otherMasks);

} // namespace scipp::dataset

namespace scipp::core {
template <> inline constexpr DType dtype<dataset::DataArray>{2000};
template <> inline constexpr DType dtype<dataset::Dataset>{2001};
template <> inline constexpr DType dtype<bucket<dataset::DataArray>>{2002};
template <> inline constexpr DType dtype<bucket<dataset::Dataset>>{2003};
} // namespace scipp::core

namespace scipp {
using dataset::Dataset;
} // namespace scipp

#include "scipp/dataset/arithmetic.h"
