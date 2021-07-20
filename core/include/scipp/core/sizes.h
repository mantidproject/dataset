// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include "scipp-core_export.h"
#include "scipp/common/index.h"
#include "scipp/common/span.h"
#include "scipp/core/slice.h"
#include "scipp/units/dim.h"

namespace scipp::core {

constexpr int32_t NDIM_MAX = 6;

class Dimensions;

/// Small (fixed maximum size) and stable (preserving key order) map
template <class Key, class Value, int16_t Capacity>
class SCIPP_CORE_EXPORT small_stable_map {
public:
  static constexpr auto capacity = Capacity;

  small_stable_map() = default;

  bool operator==(const small_stable_map &other) const noexcept;
  bool operator!=(const small_stable_map &other) const noexcept;

  auto begin() const noexcept { return m_keys.begin(); }
  auto end() const noexcept { return m_keys.begin() + size(); }
  auto rbegin() const noexcept { return std::reverse_iterator(end()); }
  auto rend() const noexcept { return std::reverse_iterator(begin()); }
  typename std::array<Key, Capacity>::const_iterator find(const Key &key) const;
  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
  constexpr scipp::index size() const noexcept { return m_size; }
  bool contains(const Key &key) const;
  scipp::index index(const Key &key) const;
  const Value &operator[](const Key &key) const;
  const Value &at(const Key &key) const;
  void assign(const Key &key, const Value &value);
  void insert_left(const Key &key, const Value &value);
  void insert_right(const Key &key, const Value &value);
  void erase(const Key &key);
  void clear() noexcept;
  void replace_key(const Key &from, const Key &to);
  constexpr scipp::span<const Key> keys() const &noexcept {
    return {m_keys.data(), static_cast<size_t>(size())};
  }
  constexpr scipp::span<const Value> values() const &noexcept {
    return {m_values.data(), static_cast<size_t>(size())};
  }

private:
  int16_t m_size{0};
  std::array<Key, Capacity> m_keys{};
  std::array<Value, Capacity> m_values{};
};

/// Similar to class Dimensions but without implied ordering
class SCIPP_CORE_EXPORT Sizes
    : public small_stable_map<Dim, scipp::index, NDIM_MAX> {
private:
  using base = small_stable_map<Dim, scipp::index, NDIM_MAX>;

protected:
  using base::assign;
  using base::insert_left;
  using base::insert_right;

public:
  Sizes() = default;

  void set(const Dim dim, const scipp::index size);
  void resize(const Dim dim, const scipp::index size);
  bool includes(const Sizes &sizes) const;
  Sizes slice(const Slice &params) const;

  /// Return the labels of the space defined by *this.
  constexpr auto labels() const &noexcept { return keys(); }
  /// Return the shape of the space defined by *this.
  constexpr auto sizes() const &noexcept { return values(); }
};

[[nodiscard]] SCIPP_CORE_EXPORT Sizes concatenate(const Sizes &a,
                                                  const Sizes &b,
                                                  const Dim dim);

[[nodiscard]] SCIPP_CORE_EXPORT Sizes merge(const Sizes &a, const Sizes &b);

SCIPP_CORE_EXPORT bool is_edges(const Sizes &sizes, const Sizes &dataSizes,
                                const Dim dim);

SCIPP_CORE_EXPORT std::string to_string(const Sizes &sizes);

} // namespace scipp::core

namespace scipp {
using core::Sizes;
}
