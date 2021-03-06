// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Neil Vaytet
#pragma once

#include "scipp-core_export.h"
#include "scipp/common/index_composition.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/strides.h"

namespace scipp::core {

class SCIPP_CORE_EXPORT ViewIndex {
public:
  ViewIndex(const Dimensions &target_dimensions, const Strides &strides);

  constexpr void increment_outer() noexcept {
    scipp::index d = 0;
    while ((m_coord[d] == m_shape[d]) && (d < NDIM_MAX - 1)) {
      m_memory_index += m_delta[d + 1];
      ++m_coord[d + 1];
      m_coord[d] = 0;
      ++d;
    }
  }
  constexpr void increment() noexcept {
    m_memory_index += m_delta[0];
    ++m_coord[0];
    if (m_coord[0] == m_shape[0])
      increment_outer();
    ++m_view_index;
  }

  constexpr void set_index(const scipp::index index) noexcept {
    m_view_index = index;
    extract_indices(index, m_ndim, m_shape, m_coord);
    m_memory_index = flat_index_from_strides(
        m_strides.begin(), m_strides.end(m_ndim), m_coord.begin());
  }

  void set_to_end() noexcept {
    m_view_index = 0;
    for (scipp::index dim = 0; dim < m_ndim - 1; ++dim) {
      m_view_index *= m_shape[dim];
    }
    std::fill(m_coord.begin(), m_coord.begin() + std::max(m_ndim - 1, 0), 0);
    m_coord[m_ndim] = m_shape[m_ndim];
    m_memory_index = m_coord[m_ndim] * m_strides[m_ndim];
  }

  [[nodiscard]] constexpr scipp::index get() const noexcept {
    return m_memory_index;
  }
  [[nodiscard]] constexpr scipp::index index() const noexcept {
    return m_view_index;
  }

  constexpr bool operator==(const ViewIndex &other) const noexcept {
    return m_view_index == other.m_view_index;
  }
  constexpr bool operator!=(const ViewIndex &other) const noexcept {
    return m_view_index != other.m_view_index;
  }

private:
  // NOTE:
  // We investigated different containers for the m_delta, m_coord & m_extent
  // arrays, and their impact on performance when iterating over a variable
  // view.
  // Using std::array or C-style arrays give good performance (7.5 Gb/s) as long
  // as a range based loop is used:
  //
  //   for ( const auto x : view ) {
  //
  // If a loop which explicitly accesses the begin() and end() of the container
  // is used, e.g.
  //
  //   for ( auto it = view.begin(); it != view.end(); ++it ) {
  //
  // then the results differ widely.
  // - using std::array is 80x slower than above, at ~90 Mb/s
  // - using C-style arrays is 20x slower than above, at ~330 Mb/s
  //
  // We can recover the maximum performance by storing the view.end() in a
  // variable, e.g.
  //
  //   auto iend = view.end();
  //   for ( auto it = view.begin(); it != iend; ++it ) {
  //
  // for both std::array and C-style arrays.
  //
  // Finally, when using C-style arrays, we get a compilation warning from L37
  //
  //   m_delta[d] -= m_delta[d2];
  //
  // with the GCC compiler:
  //
  //  warning: array subscript is above array bounds [-Warray-bounds]
  //
  // which disappears when switching to std::array. This warning is not given
  // by the CLANG compiler, and is not fully understood as d2 is always less
  // than d and should never overflow the array bounds.
  // We decided to go with std::array as our final choice to avoid the warning,
  // as the performance is identical to C-style arrays, as long as range based
  // loops are used.

  /// Index into memory.
  scipp::index m_memory_index{0};
  /// Steps to advance one element.
  std::array<scipp::index, NDIM_MAX> m_delta = {};
  /// Multi-dimensional index in iteration dimensions.
  std::array<scipp::index, NDIM_MAX> m_coord = {};
  /// Shape in iteration dimensions.
  std::array<scipp::index, NDIM_MAX> m_shape = {};
  /// Strides in memory.
  Strides m_strides;
  /// Index in iteration dimensions.
  scipp::index m_view_index{0};
  /// Number of dimensions.
  int32_t m_ndim;
};
/*
 * TODO document implementation
 * - arrays have fasted dimension first, i.e. opposite to `Dimensions`
 * - explain attributes: role, exact behavior, special values (e.g. at the end)
 */

} // namespace scipp::core
