// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Jan-Lukas Wynen
#pragma once

#include "scipp-core_export.h"
#include "scipp/common/index_composition.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/strides.h"

/*
 * See page on Multi-Dimensional Indexing in developer documentation
 * for an explanation of the implementation.
 */

namespace scipp::core {

/// A flat index into a multi-dimensional view.
class SCIPP_CORE_EXPORT ViewIndex {
public:
  ViewIndex(const Dimensions &target_dimensions, const Strides &strides);

  constexpr void increment_outer() noexcept {
    for (scipp::index d = 0; (m_coord[d] == m_shape[d]) && (d < NDIM_MAX - 1);
         ++d) {
      m_memory_index += m_delta[d + 1];
      ++m_coord[d + 1];
      m_coord[d] = 0;
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
    extract_indices(index, m_shape.begin(), m_shape.begin() + m_ndim,
                    m_coord.begin());
    m_memory_index = flat_index_from_strides(
        m_strides.begin(), m_strides.end(m_ndim), m_coord.begin());
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
  /// Index into memory.
  scipp::index m_memory_index{0};
  /// Index in iteration dimensions.
  scipp::index m_view_index{0};
  /// Steps in memory to advance one element.
  std::array<scipp::index, NDIM_MAX> m_delta = {};
  /// Multi-dimensional index in iteration dimensions.
  std::array<scipp::index, NDIM_MAX> m_coord = {};
  /// Shape in iteration dimensions.
  std::array<scipp::index, NDIM_MAX> m_shape = {};
  /// Strides in memory.
  Strides m_strides;
  /// Number of dimensions.
  int32_t m_ndim;
};
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

} // namespace scipp::core
