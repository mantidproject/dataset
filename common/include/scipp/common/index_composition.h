// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Jan-Lukas Wynen
#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include "scipp/common/index.h"

namespace scipp {
/// Compute a flat index from strides and a multi-dimensional index.
///
/// @return sum_{i=0}^{ndim} ( strides[i] * indices[i] )
/// @note This function uses *strides* and not a shape, meaning strides[d]
///       is not the extent of the array in dimension d but rather the step
///       length in the flat index to advance one element in d.
///       Therefore, some conversion of parameters is required when inverting
///       the result with `extract_indices`.
template <class ForwardIt1, class ForwardIt2>
constexpr auto flat_index_from_strides(ForwardIt1 strides_it,
                                       const ForwardIt1 strides_end,
                                       ForwardIt2 indices_it) noexcept {
  std::decay_t<decltype(*strides_it)> result = 0;
  for (; strides_it != strides_end; ++strides_it, ++indices_it) {
    result += *strides_it * *indices_it;
  }
  return result;
}

/// Compute the bounds of a piece of memory.
///
/// Given a pointer `p` to some memory, a shape, and strides, this function
/// returns a begin and an end index such that
/// - `p + begin` is the smallest reachable address and
/// - `p + end` is one past the largest reachable address
///
/// \return A pair of indices `{begin, end}`.
template <class ForwardIt1, class ForwardIt2>
constexpr auto memory_bounds(ForwardIt1 shape_it, const ForwardIt1 shape_end,
                             ForwardIt2 strides_it) noexcept {
  if (shape_it == shape_end) {
    // Scalars are one element wide in memory, this would not be handled
    // correctly by the code below.
    return std::pair{scipp::index{0}, scipp::index{1}};
  }
  scipp::index begin = 0;
  scipp::index end = 0;
  for (; shape_it != shape_end; ++shape_it, ++strides_it) {
    if (*strides_it < 0)
      begin += *shape_it * *strides_it;
    else
      end += *shape_it * *strides_it;
  }
  return std::pair{begin, end};
}

/// Extract individual indices from a flat index.
///
/// Let
///     I = i_0 + l_0 * (i_1 + l_1 * (i_2 + ... (i_{n-2} + l_{n-2} * i_{n-1})))
/// be a flat index computed from indices {i_d} and shape {l_d} in
/// 'column-major' order. Here, this means that i_0 is the fasted moving index
/// and i_{n-1} is slowest.
///
/// If I == prod_{d=0}^ndim (l_d), i.e. one element past the end,
/// the resulting indices are i_d = 0 for d < ndim-1, i_{ndim-1} = l_{ndim-1}
/// unless l_{ndim-1} = 0, see below.
/// This allows setting 'end-iterators' in a well defined manner.
/// However, the result is undefined for greater values of I.
///
/// Values of array elements in `indices` with d > ndim-1 are unspecified
/// except when ndim == 0, i_0 = I.
///
/// Any number of l_d maybe 0 which yields i_d = 0.
/// Except for the one-past-the-end case described above, i_{ndim-1} = 1 if
/// l_{ndim-1} to allow this case to be distinguishable from an index
/// to the end.
///
/// @param flat_index I
/// @param shape_it Begin iterator for {l_d}.
/// @param shape_end End iterator for {l_d}.
/// @param indices_it Begin iterator for {i_d}.
///                   `*indices_it` must always be writeable, even when
///                   `shape_it == shape_end`.
/// @note This function uses a *shape*, i.e. individual dimension sizes
///       to encode the size of the array.
///       Therefore, some conversion of parameters is required when inverting
///       the result with `flat_index_from_strides`.
template <class It1, class It2>
constexpr void extract_indices(scipp::index flat_index, It1 shape_it,
                               It1 shape_end, It2 indices_it) noexcept {
  if (shape_it == shape_end) {
    *indices_it = flat_index;
    return;
  }
  shape_end--; // The last element is set after the loop.
  for (; shape_it != shape_end; ++shape_it, ++indices_it) {
    if (*shape_it != 0) {
      const scipp::index aux = flat_index / *shape_it;
      *indices_it = flat_index - aux * *shape_it;
      flat_index = aux;
    } else {
      *indices_it = 0;
    }
  }
  *indices_it = flat_index;
}
/* Implementation notes for extract_indices
 *
 * With ndim == 2, we have
 *     I = i_0 + l_0 * i_1
 * All numbers are positive integers. Thus I can be decomposed using
 * integer division as follows (note that i_0 < l_0):
 *     x = I / l_0
 *     i_0 = I - x * l_0
 *     i_1 = x
 *
 * With ndim == 3, we have
 *     I = i_0 + l_0 * (i_1 + l_1 * i_2)
 * which can be decomposed as above:
 *     x = I / l_0
 *     i_0 = I - x * l_0
 * Noting that
 *     x = i_1 _ l_1 * i_2
 * we can compute i_1 and i_2 by applying the algorithm recursively.
 *
 * The function implements this algorithm for arbitrary dimensions by rolling
 * the recursion into a loop.
 */

} // namespace scipp