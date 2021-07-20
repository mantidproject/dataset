// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Jan-Lukas Wynen
#include "scipp/core/view_index.h"

#include "scipp/core/except.h"

namespace scipp::core {

ViewIndex::ViewIndex(const Dimensions &target_dimensions,
                     const Strides &strides) {
  scipp::index rewind = 0;
  scipp::index dim_write = 0;
  for (scipp::index dim_read = target_dimensions.ndim() - 1; dim_read >= 0;
       --dim_read) {
    const auto stride = strides[dim_read];
    const auto delta = stride - rewind;
    const auto size = target_dimensions.size(dim_read);
    rewind = size * stride;
    if (delta != 0 || stride == 0) {
      m_shape[dim_write] = size;
      m_delta[dim_write] = delta;
      m_strides[dim_write] = stride;
      ++dim_write;
    } else {
      // The memory for this dimension is contiguous with the previous dim,
      // so we flatten them into one dimension.
      // This cannot happen in the innermost dim because if stride != 0,
      // delta != 0 because rewind == 0.
      m_shape[dim_write - 1] *= size;
    }
  }
  m_ndim = dim_write;
}

} // namespace scipp::core
