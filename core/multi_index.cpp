
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/core/multi_index.h"
#include "scipp/core/except.h"

namespace scipp::core {

namespace detail {
void validate_bin_indices_impl(const ElementArrayViewParams &param0,
                               const ElementArrayViewParams &param1) {
  const auto iterDims = param0.dims();
  auto index = MultiIndex(iterDims, param0.strides(), param1.strides());
  const auto indices0 = param0.bucketParams().indices;
  const auto indices1 = param1.bucketParams().indices;
  constexpr auto size = [](const auto range) {
    return range.second - range.first;
  };
  for (scipp::index i = 0; i < iterDims.volume(); ++i) {
    const auto [i0, i1] = index.get();
    if (size(indices0[i0]) != size(indices1[i1]))
      throw except::BinnedDataError(
          "Bin size mismatch in operation with binned data. Refer to "
          "https://scipp.github.io/user-guide/binned-data/"
          "computation.html#Overview-and-Quick-Reference for equivalent "
          "operations for binned data (event data).");
    index.increment();
  }
}
} // namespace detail

template <scipp::index N> void MultiIndex<N>::increment_outer() noexcept {
  // Go through all nested dims (with bins) / all dims (without bins)
  // where we have reached the end.
  for (scipp::index d = 0; (d < m_inner_ndim - 1) && dim_at_end(d); ++d) {
    for (scipp::index data = 0; data < N; ++data) {
      data_index(data) +=
          // take a step in dimension d+1
          stride(d + 1, data)
          // rewind dimension d (coord(d) == m_shape[d])
          - coord(d) * stride(d, data);
    }
    ++coord(d + 1);
    coord(d) = 0;
  }
  // Nested dims incremented, move on to bins.
  // Note that we do not check whether there are any bins but whether
  // the outer Variable is scalar because to loop above is enough to set up
  // the coord in that case.
  if (bin_ndim() != 0 && dim_at_end(m_inner_ndim - 1))
    seek_bin();
}

template <scipp::index N> void MultiIndex<N>::increment_bins() noexcept {
  const auto dim = m_inner_ndim;
  for (scipp::index data = 0; data < N; ++data) {
    m_bin[data].m_bin_index += stride(dim, data);
  }
  std::fill(coord_it(), coord_it(m_inner_ndim), 0);
  ++coord(dim);
  if (dim_at_end(dim))
    increment_outer_bins();
  if (!dim_at_end(m_ndim - 1)) {
    for (scipp::index data = 0; data < N; ++data) {
      load_bin_params(data);
    }
  }
}

template <scipp::index N> void MultiIndex<N>::increment_outer_bins() noexcept {
  for (scipp::index dim = m_inner_ndim; (dim < m_ndim - 1) && dim_at_end(dim);
       ++dim) {
    for (scipp::index data = 0; data < N; ++data) {
      m_bin[data].m_bin_index +=
          // take a step in dimension dim+1
          stride(dim + 1, data)
          // rewind dimension dim (coord(d) == m_shape[d])
          - coord(dim) * stride(dim, data);
    }
    ++coord(dim + 1);
    coord(dim) = 0;
  }
}

template <scipp::index N> void MultiIndex<N>::seek_bin() noexcept {
  do {
    increment_bins();
  } while (shape(m_nested_dim_index) == 0 && !dim_at_end(m_ndim - 1));
}

template <scipp::index N>
void MultiIndex<N>::load_bin_params(const scipp::index data) noexcept {
  if (!m_bin[data].is_binned()) {
    data_index(data) = flat_index(data, 0, m_ndim);
  } else if (!dim_at_end(m_ndim - 1)) {
    // All bins are guaranteed to have the same size.
    // Use common m_shape and m_nested_stride for all.
    const auto [begin, end] = m_bin[data].m_indices[m_bin[data].m_bin_index];
    shape(m_nested_dim_index) = end - begin;
    data_index(data) = m_bin_stride * begin;
  }
  // else: at end of bins
}

template <scipp::index N>
void MultiIndex<N>::set_index(const scipp::index index) noexcept {
  if (has_bins()) {
    set_bins_index(index);
  } else {
    extract_indices(index, shape_it(), shape_it(m_inner_ndim), coord_it());
    for (scipp::index data = 0; data < N; ++data) {
      data_index(data) = flat_index(data, 0, m_inner_ndim);
    }
  }
}

template <scipp::index N>
void MultiIndex<N>::set_bins_index(const scipp::index index) noexcept {
  std::fill(coord_it(0), coord_it(m_inner_ndim), 0);
  if (bin_ndim() == 0 && index != 0) {
    coord(m_nested_dim_index) = shape(m_nested_dim_index);
  } else {
    extract_indices(index, shape_it(m_inner_ndim), shape_end(),
                    coord_it(m_inner_ndim));
  }

  for (scipp::index data = 0; data < N; ++data) {
    m_bin[data].m_bin_index = flat_index(data, m_inner_ndim, m_ndim);
    load_bin_params(data);
  }
  if (shape(m_nested_dim_index) == 0 && !dim_at_end(m_ndim - 1))
    seek_bin();
}

template <scipp::index N> void MultiIndex<N>::set_to_end() noexcept {
  if (has_bins()) {
    set_to_end_bin();
  } else {
    if (m_inner_ndim == 0) {
      coord(0) = 1;
    } else {
      std::fill(coord_it(0), coord_it(m_inner_ndim - 1), 0);
      coord(m_inner_ndim - 1) = shape(m_inner_ndim - 1);
    }
    for (scipp::index data = 0; data < N; ++data) {
      data_index(data) = flat_index(data, 0, m_inner_ndim);
    }
  }
}

template <scipp::index N> void MultiIndex<N>::set_to_end_bin() noexcept {
  std::fill(coord_it(), coord_end(), 0);
  const auto last_dim = (bin_ndim() == 0 ? m_nested_dim_index : m_ndim - 1);
  coord(last_dim) = shape(last_dim);

  for (scipp::index data = 0; data < N; ++data) {
    // Only one dim contributes, all others have coord = 0.
    m_bin[data].m_bin_index = coord(last_dim) * stride(last_dim, data);
    load_bin_params(data);
  }
}

template <scipp::index N>
scipp::index MultiIndex<N>::flat_index(const scipp::index i_data,
                                       scipp::index begin_index,
                                       const scipp::index end_index) {
  scipp::index res = 0;
  for (; begin_index < end_index; ++begin_index) {
    res += coord(begin_index) * stride(begin_index, i_data);
  }
  return res;
}

template class MultiIndex<scipp::index{1}>;
template class MultiIndex<scipp::index{2}>;
template class MultiIndex<scipp::index{3}>;
template class MultiIndex<scipp::index{4}>;

} // namespace scipp::core