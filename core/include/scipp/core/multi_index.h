// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <optional>

#include "scipp-core_export.h"
#include "scipp/common/index_composition.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/element_array_view.h"

namespace scipp::core {
namespace detail {
SCIPP_CORE_EXPORT void
validate_bin_indices_impl(const ElementArrayViewParams &param0,
                          const ElementArrayViewParams &param1);

template <class Param> void validate_bin_indices(const Param &) {}

/// Check that corresponding bins have matching sizes.
template <class Param0, class Param1, class... Params>
void validate_bin_indices(const Param0 &param0, const Param1 &param1,
                          const Params &... params) {
  if (param0.bucketParams() && param1.bucketParams())
    detail::validate_bin_indices_impl(param0, param1);
  if (param0.bucketParams())
    validate_bin_indices(param0, params...);
  else
    validate_bin_indices(param1, params...);
}

inline auto get_nested_dims() { return Dimensions(); }
template <class T, class... Ts>
auto get_nested_dims(const T &param, const Ts &... params) {
  const auto &bin_param = param.bucketParams();
  return bin_param ? bin_param.dims : get_nested_dims(params...);
}

inline auto get_slice_dim() { return Dim::Invalid; }
template <class T, class... Ts>
auto get_slice_dim(const T &param, const Ts &... params) {
  return param ? param.dim : get_slice_dim(params...);
}

template <scipp::index N_Operands>
constexpr auto get_buffer_size(const scipp::index ndim) noexcept {
  return N_Operands + (N_Operands + 2) * std::max(ndim, scipp::index{2});
}
} // namespace detail

template <scipp::index N> class SCIPP_CORE_EXPORT MultiIndex {
  static_assert(
      N > 0 && N < 5,
      "Explicit instantiations are only provided for up to 4 operands");

public:
  /// Construct without bins.
  template <class... StridesArgs>
  explicit MultiIndex(const Dimensions &iter_dims,
                      const StridesArgs &... strides)
      : m_buffer{std::make_unique<scipp::index[]>(
            detail::get_buffer_size<N>(iter_dims.ndim()))},
        m_inner_ndim{iter_dims.ndim()}, m_ndim{iter_dims.ndim()} {
    store_strides(m_inner_ndim, 0, strides...);
    std::reverse_copy(iter_dims.shape().begin(), iter_dims.shape().end(),
                      shape_it());
  }

  /// Determine from arguments if binned.
  template <class... Params>
  explicit MultiIndex(const ElementArrayViewParams &param,
                      const Params &... params)
      : MultiIndex{
            (!param.bucketParams() && (!params.bucketParams() && ...))
                ? MultiIndex(param.dims(), param.strides(), params.strides()...)
                : MultiIndex{binned_tag{},
                             detail::get_nested_dims(param, params...),
                             param.dims(), param, params...}} {}

private:
  /// Use to disambiguate between constructors.
  struct binned_tag {};

  /// Construct with bins.
  template <class... Params>
  explicit MultiIndex(binned_tag, const Dimensions &inner_dims,
                      const Dimensions &bin_dims, const Params &... params)
      : m_buffer{std::make_unique<scipp::index[]>(
            detail::get_buffer_size<N>(inner_dims.ndim() + bin_dims.ndim()))},
        m_inner_ndim{inner_dims.ndim()}, m_ndim{inner_dims.ndim() +
                                                bin_dims.ndim()},
        m_bin{BinIterator(params)...} {
    detail::validate_bin_indices(params...);

    store_strides(m_inner_ndim, 0,
                  params.bucketParams() ? Strides{inner_dims} : Strides{}...);
    store_strides(bin_ndim(), m_inner_ndim, params.strides()...);
    std::reverse_copy(inner_dims.shape().begin(), inner_dims.shape().end(),
                      shape_it());
    std::reverse_copy(bin_dims.shape().begin(), bin_dims.shape().end(),
                      shape_it() + m_inner_ndim);

    const Dim slice_dim = detail::get_slice_dim(params.bucketParams()...);
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_bin_stride = inner_dims.offset(slice_dim);
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    m_nested_dim_index = m_inner_ndim - inner_dims.index(slice_dim) - 1;

    if (bin_dims.volume() == 0) {
      return; // Operands are empty, there are no bins to load.
    }
    for (scipp::index data = 0; data < N; ++data) {
      load_bin_params(data);
    }
    if (shape(m_nested_dim_index) == 0)
      seek_bin();
  }

public:
  MultiIndex(const MultiIndex &other)
      : m_buffer{other.copy_buffer()}, m_inner_ndim{other.m_inner_ndim},
        m_ndim{other.m_ndim}, m_bin_stride{other.m_bin_stride},
        m_nested_dim_index{other.m_nested_dim_index}, m_bin{other.m_bin} {}

  MultiIndex(MultiIndex &&) noexcept = default;

  MultiIndex &operator=(const MultiIndex &other) {
    m_buffer = other.copy_buffer();
    m_inner_ndim = other.m_inner_ndim;
    m_ndim = other.m_ndim;
    m_bin_stride = other.m_bin_stride;
    m_nested_dim_index = other.m_nested_dim_index;
    m_bin = other.m_bin;
    return *this;
  }

  MultiIndex &operator=(MultiIndex &&) noexcept = default;

  ~MultiIndex() noexcept = default;

  constexpr void increment_outer() noexcept {
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

  constexpr void increment() noexcept {
    for (scipp::index data = 0; data < N; ++data)
      data_index(data) += stride(0, data);
    ++coord(0);
    if (dim_at_end(0))
      increment_outer();
  }

  constexpr void increment_inner_by(const scipp::index distance) noexcept {
    for (scipp::index data = 0; data < N; ++data) {
      data_index(data) += distance * stride(0, data);
    }
    coord(0) += distance;
  }

  [[nodiscard]] auto inner_strides() const noexcept {
    return scipp::span<const scipp::index>(&stride(0, 0), N);
  }

  [[nodiscard]] constexpr scipp::index inner_distance_to_end() const noexcept {
    return shape(0) - coord(0);
  }

  [[nodiscard]] constexpr scipp::index
  inner_distance_to(const MultiIndex &other) const noexcept {
    return other.coord(0) - coord(0);
  }

  /// Set the absolute index. In the special case of iteration with bins,
  /// this sets the *index of the bin* and NOT the full index within the
  /// iterated data.
  constexpr void set_index(const scipp::index index) noexcept {
    if (has_bins()) {
      set_bins_index(index);
    } else {
      extract_indices(index, shape_it(), shape_it(m_inner_ndim), coord_it());
      for (scipp::index data = 0; data < N; ++data) {
        data_index(data) = flat_index(data, 0, m_inner_ndim);
      }
    }
  }

  constexpr void set_to_end() noexcept {
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

  constexpr auto get() const noexcept {
    std::array<scipp::index, N> idx;
    std::copy(&data_index(0), &data_index(0) + N, idx.begin());
    return idx;
  }

  constexpr bool operator==(const MultiIndex &other) const noexcept {
    // Assuming the number dimensions match to make the check cheaper.
    return std::equal(coord_it(), coord_end(), other.coord_it());
  }
  constexpr bool operator!=(const MultiIndex &other) const noexcept {
    return !(*this == other);
  }

  constexpr bool in_same_chunk(const MultiIndex &other,
                               const scipp::index first_dim) const noexcept {
    for (scipp::index dim = first_dim; dim < m_ndim; ++dim) {
      if (coord(dim) != other.coord(dim)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] constexpr auto inner_size() const noexcept { return shape(0); }

  auto begin() const noexcept {
    auto it(*this);
    it.set_index(0);
    return it;
  }

  auto end() const noexcept {
    auto it(*this);
    it.set_to_end();
    return it;
  }

  [[nodiscard]] bool has_bins() const noexcept {
    return m_nested_dim_index != -1;
  }

  /// Return true if the first subindex has a 0 stride
  [[nodiscard]] bool has_stride_zero() const noexcept {
    for (scipp::index dim = 0; dim < m_ndim; ++dim)
      if (stride(dim, 0) == 0)
        return true;
    return false;
  }

private:
  constexpr auto dim_at_end(const scipp::index dim) const noexcept {
    return coord(dim) == std::max(shape(dim), scipp::index{1});
  }

  constexpr auto bin_ndim() const noexcept { return m_ndim - m_inner_ndim; }

  struct BinIterator {
    BinIterator() = default;
    explicit BinIterator(const ElementArrayViewParams &params)
        : m_indices{params.bucketParams().indices} {}

    [[nodiscard]] bool is_binned() noexcept { return m_indices != nullptr; }

    scipp::index m_bin_index{0};
    const std::pair<scipp::index, scipp::index> *m_indices{nullptr};
  };

  constexpr void set_bins_index(const scipp::index index) noexcept {
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

  constexpr void set_to_end_bin() noexcept {
    std::fill(coord_it(), coord_end(), 0);
    const auto last_dim = (bin_ndim() == 0 ? m_nested_dim_index : m_ndim - 1);
    coord(last_dim) = shape(last_dim);

    for (scipp::index data = 0; data < N; ++data) {
      // Only one dim contributes, all others have coord = 0.
      m_bin[data].m_bin_index = coord(last_dim) * stride(last_dim, data);
      load_bin_params(data);
    }
  }

  constexpr void increment_outer_bins() noexcept {
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

  constexpr void increment_bins() noexcept {
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

  constexpr void seek_bin() noexcept {
    do {
      increment_bins();
    } while (shape(m_nested_dim_index) == 0 && !dim_at_end(m_ndim - 1));
  }

  constexpr void load_bin_params(const scipp::index data) noexcept {
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

  scipp::index flat_index(const scipp::index i_data, scipp::index begin_index,
                          const scipp::index end_index) {
    scipp::index res = 0;
    for (; begin_index < end_index; ++begin_index) {
      res += coord(begin_index) * stride(begin_index, i_data);
    }
    return res;
  }

  /// Current flat index into the operands.
  [[nodiscard]] scipp::index &data_index(const scipp::index data) noexcept {
    return m_buffer.get()[data];
  }

  [[nodiscard]] const scipp::index &
  data_index(const scipp::index data) const noexcept {
    return m_buffer.get()[data];
  }

  /// Stride for each operand in each dimension.
  [[nodiscard]] scipp::index &stride(const scipp::index dim,
                                     const scipp::index data) noexcept {
    return m_buffer[data + dim * N + N];
  }

  [[nodiscard]] const scipp::index &
  stride(const scipp::index dim, const scipp::index data) const noexcept {
    return m_buffer[data + dim * N + N];
  }

  template <size_t... I, class... StridesArgs>
  void store_strides_impl(const scipp::index ndim, const scipp::index first_dim,
                          std::index_sequence<I...>,
                          const StridesArgs &... strides) noexcept {
    (
        [&]() {
          for (scipp::index dim = 0; dim < ndim; ++dim) {
            stride(first_dim + dim, I) = strides[ndim - 1 - dim];
          }
        }(),
        ...);
  }

  template <class... StridesArgs>
  void store_strides(const scipp::index ndim, const scipp::index first_dim,
                     const StridesArgs &... strides) noexcept {
    store_strides_impl(ndim, first_dim,
                       std::index_sequence_for<StridesArgs...>(), strides...);
  }

  /// Current index in iteration dimensions for both bin and inner dims.
  [[nodiscard]] scipp::index &coord(const scipp::index dim) noexcept {
    return *coord_it(dim);
  }

  [[nodiscard]] const scipp::index &
  coord(const scipp::index dim) const noexcept {
    return *coord_it(dim);
  }

  [[nodiscard]] auto coord_it(const scipp::index dim = 0) const noexcept {
    return m_buffer.get() + (std::max(m_ndim, scipp::index{2}) * N + dim + N);
  }

  [[nodiscard]] auto coord_end() const noexcept {
    return m_buffer.get() + (std::max(m_ndim, scipp::index{2}) * (N + 1) + N);
  }

  /// Shape of the iteration dimensions for both bin and inner dims.
  [[nodiscard]] scipp::index &shape(const scipp::index dim) noexcept {
    return *shape_it(dim);
  }

  [[nodiscard]] const scipp::index &
  shape(const scipp::index dim) const noexcept {
    return *shape_it(dim);
  }

  [[nodiscard]] auto shape_it(const scipp::index dim = 0) const noexcept {
    return m_buffer.get() +
           (std::max(m_ndim, scipp::index{2}) * (N + 1) + dim + N);
  }

  [[nodiscard]] auto shape_end() const noexcept {
    return m_buffer.get() + (std::max(m_ndim, scipp::index{2}) * (N + 2) + N);
  }

  auto copy_buffer() const {
    const auto size = detail::get_buffer_size<N>(m_ndim);
    auto new_buffer = std::make_unique<scipp::index[]>(size);
    std::copy(m_buffer.get(), m_buffer.get() + size, new_buffer.get());
    return new_buffer;
  }

  /// Container for strides, coords, shape.
  std::unique_ptr<scipp::index[]> m_buffer;
  /// Number of dense dimensions, i.e. same as m_ndim when not binned,
  /// else number of dims in bins.
  scipp::index m_inner_ndim{0};
  /// Total number of dimensions.
  scipp::index m_ndim{0};
  /// Stride from one bin to the next.
  scipp::index m_bin_stride = {};
  /// Index of dim referred to by bin indices to distinguish, e.g., 2D bins
  /// slicing along first or second dim.
  /// -1 if not binned.
  scipp::index m_nested_dim_index{-1};
  /// Parameters of the currently loaded bins.
  std::array<BinIterator, N> m_bin = {};
};

template <class... StridesArgs>
MultiIndex(const Dimensions &, const StridesArgs &...)
    -> MultiIndex<sizeof...(StridesArgs)>;
template <class... Params>
MultiIndex(const ElementArrayViewParams &, const Params &...)
    -> MultiIndex<sizeof...(Params) + 1>;

extern template class MultiIndex<scipp::index{1}>;
extern template class MultiIndex<scipp::index{2}>;
extern template class MultiIndex<scipp::index{3}>;
extern template class MultiIndex<scipp::index{4}>;

} // namespace scipp::core
