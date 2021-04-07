// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <cstddef>
#include <functional>

#include "scipp/core/parallel.h"
#include "scipp/variable/variable.h"

#include "pybind11.h"

namespace py = pybind11;

using namespace scipp;

/// Map C++ types to Python types to perform conversion between scipp containers
/// and numpy arrays.
template <class T> struct ElementTypeMap {
  using CppType = T;
  using PyType = T;
  constexpr static bool convert = false;
};

template <> struct ElementTypeMap<scipp::core::time_point> {
  using CppType = scipp::core::time_point;
  using PyType = int64_t;
  constexpr static bool convert = true;
};

template <bool convert, class Source, class Destination>
void copy_element(const Source &src, Destination &&dst) {
  if constexpr (convert) {
    dst = std::remove_reference_t<Destination>{src};
  } else {
    std::forward<Destination>(dst) = src;
  }
}

/// Cast a py::object referring to an array to py::array_t<auto> if supported.
/// Otherwise, copies the contents into a std::vector<auto>.
template <class T> auto cast_to_array_like(const py::object &obj) {
  using TM = ElementTypeMap<T>;
  using PyType = typename TM::PyType;
  if constexpr (std::is_same_v<T, core::time_point>) {
    // pbj.cast<py::array_t<PyType> does not always work because
    // numpy.datetime64.__int__ delegates to datetime.datetime if the unit is
    // larger than ns and that cannot be converted to long.
    return obj.cast<py::array>()
        .attr("astype")(py::dtype::of<PyType>())
        .template cast<py::array_t<PyType>>();
  } else if constexpr (std::is_pod_v<T>) {
    // Casting to py::array_t applies all sorts of automatic conversions
    // such as integer to double, if required.
    return obj.cast<py::array_t<PyType>>();
  } else {
    // py::array only supports POD types. Use a simple but expensive
    // solution for other types.
    // TODO Related to #290, we should properly support
    //  multi-dimensional input, and ignore bad shapes.
    try {
      return obj.cast<const std::vector<PyType>>();
    } catch (std::runtime_error &) {
      const auto &array = obj.cast<py::array>();
      std::ostringstream oss;
      oss << "Unable to assign object of dtype " << py::str(array.dtype())
          << " to " << scipp::core::dtype<T>;
      throw std::invalid_argument(oss.str());
    }
  }
}

template <bool convert, class T, class View>
void copy_flattened_0d(const py::array_t<T> &data, View &&view) {
  auto r = data.unchecked();
  auto it = view.begin();
  copy_element<convert>(r(), *it);
}

template <bool convert, class T, class View>
void copy_flattened_1d(const py::array_t<T> &data, View &&view) {
  auto r = data.unchecked();
  auto it = view.begin();
  for (ssize_t i = 0; i < r.shape(0); ++i, ++it) {
    copy_element<convert>(r(i), *it);
  }
}

template <bool convert, class T, class View>
void copy_flattened_2d(const py::array_t<T> &data, View &&view) {
  auto r = data.unchecked();
  const auto begin = view.begin();
  core::parallel::parallel_for(
      core::parallel::blocked_range(0, r.shape(0)), [&](const auto &range) {
        auto it = begin + range.begin() * r.shape(1);
        for (ssize_t i = range.begin(); i < range.end(); ++i)
          for (ssize_t j = 0; j < r.shape(1); ++j, ++it)
            copy_element<convert>(r(i, j), *it);
      });
}

template <bool convert, class T, class View>
void copy_flattened_3d(const py::array_t<T> &data, View &&view) {
  auto r = data.unchecked();
  auto it = view.begin();
  for (ssize_t i = 0; i < r.shape(0); ++i)
    for (ssize_t j = 0; j < r.shape(1); ++j)
      for (ssize_t k = 0; k < r.shape(2); ++k, ++it)
        copy_element<convert>(r(i, j, k), *it);
}

template <bool convert, class T, class View>
void copy_flattened_4d(const py::array_t<T> &data, View &&view) {
  auto r = data.unchecked();
  auto it = view.begin();
  for (ssize_t i = 0; i < r.shape(0); ++i)
    for (ssize_t j = 0; j < r.shape(1); ++j)
      for (ssize_t k = 0; k < r.shape(2); ++k)
        for (ssize_t l = 0; l < r.shape(3); ++l, ++it)
          copy_element<convert>(r(i, j, k, l), *it);
}

template <class T> auto memory_begin_end(const py::buffer_info &info) {
  auto *begin = static_cast<const T *>(info.ptr);
  auto *end = static_cast<const T *>(info.ptr);
  const auto &shape = info.shape;
  const auto &strides = info.strides;
  for (scipp::index i = 0; i < scipp::size(shape); ++i)
    if (strides[i] < 0)
      begin += shape[i] * strides[i];
    else
      end += shape[i] * strides[i];
  return std::pair{begin, end};
}

template <class T, class View>
bool memory_overlaps(const py::array_t<T> &data, const View &view) {
  const auto &buffer_info = data.request();
  const auto [data_begin, data_end] = memory_begin_end<std::byte>(buffer_info);
  const auto begin = view.begin();
  const auto end = view.end();
  if (begin == end) {
    return false;
  }
  const auto view_begin = reinterpret_cast<const std::byte *>(&*begin);
  const auto view_end = reinterpret_cast<const std::byte *>(&*end);
  // Note the use of std::less, pointer comparison with operator< may be
  // undefined behavior with pointers from different arrays.
  return std::less<>()(data_begin, view_end) &&
         std::greater_equal<>()(data_end, view_begin);
}

/// Copy all elements from src into dst.
/// Performs an explicit conversion of elements in `src` to the element type of
/// `dst` `convert == true`.
/// Otherwise, elements in src are simply assigned to dst.
template <bool convert, class T, class View>
void copy_flattened(const py::array_t<T> &src, View &&dst) {
  if (scipp::size(dst) != src.size())
    throw std::runtime_error(
        "Numpy data size does not match size of target object.");

  const auto dispatch = [](const py::array_t<T> &src_, View &&dst_) {
    switch (src_.ndim()) {
    case 0:
      return copy_flattened_0d<convert>(src_, std::forward<View>(dst_));
    case 1:
      return copy_flattened_1d<convert>(src_, std::forward<View>(dst_));
    case 2:
      return copy_flattened_2d<convert>(src_, std::forward<View>(dst_));
    case 3:
      return copy_flattened_3d<convert>(src_, std::forward<View>(dst_));
    case 4:
      return copy_flattened_4d<convert>(src_, std::forward<View>(dst_));
    default:
      throw std::runtime_error("Numpy array has more dimensions than supported "
                               "in the current implementation.");
    }
  };
  dispatch(memory_overlaps(src, dst) ? py::array_t<T>(src.request()) : src,
           std::forward<View>(dst));
}

template <class SourceDType, class Destination>
void copy_array_into_view(const py::array_t<SourceDType> &src,
                          Destination &&dst, const Dimensions &dims) {
  const auto &shape = dims.shape();
  if (!std::equal(shape.begin(), shape.end(), src.shape(),
                  src.shape() + src.ndim()))
    throw except::DimensionError("The shape of the provided data "
                                 "does not match the existing "
                                 "object.");
  copy_flattened<ElementTypeMap<
      typename std::remove_reference_t<Destination>::value_type>::convert>(
      src, std::forward<Destination>(dst));
}

template <class SourceDType, class Destination>
void copy_array_into_view(const std::vector<SourceDType> &src,
                          Destination &&dst, const Dimensions &) {
  core::expect::sizeMatches(dst, src);
  std::copy(begin(src), end(src), dst.begin());
}

core::time_point make_time_point(const py::buffer &buffer, double scale = 1.0);
