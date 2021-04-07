// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Matthew Andrew
#include "scipp/dataset/util.h"
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/misc_operations.h"
#include "scipp/variable/reduction.h"
#include "scipp/variable/util.h"

using namespace scipp::variable;
namespace scipp {

template <class T> scipp::index size_of_bucket_impl(const Variable &view) {
  const auto &[indices, dim, buffer] = view.constituents<T>();
  const auto &[begin, end] = unzip(indices);
  const auto sizes = sum(end - begin).template value<scipp::index>();
  const auto scale = // avoid division by zero
      sizes == 0 ? 0.0 : sizes / static_cast<double>(buffer.dims()[dim]);
  return size_of(indices) + size_of(buffer) * scale;
}

scipp::index size_of(const Variable &view) {
  if (view.dtype() == dtype<bucket<Variable>>) {
    return size_of_bucket_impl<bucket<Variable>>(view);
  }
  if (view.dtype() == dtype<bucket<DataArray>>) {
    return size_of_bucket_impl<bucket<DataArray>>(view);
  }
  if (view.dtype() == dtype<bucket<Dataset>>) {
    return size_of_bucket_impl<bucket<Dataset>>(view);
  }

  auto value_size = view.data().dtype_size();
  auto variance_scale = view.hasVariances() ? 2 : 1;
  return view.dims().volume() * value_size * variance_scale;
}

/// Return the size in memory of a DataArray object. The aligned coord is
/// optional because for a DataArray owned by a dataset aligned coords are
/// assumed to be owned by the dataset as they can apply to multiple arrays.
scipp::index size_of(const DataArray &dataarray, bool include_aligned_coords) {
  scipp::index size = 0;
  size += size_of(dataarray.data());
  for (const auto coord : dataarray.attrs()) {
    size += size_of(coord.second);
  }
  for (const auto mask : dataarray.masks()) {
    size += size_of(mask.second);
  }
  if (include_aligned_coords) {
    for (const auto coord : dataarray.coords()) {
      size += size_of(coord.second);
    }
  }
  return size;
}

scipp::index size_of(const Dataset &dataset) {
  scipp::index size = 0;
  for (const auto &data : dataset) {
    size += size_of(data, false);
  }
  for (const auto coord : dataset.coords()) {
    size += size_of(coord.second);
  }
  return size;
}

DataArray astype(const DataArray &var, const DType type) {
  return DataArray(astype(var.data(), type), var.coords(), var.masks(),
                   var.attrs(), var.name());
}

} // namespace scipp
