// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <algorithm>

#include "scipp/common/overloaded.h"
#include "scipp/core/bucket.h"
#include "scipp/core/element/event_operations.h"
#include "scipp/core/element/histogram.h"
#include "scipp/core/except.h"
#include "scipp/core/histogram.h"

#include "scipp/variable/arithmetic.h"
#include "scipp/variable/bins.h"
#include "scipp/variable/bucket_model.h"
#include "scipp/variable/cumulative.h"
#include "scipp/variable/misc_operations.h"
#include "scipp/variable/reduction.h"
#include "scipp/variable/shape.h"
#include "scipp/variable/subspan_view.h"
#include "scipp/variable/transform.h"
#include "scipp/variable/transform_subspan.h"
#include "scipp/variable/variable_factory.h"

#include "scipp/dataset/bin.h"
#include "scipp/dataset/bins.h"
#include "scipp/dataset/bins_view.h"
#include "scipp/dataset/dataset.h"
#include "scipp/dataset/histogram.h"
#include "scipp/dataset/reduction.h"
#include "scipp/dataset/shape.h"

#include "../variable/operations_common.h"
#include "bin_common.h"
#include "dataset_operations_common.h"

namespace scipp::dataset {
namespace {
constexpr auto copy_or_match = [](const auto &a, auto &&b, const Dim dim,
                                  const Variable &srcIndices,
                                  const Variable &dstIndices) {
  if (a.dims().contains(dim))
    copy_slices(a, b, dim, srcIndices, dstIndices);
  else
    core::expect::equals(a, b);
};

constexpr auto expect_matching_keys = [](const auto &a, const auto &b) {
  bool ok = true;
  constexpr auto key = [](const auto &x_) {
    if constexpr (std::is_base_of_v<DataArrayConstView,
                                    std::decay_t<decltype(x_)>>)
      return x_.name();
    else
      return x_.first;
  };
  for (const auto &x : a)
    ok &= b.contains(key(x));
  for (const auto &x : b)
    ok &= a.contains(key(x));
  if (!ok)
    throw std::runtime_error("Mismatching keys in\n" + to_string(a) + " and\n" +
                             to_string(b));
};

} // namespace

void copy_slices(const DataArrayConstView &src, DataArray dst, const Dim dim,
                 const Variable &srcIndices, const Variable &dstIndices) {
  copy_slices(src.data(), dst.data(), dim, srcIndices, dstIndices);
  expect_matching_keys(src.meta(), dst.meta());
  expect_matching_keys(src.masks(), dst.masks());
  for (const auto &[name, coord] : src.meta())
    copy_or_match(coord, dst.meta()[name], dim, srcIndices, dstIndices);
  for (const auto &[name, mask] : src.masks())
    copy_or_match(mask, dst.masks()[name], dim, srcIndices, dstIndices);
}

void copy_slices(const DatasetConstView &src, Dataset dst, const Dim dim,
                 const Variable &srcIndices, const Variable &dstIndices) {
  for (const auto &[name, var] : src.coords())
    copy_or_match(var, dst.coords()[name], dim, srcIndices, dstIndices);
  expect_matching_keys(src.coords(), dst.coords());
  expect_matching_keys(src, dst);
  for (const auto &item : src) {
    const auto &dst_ = dst[item.name()];
    expect_matching_keys(item.attrs(), dst_.attrs());
    expect_matching_keys(item.masks(), dst_.masks());
    copy_or_match(item.data(), dst_.data(), dim, srcIndices, dstIndices);
    for (const auto &[name, var] : item.masks())
      copy_or_match(var, dst_.masks()[name], dim, srcIndices, dstIndices);
    for (const auto &[name, var] : item.attrs())
      copy_or_match(var, dst_.attrs()[name], dim, srcIndices, dstIndices);
  }
}

namespace {
constexpr auto copy_or_resize = [](const auto &var, const Dim dim,
                                   const scipp::index size) {
  auto dims = var.dims();
  if (dims.contains(dim))
    dims.resize(dim, size);
  // Using variableFactory instead of variable::resize for creating
  // _uninitialized_ variable.
  return var.dims().contains(dim)
             ? variable::variableFactory().create(var.dtype(), dims, var.unit(),
                                                  var.hasVariances())
             : copy(var);
};
}

// TODO These functions are an unfortunate near-duplicate of `resize`. However,
// the latter drops coords along the resized dimension. Is there a way to unify
// this? Can the need to drop coords in resize be avoided?
DataArray resize_default_init(const DataArrayConstView &parent, const Dim dim,
                              const scipp::index size) {
  DataArray buffer(copy_or_resize(parent.data(), dim, size));
  for (const auto &[name, var] : parent.coords())
    buffer.coords().set(name, copy_or_resize(var, dim, size));
  for (const auto &[name, var] : parent.masks())
    buffer.masks().set(name, copy_or_resize(var, dim, size));
  for (const auto &[name, var] : parent.attrs())
    buffer.attrs().set(name, copy_or_resize(var, dim, size));
  return buffer;
}

Dataset resize_default_init(const DatasetConstView &parent, const Dim dim,
                            const scipp::index size) {
  Dataset buffer;
  for (const auto &[name, var] : parent.coords())
    buffer.coords().set(name, copy_or_resize(var, dim, size));
  for (const auto &item : parent) {
    buffer.setData(item.name(), copy_or_resize(item.data(), dim, size));
    for (const auto &[name, var] : item.masks())
      buffer[item.name()].masks().set(name, copy_or_resize(var, dim, size));
    for (const auto &[name, var] : item.attrs())
      buffer[item.name()].attrs().set(name, copy_or_resize(var, dim, size));
  }
  return buffer;
}

template <class T>
Variable make_bins_impl(Variable indices, const Dim dim, T &&buffer) {
  indices.setDataHandle(std::make_unique<variable::DataModel<bucket<T>>>(
      indices.data_handle(), dim, std::move(buffer)));
  return indices;
}

/// Construct a bin-variable over a data array.
///
/// Each bin is represented by a Variable slice. `indices` defines the array of
/// bins as slices of `buffer` along `dim`.
Variable make_bins(Variable indices, const Dim dim, DataArray buffer) {
  return make_bins_impl(std::move(indices), dim, std::move(buffer));
}

/// Construct a bin-variable over a dataset.
///
/// Each bin is represented by a Variable slice. `indices` defines the array of
/// bins as slices of `buffer` along `dim`.
Variable make_bins(Variable indices, const Dim dim, Dataset buffer) {
  return make_bins_impl(std::move(indices), dim, std::move(buffer));
}

namespace {
template <class T> Variable bucket_sizes_impl(const Variable &view) {
  const auto &indices = std::get<0>(view.constituents<bucket<T>>());
  const auto [begin, end] = unzip(indices);
  return end - begin;
}
} // namespace

Variable bucket_sizes(const Variable &var) {
  if (var.dtype() == dtype<bucket<Variable>>)
    return bucket_sizes_impl<Variable>(var);
  else if (var.dtype() == dtype<bucket<DataArray>>)
    return bucket_sizes_impl<DataArray>(var);
  else if (var.dtype() == dtype<bucket<Dataset>>)
    return bucket_sizes_impl<Dataset>(var);
  else
    return makeVariable<scipp::index>(var.dims());
}

DataArray bucket_sizes(const DataArrayConstView &array) {
  return {bucket_sizes(array.data()), array.coords(), array.masks(),
          array.attrs()};
}

Dataset bucket_sizes(const DatasetConstView &dataset) {
  return apply_to_items(dataset, [](auto &&_) { return bucket_sizes(_); });
}

bool is_bins(const DataArrayConstView &array) { return is_bins(array.data()); }

bool is_bins(const DatasetConstView &dataset) {
  return std::any_of(dataset.begin(), dataset.end(),
                     [](const auto &item) { return is_bins(item); });
}

} // namespace scipp::dataset

namespace scipp::dataset::buckets {
namespace {

template <class T> auto combine(const Variable &var0, const Variable &var1) {
  const auto &[indices0, dim0, buffer0] = var0.constituents<bucket<T>>();
  const auto &[indices1, dim1, buffer1] = var1.constituents<bucket<T>>();
  static_cast<void>(buffer1);
  static_cast<void>(dim1);
  const Dim dim = dim0;
  const auto [begin0, end0] = unzip(indices0);
  const auto [begin1, end1] = unzip(indices1);
  const auto sizes0 = end0 - begin0;
  const auto sizes1 = end1 - begin1;
  const auto sizes = sizes0 + sizes1;
  const auto end = cumsum(sizes);
  const auto begin = end - sizes;
  const auto total_size =
      end.dims().volume() > 0
          ? end.template values<scipp::index>().as_span().back()
          : 0;
  auto buffer = resize_default_init(buffer0, dim, total_size);
  copy_slices(buffer0, buffer, dim, indices0, zip(begin, end - sizes1));
  copy_slices(buffer1, buffer, dim, indices1, zip(begin + sizes0, end));
  return std::make_shared<variable::DataModel<bucket<T>>>(
      zip(begin, end).data_handle(), dim, std::move(buffer));
}

template <class T>
auto concatenate_impl(const Variable &var0, const Variable &var1) {
  return Variable{merge(var0.dims(), var1.dims()), combine<T>(var0, var1)};
}

template <class T> void reserve_impl(Variable &var, const Variable &shape) {
  // TODO this only reserves in the bins, but assumes buffer has enough space
  auto &&[indices, dim, buffer] = var.constituents<bucket<T>>();
  static_cast<void>(dim);
  static_cast<void>(buffer);
  variable::transform_in_place(
      indices, shape,
      overloaded{
          core::element::arg_list<std::tuple<scipp::index_pair, scipp::index>>,
          core::keep_unit,
          [](auto &begin_end, auto &size) { begin_end.second += size; }});
}

} // namespace

void reserve(Variable &var, const Variable &shape) {
  if (var.dtype() == dtype<bucket<Variable>>)
    return reserve_impl<Variable>(var, shape);
  else if (var.dtype() == dtype<bucket<DataArray>>)
    return reserve_impl<DataArray>(var, shape);
  else
    return reserve_impl<Dataset>(var, shape);
}

Variable concatenate(const Variable &var0, const Variable &var1) {
  if (var0.dtype() == dtype<bucket<Variable>>)
    return concatenate_impl<Variable>(var0, var1);
  else if (var0.dtype() == dtype<bucket<DataArray>>)
    return concatenate_impl<DataArray>(var0, var1);
  else
    return concatenate_impl<Dataset>(var0, var1);
}

DataArray concatenate(const DataArrayConstView &a,
                      const DataArrayConstView &b) {
  return DataArray{
      buckets::concatenate(a.data(), b.data()), union_(a.coords(), b.coords()),
      union_or(a.masks(), b.masks()), intersection(a.attrs(), b.attrs())};
}

/// Reduce a dimension by concatenating all elements along the dimension.
///
/// This is the analogue to summing non-bucket data.
Variable concatenate(const Variable &var, const Dim dim) {
  if (var.dtype() == dtype<bucket<Variable>>)
    return concat_bins<Variable>(var, dim);
  else
    return concat_bins<DataArray>(var, dim);
}

/// Reduce a dimension by concatenating all elements along the dimension.
///
/// This is the analogue to summing non-bucket data.
DataArray concatenate(const DataArrayConstView &array, const Dim dim) {
  return groupby_concat_bins(array, {}, {}, {dim});
}

void append(Variable &var0, const Variable &var1) {
  if (var0.dtype() == dtype<bucket<Variable>>)
    var0.setDataHandle(combine<Variable>(var0, var1));
  else if (var0.dtype() == dtype<bucket<DataArray>>)
    var0.setDataHandle(combine<DataArray>(var0, var1));
  else
    var0.setDataHandle(combine<Dataset>(var0, var1));
}

void append(Variable &&var0, const Variable &var1) { append(var0, var1); }

void append(DataArray &a, const DataArrayConstView &b) {
  expect::coordsAreSuperset(a, b);
  union_or_in_place(a.masks(), b.masks());
  append(a.data(), b.data());
}

Variable histogram(const Variable &data, const Variable &binEdges) {
  using namespace scipp::core;
  auto hist_dim = binEdges.dims().inner();
  auto &&[indices, dim, buffer] = data.constituents<bucket<DataArray>>();
  // `hist_dim` may be the same as a dim of data if there is existing binning.
  // We rename to a dummy to avoid duplicate dimensions, perform histogramming,
  // and then sum over the dummy dimensions, i.e., sum contributions from all
  // inputs bins to the same output histogram. This also allows for threading of
  // 1-D histogramming provided that the input has multiple bins along
  // `hist_dim`.
  std::string nonclashing_name("dummy");
  for (const auto &d : indices.dims().labels())
    nonclashing_name += d.name();
  const Dim dummy = Dim(nonclashing_name);
  indices.rename(hist_dim, dummy);
  const Masker masker(buffer, dim);
  auto hist = variable::transform_subspan(
      buffer.dtype(), hist_dim, binEdges.dims()[hist_dim] - 1,
      subspan_view(buffer.meta()[hist_dim], dim, indices),
      subspan_view(masker.data(), dim, indices), binEdges, element::histogram);
  if (hist.dims().contains(dummy))
    return sum(hist, dummy);
  else
    return hist;
}

Variable map(const DataArrayConstView &function, const Variable &x, Dim dim) {
  if (dim == Dim::Invalid)
    dim = edge_dimension(function);
  const Masker masker(function, dim);
  const auto &coord = bins_view<DataArray>(x).meta()[dim];
  const auto &edges = function.meta()[dim];
  const auto weights = subspan_view(masker.data(), dim);
  if (all(islinspace(edges, dim)).value<bool>()) {
    return variable::transform(coord, subspan_view(edges, dim), weights,
                               core::element::event::map_linspace);
  } else {
    if (!issorted(edges, dim))
      throw except::BinEdgeError("Bin edges of histogram must be sorted.");
    return variable::transform(coord, subspan_view(edges, dim), weights,
                               core::element::event::map_sorted_edges);
  }
}

void scale(DataArray &array, const DataArrayConstView &histogram, Dim dim) {
  if (dim == Dim::Invalid)
    dim = edge_dimension(histogram);
  // Coords along dim are ignored since "binning" is dynamic for buckets.
  expect::coordsAreSuperset(array, histogram.slice({dim, 0}));
  // scale applies masks along dim but others are kept
  union_or_in_place(array.masks(), histogram.slice({dim, 0}).masks());
  const Masker masker(histogram, dim);
  auto data = bins_view<DataArray>(array.data()).data();
  const auto &coord = bins_view<DataArray>(array.data()).meta()[dim];
  const auto &edges = histogram.meta()[dim];
  const auto weights = subspan_view(masker.data(), dim);
  if (all(islinspace(edges, dim)).value<bool>()) {
    transform_in_place(data, coord, subspan_view(edges, dim), weights,
                       core::element::event::map_and_mul_linspace);
  } else {
    if (!issorted(edges, dim))
      throw except::BinEdgeError("Bin edges of histogram must be sorted.");
    transform_in_place(data, coord, subspan_view(edges, dim), weights,
                       core::element::event::map_and_mul_sorted_edges);
  }
}

namespace {
Variable applyMask(const DataArrayConstView &buffer, const Variable &indices,
                   const Dim dim, const Variable &masks) {
  auto indices_copy = Variable(indices);
  auto masked_data = scipp::variable::masked_to_zero(buffer.data(), masks);
  return make_bins(std::move(indices_copy), dim, std::move(masked_data));
}

} // namespace

Variable sum(const Variable &data) {
  auto type = variable::variableFactory().elem_dtype(data);
  type = type == dtype<bool> ? dtype<int64_t> : type;
  const auto unit = variable::variableFactory().elem_unit(data);
  Variable summed;
  if (variable::variableFactory().hasVariances(data))
    summed = Variable(type, data.dims(), unit, Values{}, Variances{});
  else
    summed = Variable(type, data.dims(), unit, Values{});

  if (data.dtype() == dtype<bucket<DataArray>>) {
    const auto &&[indices, dim, buffer] =
        data.constituents<bucket<DataArray>>();
    if (const auto mask_union = irreducible_mask(buffer.masks(), dim)) {
      variable::sum_impl(summed, applyMask(buffer, indices, dim, mask_union));
    } else {
      variable::sum_impl(summed, data);
    }
  } else {
    variable::sum_impl(summed, data);
  }

  return summed;
}

DataArray sum(const DataArrayConstView &data) {
  return {buckets::sum(data.data()), data.coords(), data.masks(), data.attrs()};
}

Dataset sum(const DatasetConstView &d) {
  return apply_to_items(d, [](auto &&... _) { return buckets::sum(_...); });
}

} // namespace scipp::dataset::buckets
