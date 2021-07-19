// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/variable/convolution.h"
#include "scipp/core/element/convolution.h"
#include "scipp/variable/creation.h"
#include "scipp/variable/shape.h"
#include "scipp/variable/transform.h"

namespace scipp::variable {

Variable convolve(const Variable &var, const Variable &kernel) {
  std::string prefix("_");
  for (const auto &d : var.dims().labels())
    prefix += d.name();
  auto kernel_ = kernel;
  auto var_ = var;
  for (auto &dim : kernel.dims().labels()) {
    kernel_.rename(dim, Dim(prefix + dim.name()));
    var_ = var_.slice({dim, 0, var.dims()[dim] - kernel.dims()[dim] + 1});
  }
  auto convoluted = special_like(var_, FillValue::ZeroNotBool);
  // accumulate_in_place is not multi-threaded at this point and performance is
  // thus currently limited by iterator performance. MultiIndex appears to
  // perform poorly with short inner dimensions, as in typical kernels. We
  // therefore keep the data's inner dimension as inner, and insert the kernel
  // dims to the left of that. Compared to
  //    auto dims_ = merge(var_.dims(), kernel_.dims())
  // this provides a 4x speedup, even though the naive merge should be more
  // cache friendly:
  auto dims_ = var_.dims();
  const auto inner = var_.dims().inner();
  const auto inner_size = var_.dims()[inner];
  dims_.erase(inner);
  dims_ = merge(dims_, kernel_.dims());
  dims_.addInner(inner, inner_size);
  var_ = broadcast(var_, dims_);
  // After broadcast strides along kernel_.dims() are all 0, change to var's
  // strides to setup a sliding window
  for (auto &dim : kernel.dims().labels()) {
    const Dim dim_ = Dim(prefix + dim.name());
    var_.unchecked_strides()[var_.dims().index(dim_)] =
        var.strides()[var.dims().index(dim)];
  }
  convoluted.setUnit(var.unit() * kernel.unit());
  accumulate_in_place(convoluted, var_, kernel_, core::element::convolve,
                      "convolve");
  return convoluted;
}

} // namespace scipp::variable
