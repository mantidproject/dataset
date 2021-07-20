// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Neil Vaytet
#include <benchmark/benchmark.h>

#include "scipp/core/element_array_view.h"

using namespace scipp;
using namespace scipp::core;

// Use a macro to make sure that the view is not invalidated.
#define MAKE_VIEW(ylen)                                                        \
  scipp::index xlen_ = 2000;                                                   \
  Dimensions dims_({{Dim::Y, ylen}, {Dim::X, xlen_}});                         \
  std::vector<double> variable_(dims_.volume());                               \
  ElementArrayView<double> view(variable_.data(), 0, dims_, Strides{xlen_, 1})

static void BM_ElementArrayView_iterators(benchmark::State &state) {
  MAKE_VIEW(state.range(0));

  for (auto _ : state) {
    double sum = 0.0;
    for (auto it = view.begin(); it != view.end(); ++it) {
      sum += *it;
    }
    benchmark::DoNotOptimize(sum);
  }

  state.SetItemsProcessed(state.iterations() * view.size());
}
BENCHMARK(BM_ElementArrayView_iterators)->RangeMultiplier(2)->Range(4, 8 << 10);

static void BM_ElementArrayView_rangeFor(benchmark::State &state) {
  MAKE_VIEW(state.range(0));

  for (auto _ : state) {
    double sum = 0.0;
    for (const auto x : view) {
      sum += x;
    }
    benchmark::DoNotOptimize(sum);
  }

  state.SetItemsProcessed(state.iterations() * view.size());
}
BENCHMARK(BM_ElementArrayView_rangeFor)->RangeMultiplier(2)->Range(4, 8 << 10);

static void BM_ElementArrayView_index(benchmark::State &state) {
  MAKE_VIEW(state.range(0));

  for (auto _ : state) {
    double sum = 0.0;
    for (scipp::index i = 0; i < view.size(); ++i) {
      sum += view[i];
    }
    benchmark::DoNotOptimize(sum);
  }

  state.SetItemsProcessed(state.iterations() * view.size());
}
BENCHMARK(BM_ElementArrayView_index)->RangeMultiplier(2)->Range(4, 8 << 10);

static void BM_ElementArrayView_strided(benchmark::State &state) {
  scipp::index xlen = 100;
  scipp::index ylen = 150;
  scipp::index zlen = state.range(0);
  Dimensions data_dims({{Dim::Z, zlen}, {Dim::Y, ylen}, {Dim::X, xlen}});
  std::vector<double> variable(data_dims.volume());
  // slice Y
  Dimensions iter_dims({{Dim::Z, zlen}, {Dim::Y, 100}, {Dim::X, xlen}});
  ElementArrayView<double> view(variable.data(), 0, iter_dims,
                                Strides{ylen * xlen, xlen, 1});

  for (auto _ : state) {
    double sum = 0.0;
    for (const auto x : view) {
      sum += x;
    }
    benchmark::DoNotOptimize(sum);
  }

  state.SetItemsProcessed(state.iterations() * view.size());
}
BENCHMARK(BM_ElementArrayView_strided)->Range(4, 8 << 8);

BENCHMARK_MAIN();
