// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include <benchmark/benchmark.h>

#include <random>

#include "event_list_proxy.h"

using namespace scipp::core;

static void BM_EventListProxy_push_back_baseline(benchmark::State &state) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::poisson_distribution dist(20);

  scipp::index totalCount = 0;
  for (auto _ : state) {
    Dataset d;
    scipp::index nSpec = 100000;
    d.insert(Data::EventTofs, "a", {Dim::X, nSpec});
    d.insert(Data::EventPulseTimes, "a", {Dim::X, nSpec});
    const auto tofs = d.get(Data::EventTofs, "a");
    const auto pulseTimes = d.get(Data::EventPulseTimes, "a");
    for (scipp::index i = 0; i < tofs.size(); ++i) {
      const scipp::index count = dist(mt);
      totalCount += count;
      for (int32_t i = 0; i < count; ++i) {
        tofs[i].push_back(0.0);
        pulseTimes[i].push_back(0.0);
      }
    }
  }

  state.SetItemsProcessed(totalCount);
}
BENCHMARK(BM_EventListProxy_push_back_baseline);

static void BM_EventListProxy_push_back(benchmark::State &state) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::poisson_distribution dist(20);

  scipp::index totalCount = 0;
  for (auto _ : state) {
    Dataset d;
    scipp::index nSpec = 100000;
    d.insert(Data::EventTofs, "a", {Dim::X, nSpec});
    d.insert(Data::EventPulseTimes, "a", {Dim::X, nSpec});
    auto eventLists = zip(d, Access::Key{Data::EventTofs, "a"},
                          Access::Key{Data::EventPulseTimes, "a"});
    for (const auto &eventList : eventLists) {
      const scipp::index count = dist(mt);
      totalCount += count;
      for (int32_t i = 0; i < count; ++i)
        eventList.push_back(0.0, 0.0);
    }
  }

  state.SetItemsProcessed(totalCount);
}
BENCHMARK(BM_EventListProxy_push_back);

static void BM_EventListProxy_read_baseline(benchmark::State &state) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::poisson_distribution dist(state.range(0));

  scipp::index totalCount = 0;
  Dataset d;
  scipp::index nSpec = state.range(1);
  d.insert(Data::EventTofs, "a", {Dim::X, nSpec});
  d.insert(Data::EventPulseTimes, "a", {Dim::X, nSpec});
  auto eventLists = zip(d, Access::Key{Data::EventTofs, "a"},
                        Access::Key{Data::EventPulseTimes, "a"});
  for (const auto &eventList : eventLists) {
    const scipp::index count = dist(mt);
    totalCount += count;
    for (int32_t i = 0; i < count; ++i)
      eventList.push_back(0.0, 0.0);
  }

  for (auto _ : state) {
    const auto tofs = d.get(Data::EventTofs, "a");
    const auto pulseTimes = d.get(Data::EventPulseTimes, "a");
    double tof = 0.0;
    double pulseTime = 0.0;
    for (scipp::index i = 0; i < tofs.size(); ++i) {
      for (size_t j = 0; j < tofs[i].size(); ++j) {
        tof += tofs[i][j];
        pulseTime += pulseTimes[i][j];
      }
    }
    benchmark::DoNotOptimize(tof + pulseTime);
  }

  state.SetItemsProcessed(state.iterations() * totalCount);
  state.SetBytesProcessed(state.iterations() * totalCount * 2 * sizeof(double));
}
// Arguments are nEvent and nSpec.
BENCHMARK(BM_EventListProxy_read_baseline)
    ->RangeMultiplier(2)
    ->Ranges({{2, 1024}, {128, 2 << 15}});

static void BM_EventListProxy_read(benchmark::State &state) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::poisson_distribution dist(state.range(0));

  scipp::index totalCount = 0;
  Dataset d;
  scipp::index nSpec = state.range(1);
  d.insert(Data::EventTofs, "a", {Dim::X, nSpec});
  d.insert(Data::EventPulseTimes, "a", {Dim::X, nSpec});
  auto eventLists = zip(d, Access::Key{Data::EventTofs, "a"},
                        Access::Key{Data::EventPulseTimes, "a"});
  for (const auto &eventList : eventLists) {
    const scipp::index count = dist(mt);
    totalCount += count;
    for (int32_t i = 0; i < count; ++i)
      eventList.push_back(0.0, 0.0);
  }
  const Dataset &const_d(d);

  for (auto _ : state) {
    auto eventLists = zip(const_d, Access::Key{Data::EventTofs, "a"},
                          Access::Key{Data::EventPulseTimes, "a"});
    double tof = 0.0;
    double pulseTime = 0.0;
    for (const auto &eventList : eventLists) {
      for (const auto event : eventList) {
        tof += std::get<0>(event);
        pulseTime += std::get<1>(event);
      }
    }
    benchmark::DoNotOptimize(tof + pulseTime);
  }

  state.SetItemsProcessed(state.iterations() * totalCount);
  state.SetBytesProcessed(state.iterations() * totalCount * 2 * sizeof(double));
}
// Arguments are nEvent and nSpec.
BENCHMARK(BM_EventListProxy_read)
    ->RangeMultiplier(2)
    ->Ranges({{2, 1024}, {128, 2 << 15}});

BENCHMARK_MAIN();
