// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <algorithm>

#include "scipp/variable/arithmetic.h"
#include "scipp/variable/misc_operations.h"

#include "dataset_test_common.h"

std::vector<bool> make_bools(const scipp::index size,
                             std::initializer_list<bool> pattern) {
  std::vector<bool> result(size);
  auto it = pattern.begin();
  for (auto &&itm : result) {
    if (it == pattern.end())
      it = pattern.begin();
    itm = *(it++);
  }
  return result;
}
std::vector<bool> make_bools(const scipp::index size, bool pattern) {
  return make_bools(size, std::initializer_list<bool>{pattern});
}

DatasetFactory3D::DatasetFactory3D(const scipp::index lx_,
                                   const scipp::index ly_,
                                   const scipp::index lz_, const Dim dim)
    : lx(lx_), ly(ly_), lz(lz_), m_dim(dim) {
  init();
}

void DatasetFactory3D::init() {
  base = Dataset();
  base.setCoord(Dim::Time, makeVariable<double>(Values{rand(1).front()}));
  base.setCoord(m_dim,
                makeVariable<double>(Dimensions{m_dim, lx}, Values(rand(lx))));
  base.setCoord(Dim::Y,
                makeVariable<double>(Dimensions{Dim::Y, ly}, Values(rand(ly))));
  base.setCoord(Dim::Z, makeVariable<double>(
                            Dimensions{{m_dim, lx}, {Dim::Y, ly}, {Dim::Z, lz}},
                            Values(rand(lx * ly * lz))));

  base.setCoord(Dim("labels_x"),
                makeVariable<double>(Dimensions{m_dim, lx}, Values(rand(lx))));
  base.setCoord(Dim("labels_xy"),
                makeVariable<double>(Dimensions{{m_dim, lx}, {Dim::Y, ly}},
                                     Values(rand(lx * ly))));
  base.setCoord(Dim("labels_z"),
                makeVariable<double>(Dimensions{Dim::Z, lz}, Values(rand(lz))));
}

void DatasetFactory3D::seed(const uint32_t value) {
  rand.seed(value);
  randBool.seed(value);
  init();
}

Dataset DatasetFactory3D::make(const bool randomMasks) {
  Dataset dataset(base);
  dataset.setData("values_x", makeVariable<double>(Dimensions{m_dim, lx},
                                                   Values(rand(lx))));
  dataset.setData("data_x",
                  makeVariable<double>(Dimensions{m_dim, lx}, Values(rand(lx)),
                                       Variances(rand(lx))));

  dataset.setData("data_xy",
                  makeVariable<double>(Dimensions{{m_dim, lx}, {Dim::Y, ly}},
                                       Values(rand(lx * ly)),
                                       Variances(rand(lx * ly))));

  dataset.setData(
      "data_zyx",
      makeVariable<double>(Dimensions{{Dim::Z, lz}, {Dim::Y, ly}, {m_dim, lx}},
                           Values(rand(lx * ly * lz)),
                           Variances(rand(lx * ly * lz))));

  dataset.setData(
      "data_xyz",
      makeVariable<double>(Dimensions{{m_dim, lx}, {Dim::Y, ly}, {Dim::Z, lz}},
                           Values(rand(lx * ly * lz))));

  dataset.setData("data_scalar", makeVariable<double>(Values{rand(1).front()}));

  if (randomMasks) {
    for (const auto &item :
         {"values_x", "data_x", "data_xy", "data_zyx", "data_xyz"})
      dataset[item].masks().set(
          "masks_x",
          makeVariable<bool>(Dimensions{m_dim, lx}, Values(randBool(lx))));
    for (const auto &item : {"data_xy", "data_zyx", "data_xyz"})
      dataset[item].masks().set(
          "masks_xy", makeVariable<bool>(Dimensions{{m_dim, lx}, {Dim::Y, ly}},
                                         Values(randBool(lx * ly))));
    for (const auto &item : {"data_zyx", "data_xyz"})
      dataset[item].masks().set(
          "masks_z",
          makeVariable<bool>(Dimensions{Dim::Z, lz}, Values(randBool(lz))));
  } else {
    for (const auto &item :
         {"values_x", "data_x", "data_xy", "data_zyx", "data_xyz"})
      dataset[item].masks().set(
          "masks_x", makeVariable<bool>(Dimensions{m_dim, lx},
                                        Values(make_bools(lx, {false, true}))));
    for (const auto &item : {"data_xy", "data_zyx", "data_xyz"})
      dataset[item].masks().set(
          "masks_xy",
          makeVariable<bool>(Dimensions{{m_dim, lx}, {Dim::Y, ly}},
                             Values(make_bools(lx * ly, {false, true}))));
    for (const auto &item : {"data_zyx", "data_xyz"})
      dataset[item].masks().set(
          "masks_z", makeVariable<bool>(Dimensions{Dim::Z, lz},
                                        Values(make_bools(lz, {false, true}))));
  }

  return dataset;
}

Dataset make_empty() { return Dataset(); }

Dataset make_1d_masked() {
  Random random;
  Dataset ds;
  ds.setData("data_x",
             makeVariable<double>(Dimensions{Dim::X, 10}, Values(random(10))));
  ds["data_x"].masks().set(
      "masks_x", makeVariable<bool>(Dimensions{Dim::X, 10},
                                    Values(make_bools(10, {false, true}))));
  return ds;
}

namespace scipp::testdata {

Dataset make_dataset_x() {
  Dataset d;
  d.setData("a", makeVariable<double>(Dims{Dim::X}, units::kg, Shape{3},
                                      Values{4, 5, 6}));
  d.setData("b", makeVariable<int32_t>(Dims{Dim::X}, units::s, Shape{3},
                                       Values{7, 8, 9}));
  d.setCoord(Dim("scalar"), 1.2 * units::K);
  d.setCoord(Dim::X, makeVariable<double>(Dims{Dim::X}, units::m, Shape{3},
                                          Values{1, 2, 4}));
  d.setCoord(Dim::Y, makeVariable<double>(Dims{Dim::X}, units::m, Shape{3},
                                          Values{1, 2, 3}));
  return d;
}

DataArray make_table(const scipp::index size) {
  Random rand;
  rand.seed(0);
  const Dimensions dims(Dim::Row, size);
  const auto data = makeVariable<double>(dims, Values(rand(dims.volume())),
                                         Variances(rand(dims.volume())));
  const auto x = makeVariable<double>(dims, Values(rand(dims.volume())));
  const auto y = makeVariable<double>(dims, Values(rand(dims.volume())));
  const auto group = astype(
      makeVariable<double>(dims, Values(rand(dims.volume()))), dtype<int64_t>);
  const auto group2 = astype(
      makeVariable<double>(dims, Values(rand(dims.volume()))), dtype<int64_t>);
  return DataArray(data, {{Dim::X, x},
                          {Dim::Y, y},
                          {Dim("group"), group},
                          {Dim("group2"), group2}});
}

} // namespace scipp::testdata
