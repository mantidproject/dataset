// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/dataset/bins.h"
#include "scipp/dataset/dataset.h"
#include "scipp/dataset/shape.h"
#include "scipp/dataset/string.h"
#include "scipp/variable/bin_variable.tcc"
#include "scipp/variable/bins.h"
#include "scipp/variable/string.h"

namespace scipp::variable {

INSTANTIATE_BIN_VARIABLE(DatasetView, bucket<Dataset>)
INSTANTIATE_BIN_VARIABLE(DataArrayView, bucket<DataArray>)

} // namespace scipp::variable

namespace scipp::dataset {

class BinVariableMakerDataArray : public variable::BinVariableMaker<DataArray> {
private:
  Variable call_make_bins(const Variable &parent, const Variable &indices,
                          const Dim dim, const DType type,
                          const Dimensions &dims, const units::Unit &unit,
                          const bool variances) const override {
    const auto &source = std::get<2>(parent.constituents<bucket<DataArray>>());
    if (parent.dims() !=
        indices
            .dims()) // would need to select and copy slices from source coords
      throw std::runtime_error(
          "Shape changing operations with bucket<DataArray> not supported yet");
    // TODO This may also fail if the input buffer has extra capacity (rows not
    // in any bucket).
    auto buffer = DataArray(
        variable::variableFactory().create(type, dims, unit, variances),
        source.coords(), source.masks(), source.attrs());
    return make_bins(Variable(indices), dim, std::move(buffer));
  }
  const Variable &data(const Variable &var) const override {
    return buffer(var).data();
  }
  Variable data(Variable &var) const override { return buffer(var).data(); }
};

/// This is currently a dummy implemented just to make `is_bins` work.
class BinVariableMakerDataset
    : public variable::BinVariableMakerCommon<Dataset> {
  Variable create(const DType, const Dimensions &, const units::Unit &,
                  const bool, const parent_list &) const override {
    throw std::runtime_error("not implemented");
  }
  Dim elem_dim(const Variable &) const override {
    throw std::runtime_error("undefined");
  }
  DType elem_dtype(const Variable &) const override {
    throw std::runtime_error("undefined");
  }
  units::Unit elem_unit(const Variable &) const override {
    throw std::runtime_error("undefined");
  }
  void expect_can_set_elem_unit(const Variable &,
                                const units::Unit &) const override {
    throw std::runtime_error("undefined");
  }
  void set_elem_unit(Variable &, const units::Unit &) const override {
    throw std::runtime_error("undefined");
  }
  bool hasVariances(const Variable &) const override {
    throw std::runtime_error("undefined");
  }
};

namespace {
auto register_dataset_types(
    (variable::formatterRegistry().emplace(
         dtype<bucket<Dataset>>,
         std::make_unique<variable::Formatter<bucket<Dataset>>>()),
     variable::formatterRegistry().emplace(
         dtype<bucket<DataArray>>,
         std::make_unique<variable::Formatter<bucket<DataArray>>>()),
     0));
auto register_variable_maker_bucket_DataArray(
    (variable::variableFactory().emplace(
         dtype<bucket<DataArray>>,
         std::make_unique<BinVariableMakerDataArray>()),
     variable::variableFactory().emplace(
         dtype<bucket<Dataset>>, std::make_unique<BinVariableMakerDataset>()),
     0));
} // namespace
} // namespace scipp::dataset
