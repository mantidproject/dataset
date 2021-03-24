// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#include "scipp/variable/arithmetic.h"
#include "scipp/variable/bucket_model.h"
#include "scipp/variable/cumulative.h"
#include "scipp/variable/shape.h"
#include "scipp/variable/variable.tcc"
#include "scipp/variable/variable_factory.h"

namespace scipp::variable {

template <class T>
std::tuple<Variable, Dim, typename T::buffer_type> Variable::to_constituents() {
  Variable tmp;
  std::swap(*this, tmp);
  auto &model = requireT<DataModel<T>>(tmp.data());
  return {tmp.bin_indices(), model.bin_dim(), std::move(model.buffer())};
}

template <class T>
std::tuple<Variable, Dim, typename T::const_element_type>
Variable::constituents() const {
  auto &model = requireT<const DataModel<T>>(data());
  return {bin_indices(), model.bin_dim(), model.buffer()};
}

template <class T>
std::tuple<Variable, Dim, typename T::element_type> Variable::constituents() {
  auto &model = requireT<DataModel<T>>(data());
  return {bin_indices(), model.bin_dim(), model.buffer()};
}

namespace {
auto contiguous_indices(const Variable &parent, const Dimensions &dims) {
  auto indices = Variable(parent, dims);
  copy(parent, indices);
  scipp::index size = 0;
  for (auto &range : indices.values<core::bucket_base::range_type>()) {
    range.second += size - range.first;
    range.first = size;
    size = range.second;
  }
  return std::tuple{indices, size};
}
} // namespace

template <class T> class BinVariableMakerCommon : public AbstractVariableMaker {
public:
  [[nodiscard]] bool is_bins() const override { return true; }
  [[nodiscard]] Variable empty_like(const Variable &prototype,
                                    const std::optional<Dimensions> &shape,
                                    const Variable &sizes) const override {
    if (shape)
      throw except::TypeError(
          "Cannot specify shape in `empty_like` for prototype with bins, shape "
          "must be given by shape of `sizes`.");
    const auto [indices, dim, buf] = prototype.constituents<bucket<T>>();
    auto sizes_ = sizes;
    if (!sizes) {
      const auto &[begin, end] = unzip(indices);
      sizes_ = end - begin;
    }
    const auto end = cumsum(sizes_);
    const auto begin = end - sizes_;
    const auto size = sum(end - begin).template value<scipp::index>();
    return make_bins(zip(begin, end), dim, resize_default_init(buf, dim, size));
  }
};

template <class T> class BinVariableMaker : public BinVariableMakerCommon<T> {
private:
  const Variable &bin_parent(const scipp::span<const Variable> &parents) const {
    constexpr auto is_bins = [](auto &x) {
      return x.dtype() == dtype<bucket<T>>;
    };
    const auto count = std::count_if(parents.begin(), parents.end(), is_bins);
    if (count == 0)
      throw except::BinnedDataError("Bin cannot have zero parents");
    if (!std::is_same_v<T, Variable> && (count > 1))
      throw except::BinnedDataError(
          "Binary operations such as '+' with binned data are only supported "
          "with dtype=VariableView, got dtype=" +
          to_string(dtype<bucket<T>>) +
          ". See "
          "https://scipp.github.io/user-guide/binned-data/"
          "computation.html#Event-centric-arithmetic for equivalent operations "
          "for binned (event) data.");
    return *std::find_if(parents.begin(), parents.end(), is_bins);
  }
  virtual Variable call_make_bins(const Variable &parent,
                                  const Variable &indices, const Dim dim,
                                  const DType type, const Dimensions &dims,
                                  const units::Unit &unit,
                                  const bool variances) const = 0;

public:
  // TODO use span to avoid Variable copies?
  Variable create(const DType elem_dtype, const Dimensions &dims,
                  const units::Unit &unit, const bool variances,
                  const std::vector<Variable> &parents) const override {
    const Variable &parent = bin_parent(parents);
    const auto &[parentIndices, dim, buffer] = parent.constituents<bucket<T>>();
    auto [indices, size] = contiguous_indices(parentIndices, dims);
    auto bufferDims = buffer.dims();
    bufferDims.resize(dim, size);
    return call_make_bins(parent, indices, dim, elem_dtype, bufferDims, unit,
                          variances);
  }

  Dim elem_dim(const Variable &var) const override {
    return std::get<1>(var.constituents<bucket<T>>());
  }
  DType elem_dtype(const Variable &var) const override {
    return std::get<2>(var.constituents<bucket<T>>()).dtype();
  }
  units::Unit elem_unit(const Variable &var) const override {
    return std::get<2>(var.constituents<bucket<T>>()).unit();
  }
  void expect_can_set_elem_unit(const Variable &var,
                                const units::Unit &u) const override {
    // TODO Is this check sufficient?
    if ((elem_unit(var) != u) && (var.dims().volume() != var.data().size()))
      throw except::UnitError("Partial view on data of variable cannot be "
                              "used to change the unit.");
  }
  void set_elem_unit(Variable &var, const units::Unit &u) const override {
    std::get<2>(var.constituents<bucket<T>>()).setUnit(u);
  }
  bool hasVariances(const Variable &var) const override {
    return std::get<2>(var.constituents<bucket<T>>()).hasVariances();
  }
};

/// Macro for instantiating classes and functions required for support a new
/// bin dtype in Variable.
#define INSTANTIATE_BIN_VARIABLE(name, ...)                                    \
  INSTANTIATE_VARIABLE_BASE(name, __VA_ARGS__)                                 \
  template SCIPP_EXPORT                                                        \
      std::tuple<Variable, Dim, typename __VA_ARGS__::const_element_type>      \
      Variable::constituents<__VA_ARGS__>() const;                             \
  template SCIPP_EXPORT                                                        \
      std::tuple<Variable, Dim, typename __VA_ARGS__::element_type>            \
      Variable::constituents<__VA_ARGS__>();                                   \
  template SCIPP_EXPORT                                                        \
      std::tuple<Variable, Dim, typename __VA_ARGS__::buffer_type>             \
      Variable::to_constituents<__VA_ARGS__>();

} // namespace scipp::variable
