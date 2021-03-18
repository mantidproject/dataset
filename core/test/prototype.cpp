// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
#include <gtest/gtest.h>

#include "scipp/common/shared_deep_ptr.h"
#include "scipp/core/dimensions.h"
#include "scipp/core/element_array.h"
#include "scipp/units/dim.h"
#include "scipp/units/unit.h"

#include "test_macros.h"

using namespace scipp;

class Variable {
private:
  Dimensions m_dims;
  scipp::index m_offset{0};
  units::Unit m_unit; // TODO must share this somehow
  element_array<double> m_values;
  bool is_slice() const noexcept {
    return m_offset != 0 || m_dims.volume() != m_values.size();
  }

public:
  Variable() = default;
  Variable(const Dimensions &dims, const units::Unit &unit,
           const element_array<double> &values)
      : m_dims(dims), m_unit(unit), m_values(values) {
    if (dims.volume() != values.size())
      throw std::runtime_error("Dims do not match size");
  }

  const auto &dims() const { return m_dims; }
  const auto &unit() const { return m_unit; }

  auto begin() { return m_values.begin() + m_offset; }
  auto end() { return m_values.begin() + m_offset + m_dims.volume(); }
  auto begin() const { return m_values.begin() + m_offset; }
  auto end() const { return m_values.begin() + m_offset + m_dims.volume(); }

  // for Python, should return element_array by value, sharing ownership
  scipp::span<const double> values() const { return {begin(), end()}; }
  scipp::span<double> values() { return {begin(), end()}; }

  bool operator==(const Variable &other) const {
    return dims() == other.dims() && unit() == other.unit() &&
           std::equal(begin(), end(), other.begin(), other.end());
  }
  bool operator!=(const Variable &other) const { return !operator==(other); }

  Variable slice(const Dim dim, const scipp::index offset) const {
    Variable out(*this);
    auto out_dims = dims();
    out_dims.erase(dim);
    out.m_dims = out_dims;
    out.m_offset = offset;
    return out;
  }

  void setunit(const units::Unit &unit) {
    if (m_unit == unit)
      return;
    if (m_dims.volume() != m_values.size())
      throw std::runtime_error("Cannot set unit on slice");
    m_unit = unit;
  }

  Variable deepcopy() const {
    return is_slice() ? Variable{dims(), unit(), {begin(), end()}}
                      : Variable{dims(), unit(), m_values.deepcopy()};
  }
};

// Sibling of class Dimensions, but unordered
class Sizes {
public:
  Sizes() = default;
  Sizes(const Dimensions &dims) {
    for (const auto dim : dims.labels())
      m_sizes[dim] = dims[dim];
  }
  Sizes(const std::unordered_map<Dim, scipp::index> &sizes) : m_sizes(sizes) {}

  bool contains(const Dim dim) const noexcept {
    return m_sizes.count(dim) != 0;
  }

  scipp::index operator[](const Dim dim) const {
    if (!contains(dim))
      throw std::runtime_error("dim not found");
    return m_sizes.at(dim);
  }

  bool contains(const Dimensions &dims) {
    for (const auto &dim : dims.labels())
      if (m_sizes.count(dim) == 0 || m_sizes.at(dim) != dims[dim])
        return false;
    return true;
  }

  Sizes slice(const Dim dim, const scipp::index) const {
    auto sizes = m_sizes;
    sizes.erase(dim);
    return {sizes};
  }

private:
  std::unordered_map<Dim, scipp::index> m_sizes;
};

template <class T>
auto slice_map(const T &map, const Dim dim, const scipp::index offset) {
  T out;
  for (const auto &[key, value] : map) {
    if (value.dims().contains(dim))
      out[key] = value.slice(dim, offset);
    else
      out[key] = value;
  }
  return out;
}

// Dataset: dims can be extended
// Coords: cannot extend, except for special case bin edges
// slice of coords: drop items, slice items
template <class Key, class Value> class Dict {
public:
  using map_type = std::unordered_map<Key, Value>;
  Dict() = default;
  Dict(const Sizes &sizes,
       std::initializer_list<std::pair<const Key, Value>> items)
      : Dict(sizes, map_type(items)) {}
  Dict(const Sizes &sizes, const map_type &items) : m_sizes(sizes) {
    for (const auto &[key, value] : items)
      setitem(key, value);
  }
  auto operator[](const Key &key) const { return m_items.at(key); }
  void setitem(const Key &key, Value coord) {
    if (!m_sizes.contains(coord.dims()))
      throw std::runtime_error("cannot add coord exceeding DataArray dims");
    m_items.operator[](key) = coord;
  }
  bool contains(const Key &key) const { return m_items.count(key) == 1; }
  auto begin() const { return m_items.begin(); }
  auto end() const { return m_items.end(); }
  auto begin() { return m_items.begin(); }
  auto end() { return m_items.end(); }

  auto slice(const Dim dim, const scipp::index offset) const {
    return Dict(m_sizes.slice(dim, offset), slice_map(m_items, dim, offset));
  }

private:
  // Note: We have no way of preventing name clashes of coords with attrs, this
  // would need to be handled dynamically on *access*
  map_type m_items;
  Sizes m_sizes;
};

using Coords = Dict<Dim, Variable>;
using Masks = Dict<std::string, Variable>;

// DataArray slice converts coords to attrs => slice contains new attrs dict =>
// cannot add attr via slice (works but does notthing)

// Requires:
// Variable: dims and shape do not change
// Coords: sizes dict does not change
class DataArray {
public:
  DataArray() = default;
  DataArray(Variable data, const std::unordered_map<Dim, Variable> &coords)
      : m_data(std::move(data)), m_coords(m_data.dims(), coords),
        m_masks(std::make_shared<Masks>(m_data.dims(),
                                        typename Masks::map_type{})) {}
  DataArray(Variable data, Coords coords, Masks masks)
      : m_data(data), m_coords(coords),
        m_masks(std::make_shared<Masks>(masks)) {
    // TODO verify sizes of coords and masks
  }

  const auto &dims() const { return m_data.dims(); }
  // should share whole var, not just values?
  // ... or include unit in shared part?
  // da.data.unit = 'm' ok, DataArray does not care
  // da.data.rename_dims(...) shoud NOT affect da?! since dims is invariant
  // => rename_dims should return *new* variable
  // required by DataArray
  auto data() const {
    return m_data;
  } // should never return mutable reference since this could break
    // invariants... see current impl which returns VariableView, preventing bad
    // changes
  // auto shared_data() { return m_data; } // bind to da.data
  void setdata(const Variable &var) { m_data = var; }
  // metadata dicts return by reference, in Python bindings we need to use
  // keep_alive on the owning DataArray
  const auto &coords() const { return m_coords; }
  auto &coords() { return m_coords; }
  const auto &masks() const { return *m_masks; }
  auto &masks() { return *m_masks; }
  // da.coords['x'] = x # must check dims... should Coords store data dims?
  // auto shared_coords() { return m_coords; } // bind to da.coords

  DataArray slice(const Dim dim, const scipp::index offset) const {
    return {m_data.slice(dim, offset), m_coords.slice(dim, offset),
            m_masks->slice(dim, offset)};
  }

  DataArray view_with_coords(const Coords &coords) const {
    DataArray out;
    out.m_data = m_data;
    out.m_coords = Coords(m_data.dims(), {});
    for (const auto &[dim, coord] : coords)
      if (m_data.dims().contains(coord.dims()))
        out.m_coords.setitem(dim, coord);
    out.m_masks = m_masks.owner();
    return out;
  }

private:
  Variable m_data;
  Coords m_coords;
  shared_deep_ptr<Masks> m_masks;
};

// Requires:
// DataArray: dims and shape do not change, coords aligned + do not change
class Dataset {
public:
  const auto &coords() const { return m_coords; }
  auto &coords() { return m_coords; }
  auto operator[](const std::string &name) const {
    return m_items.at(name).view_with_coords(m_coords);
  }
  void setitem(const std::string &name, const DataArray &item) {
    // TODO properly check compatible dims and grow
    m_coords =
        Coords(Sizes(item.data().dims()), {m_coords.begin(), m_coords.end()});
    for (auto &&[dim, coord] : item.coords())
      setcoord(dim, coord);
    m_items[name] = DataArray(
        item.data(), {}); // no coords, gets set dynamically in operator[]
  }
  void setcoord(const Dim dim, const Variable &coord) {
    if (m_coords.contains(dim) && m_coords[dim] != coord)
      throw std::runtime_error("Coords not aligned");
    m_coords.setitem(dim, coord);
  }

  Dataset slice(const Dim dim, const scipp::index offset) const {
    Dataset out;
    out.m_coords = m_coords.slice(dim, offset);
    out.m_items = slice_map(m_items, dim, offset);
    return out;
  }

private:
  Coords m_coords;
  std::unordered_map<std::string, DataArray> m_items;
};

Variable copy(const Variable &var) { return var.deepcopy(); }
DataArray copy(const DataArray &da) { return da; }
Dataset copy(const Dataset &ds) { return ds; }

class PrototypeTest : public ::testing::Test {
protected:
  Dimensions dimsX = Dimensions(Dim::X, 3);
  Variable var{dimsX, units::m, {1, 2, 3}};
};

TEST_F(PrototypeTest, variable) {
  EXPECT_EQ(Variable(var).values().data(), var.values().data()); // shallow copy
  EXPECT_NE(copy(var).values().data(), var.values().data());     // deep copy
  auto shared = var;
  shared.values()[0] = 1.1;
  EXPECT_EQ(var.values()[0], 1.1);
}

TEST_F(PrototypeTest, variable_slice) {
  auto slice = var.slice(Dim::X, 1);
  EXPECT_EQ(slice, Variable(Dimensions(), units::m, {2}));
  EXPECT_ANY_THROW(slice.setunit(units::s));
  slice.values()[0] = 1.1;
  EXPECT_EQ(var.values()[1], 1.1);
  EXPECT_EQ(copy(slice), slice);
}

TEST_F(PrototypeTest, data_array) {
  auto da = DataArray(var, {});
  EXPECT_EQ(da.data().values().data(), var.values().data()); // shallow copy
  da.coords().setitem(Dim::X, var);
  EXPECT_EQ(da.coords()[Dim::X].values().data(),
            var.values().data()); // shallow copy
  for (const auto da2 : {DataArray(da), copy(da)}) {
    // shallow copy of data and coords
    EXPECT_EQ(da2.data().values().data(), da.data().values().data());
    EXPECT_EQ(da2.coords()[Dim::X].values().data(),
              da.coords()[Dim::X].values().data());
  }
}

TEST_F(PrototypeTest, data_array_coord) {
  auto da = DataArray(var, {{Dim::X, Variable(dimsX, units::m, {2, 4, 8})}});
  const auto coord = da.coords()[Dim::X];
  da = DataArray(var, {});
  EXPECT_TRUE(equals(
      coord.values(),
      Variable(dimsX, units::m, {2, 4, 8}).values())); // coord is sole owner
}

TEST_F(PrototypeTest, dataset) {
  auto da1 = DataArray(Variable(dimsX, units::m, {1, 2, 3}),
                       {{Dim::X, Variable(dimsX, units::m, {1, 1, 1})}});
  auto da2 = DataArray(Variable(dimsX, units::m, {1, 2, 3}), {});
  auto ds = Dataset();
  ds.setitem("a", da1);
  for (const auto ds2 : {Dataset(ds), copy(ds)}) {
    // shallow copy of items and coords
    EXPECT_EQ(ds2["a"].data().values().data(), ds["a"].data().values().data());
    EXPECT_EQ(ds2.coords()[Dim::X].values().data(),
              ds.coords()[Dim::X].values().data());
  }

  ds.coords().setitem(Dim("coord1"), Variable(dimsX, units::m, {1, 2, 3}));
  EXPECT_TRUE(ds["a"].coords().contains(Dim("coord1")));
  EXPECT_TRUE(ds.coords().contains(Dim("coord1")));

  // ds["a"] returns DataArray with new coords dict
  ds["a"].coords().setitem(Dim("coord2"), Variable(dimsX, units::m, {1, 2, 3}));
  EXPECT_FALSE(ds["a"].coords().contains(Dim("coord2")));
  EXPECT_FALSE(ds.coords().contains(Dim("coord2")));

  // ds["a"] returns DataArray referencing existing masks dict
  ds["a"].masks().setitem("mask", Variable(dimsX, units::m, {1, 2, 3}));
  EXPECT_TRUE(ds["a"].masks().contains("mask"));
}

class VariableContractTest : public ::testing::Test {
protected:
  Dimensions dimsX = Dimensions(Dim::X, 3);
  Variable var{dimsX, units::m, {1, 2, 3}};
};

TEST_F(VariableContractTest, values_can_be_set) {
  var.values()[0] = 17;
  EXPECT_EQ(var.values()[0], 17);
}

TEST_F(VariableContractTest, unit_can_be_set) {
  var.setunit(units::s);
  EXPECT_EQ(var.unit(), units::s);
}

TEST_F(VariableContractTest, shallow_copy_values_can_be_set) {
  auto shallow = var;
  shallow.values()[0] = 17;
  EXPECT_EQ(var.values()[0], 17);
}

TEST_F(VariableContractTest, shallow_copy_unit_can_be_set) {
  auto shallow = var;
  shallow.setunit(units::s);
  EXPECT_EQ(var.unit(), units::s);
}

TEST_F(VariableContractTest, slice_values_can_be_set) {
  auto slice = var.slice(Dim::X, 1);
  slice.values()[0] = 17;
  EXPECT_EQ(var.values()[1], 17);
}

TEST_F(VariableContractTest, slice_unit_cannot_be_changed) {
  auto slice = var.slice(Dim::X, 1);
  EXPECT_ANY_THROW(slice.setunit(units::s));
}

class DataArrayContractTest : public ::testing::Test {
protected:
  Dimensions dimsX = Dimensions(Dim::X, 3);
  Variable var{dimsX, units::m, {1, 2, 3}};
  DataArray da{var, {}};
};

TEST_F(DataArrayContractTest, data_can_be_set) {
  const Variable data{dimsX, units::s, {2, 4, 8}};
  // Note that unlike Python we cannot use da.data() = data,
  // The property setter for `data` has to be bound to setdata
  da.setdata(data);
  EXPECT_EQ(da.data(), data);
}

TEST_F(DataArrayContractTest, data_values_can_be_set) {
  da.data().values()[0] = 17;
  EXPECT_EQ(da.data().values()[0], 17);
}

TEST_F(DataArrayContractTest, data_unit_can_be_set) {
  da.data().setunit(units::s);
  EXPECT_EQ(da.data().unit(), units::s);
}

TEST_F(DataArrayContractTest, coords_can_be_added) {
  da.coords().setitem(Dim("new"), var);
  EXPECT_TRUE(da.coords().contains(Dim("new")));
}

TEST_F(DataArrayContractTest, coord_values_can_be_set) {
  da.coords().setitem(Dim::X, var);
  da.coords()[Dim::X].values()[0] = 17;
  EXPECT_EQ(da.coords()[Dim::X].values()[0], 17);
}

TEST_F(DataArrayContractTest, coord_unit_can_be_set) {
  da.coords().setitem(Dim::X, var);
  da.coords()[Dim::X].setunit(units::s);
  EXPECT_EQ(da.coords()[Dim::X].unit(), units::s);
}

TEST_F(DataArrayContractTest, masks_can_be_added) {
  da.masks().setitem("mask", var);
  EXPECT_TRUE(da.masks().contains("mask"));
}

TEST_F(DataArrayContractTest, mask_values_can_be_set) {
  da.masks().setitem("mask", var);
  da.masks()["mask"].values()[0] = 17;
  EXPECT_EQ(da.masks()["mask"].values()[0], 17);
}

TEST_F(DataArrayContractTest, mask_unit_can_be_set) {
  da.masks().setitem("mask", var);
  da.masks()["mask"].setunit(units::s);
  EXPECT_EQ(da.masks()["mask"].unit(), units::s);
}

TEST_F(DataArrayContractTest, shallow_copy_data_values_can_be_set) {
  auto shallow = da;
  da.data().values()[0] = 17;
  EXPECT_EQ(da.data().values()[0], 17);
}

TEST_F(DataArrayContractTest, shallow_copy_data_unit_can_be_set) {
  auto shallow = da;
  shallow.data().setunit(units::s);
  EXPECT_EQ(da.data().unit(), units::s);
}

TEST_F(DataArrayContractTest, shallow_copy_coords_cannot_be_added) {
  auto shallow = da;
  shallow.coords().setitem(Dim("new"), var);
  EXPECT_TRUE(shallow.coords().contains(Dim("new")));
  EXPECT_FALSE(da.coords().contains(Dim("new")));
}

TEST_F(DataArrayContractTest, shallow_copy_coord_values_can_be_set) {
  da.coords().setitem(Dim::X, var);
  auto shallow = da;
  shallow.coords()[Dim::X].values()[0] = 17;
  EXPECT_EQ(da.coords()[Dim::X].values()[0], 17);
}

TEST_F(DataArrayContractTest, shallow_copy_coord_unit_can_be_set) {
  da.coords().setitem(Dim::X, var);
  auto shallow = da;
  shallow.coords()[Dim::X].setunit(units::s);
  EXPECT_EQ(da.coords()[Dim::X].unit(), units::s);
}

TEST_F(DataArrayContractTest, shallow_copy_masks_cannot_be_added) {
  auto shallow = da;
  shallow.masks().setitem("mask", var);
  EXPECT_TRUE(shallow.masks().contains("mask"));
  EXPECT_FALSE(da.masks().contains("mask"));
}

TEST_F(DataArrayContractTest, shallow_copy_mask_values_can_be_set) {
  da.masks().setitem("mask", var);
  auto shallow = da;
  shallow.masks()["mask"].values()[0] = 17;
  EXPECT_EQ(da.masks()["mask"].values()[0], 17);
}

TEST_F(DataArrayContractTest, shallow_copy_mask_unit_can_be_set) {
  da.masks().setitem("mask", var);
  auto shallow = da;
  shallow.masks()["mask"].setunit(units::s);
  EXPECT_EQ(da.masks()["mask"].unit(), units::s);
}

TEST_F(DataArrayContractTest, slice_data_values_can_be_set) {
  auto slice = da.slice(Dim::X, 1);
  da.data().values()[0] = 17;
  EXPECT_EQ(da.data().values()[0], 17);
}

TEST_F(DataArrayContractTest, slice_data_unit_cannot_be_set) {
  auto slice = da.slice(Dim::X, 1);
  EXPECT_ANY_THROW(slice.data().setunit(units::s));
}

TEST_F(DataArrayContractTest, slice_coords_cannot_be_added) {
  auto slice = da.slice(Dim::X, 1);
  slice.coords().setitem(Dim("new"), var.slice(Dim::X, 1));
  EXPECT_TRUE(slice.coords().contains(Dim("new")));
  EXPECT_FALSE(da.coords().contains(Dim("new")));
}

TEST_F(DataArrayContractTest, slice_coord_values_can_be_set) {
  da.coords().setitem(Dim::X, var);
  auto slice = da.slice(Dim::X, 1);
  slice.coords()[Dim::X].values()[0] = 17;
  EXPECT_EQ(da.coords()[Dim::X].values()[1], 17);
}

TEST_F(DataArrayContractTest, slice_coord_unit_cannot_be_set) {
  da.coords().setitem(Dim::X, var);
  auto slice = da.slice(Dim::X, 1);
  EXPECT_ANY_THROW(slice.coords()[Dim::X].setunit(units::s));
}

TEST_F(DataArrayContractTest, slice_masks_cannot_be_added) {
  auto slice = da.slice(Dim::X, 1);
  slice.masks().setitem("mask", var.slice(Dim::X, 1));
  EXPECT_TRUE(slice.masks().contains("mask"));
  EXPECT_FALSE(da.masks().contains("mask"));
}

TEST_F(DataArrayContractTest, slice_mask_values_can_be_set) {
  da.masks().setitem("mask", var);
  auto slice = da.slice(Dim::X, 1);
  slice.masks()["mask"].values()[0] = 17;
  EXPECT_EQ(da.masks()["mask"].values()[1], 17);
}

TEST_F(DataArrayContractTest, slice_mask_unit_cannot_be_set) {
  da.masks().setitem("mask", var);
  auto slice = da.slice(Dim::X, 1);
  EXPECT_ANY_THROW(slice.masks()["mask"].setunit(units::s));
}

class DatasetContractTest : public ::testing::Test {
protected:
  DatasetContractTest() { ds.setitem("a", da); }
  Dimensions dimsX = Dimensions(Dim::X, 3);
  Variable var{dimsX, units::m, {1, 2, 3}};
  DataArray da{var, {}};
  Dataset ds;
};

TEST_F(DatasetContractTest, coords_can_be_added) {
  ds.coords().setitem(Dim("new"), var);
  EXPECT_TRUE(ds.coords().contains(Dim("new")));
}

TEST_F(DatasetContractTest, coord_values_can_be_set) {
  ds.coords().setitem(Dim::X, var);
  ds.coords()[Dim::X].values()[0] = 17;
  EXPECT_EQ(ds.coords()[Dim::X].values()[0], 17);
}

TEST_F(DatasetContractTest, item_values_can_be_set) {
  ds["a"].data().values()[0] = 17;
  EXPECT_EQ(ds["a"].data().values()[0], 17);
}

TEST_F(DatasetContractTest, item_coord_cannot_be_added) {
  ds["a"].coords().setitem(Dim("ignored"), var);
  EXPECT_FALSE(ds["a"].coords().contains(Dim("ignored")));
}

TEST_F(DatasetContractTest, item_mask_can_be_added) {
  ds["a"].masks().setitem("mask", var);
  EXPECT_TRUE(ds["a"].masks().contains("mask"));
}