#include <gtest/gtest.h>

#include "dataset.h"

TEST(Dataset, construct_empty) {
  ASSERT_NO_THROW(Dataset d);
}

TEST(Dataset, construct) {
  // TODO different construction mechanism that does not require passing length
  // 1 vectors.
  ASSERT_NO_THROW(Dataset d(std::vector<double>(1), std::vector<int>(1)));
}

TEST(Dataset, columns) {
  Dataset d(std::vector<double>(1), std::vector<int>(1));
  ASSERT_EQ(d.columns(), 2);
}

TEST(Dataset, extendAlongDimension) {
  Dataset d(std::vector<double>(1), std::vector<int>(1));
  d.addDimension("tof", 10);
  d.extendAlongDimension(ColumnType::Doubles, "tof");
}

TEST(Dataset, get) {
  Dataset d(std::vector<double>(1), std::vector<int>(1));
  auto &view = d.get<Doubles>();
  ASSERT_EQ(view.size(), 1);
  view[0] = 1.2;
  ASSERT_EQ(view[0], 1.2);
}

TEST(Dataset, view_tracks_changes) {
  Dataset d(std::vector<double>(1), std::vector<int>(1));
  auto &view = d.get<Doubles>();
  ASSERT_EQ(view.size(), 1);
  view[0] = 1.2;
  d.addDimension("tof", 3);
  d.extendAlongDimension(ColumnType::Doubles, "tof");
  ASSERT_EQ(view.size(), 3);
  EXPECT_EQ(view[0], 1.2);
  EXPECT_EQ(view[1], 0.0);
  EXPECT_EQ(view[2], 0.0);
}

TEST(Dataset, histogram_view) {
  // bin edges, values, and errors.
  Dataset d(std::vector<double>(1), std::vector<double>(1),
            std::vector<double>(1));
  d.addDimension("spectrum", 3);
  // TODO ColumnId?
  d.extendAlongDimension(ColumnId::BinEdges, "spectrum"); // optional
  d.extendAlongDimension(ColumnId::Values, "spectrum");
  d.extendAlongDimension(ColumnId::Errors, "spectrum");
  // How to handle non-const fields that have extra dimensions? would need to
  // have extra index/stride access? but type would be different, i.e., we need
  // different client code, handling different cases!
  // Can we handle BinEdges/Values/Errors with units?
  auto &view = d.get<const BinEdges, Values, Errors>(); // provide getters/setters in view via mixins?
  // if all data in flat array, how can we distinguish iterations over
  // individual values vs. full histograms? use a special helper type
  // `Histograms` when getting the view?
  for(auto &histogram : view) {
    histogram *= 2.0;
  }

  auto &slice = d.slice("spectrum", 2); // what does this return? performance implications of converting to view after slicing?
  for(auto &slice : d.slices("spectrum")) {
  }
  auto &view = d.get<const BinEdges, Value, Error>;
  // this is maybe not so useful, item would contain extra dimensions such as polarization.
  for(auto &item : view.slice("spectrum")) { // how can this work if view does not contain spectrum axis?
    //
  }

  // similar to view?? view provides iteration over 1 or more columns, slice is access

  //auto &view = d.slice<Polarization>.get<const BinEdges, Values, Errors>();
  //auto &slice = d.slice<const BinEdges, Values, Errors>(polarization);
}
