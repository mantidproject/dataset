// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock

#include "scipp/neutron/beamline.h"
#include "scipp/core/dataset.h"

using namespace scipp::core;

namespace scipp::neutron {

auto component_positions(const Dataset &d) {
  return d.labels()["component_info"].values<Dataset>()[0]["position"].data();
}

Variable source_position(const Dataset &d) {
  // TODO Need a better mechanism to identify source and sample.
  return Variable(component_positions(d).slice({Dim::Row, 0}));
}

Variable sample_position(const Dataset &d) {
  return Variable(component_positions(d).slice({Dim::Row, 1}));
}

Variable l1(const Dataset &d) {
  return norm(sample_position(d) - source_position(d));
}

Variable l2(const Dataset &d) {
  return norm(d.coords()[Dim::Position] - sample_position(d));
}

Variable scattering_angle(const Dataset &d) { return 0.5 * two_theta(d); }

Variable two_theta(const Dataset &d) {
  auto beam = sample_position(d) - source_position(d);
  const auto l1 = norm(beam);
  beam /= l1;
  auto scattered = d.coords()[Dim::Position] - sample_position(d);
  const auto l2 = norm(scattered);
  scattered /= l2;

  return acos(dot(beam, scattered));
}

} // namespace scipp::neutron
