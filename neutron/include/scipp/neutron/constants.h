// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <tuple>

#include <boost/units/systems/si/codata/electromagnetic_constants.hpp>
#include <boost/units/systems/si/codata/neutron_constants.hpp>
#include <boost/units/systems/si/codata/universal_constants.hpp>

#include "scipp/common/constants.h"

#include "scipp/units/unit.h"

#include "scipp/variable/operations.h"

#include "scipp/neutron/beamline.h"

namespace scipp::neutron::constants {

constexpr auto tof_to_s = boost::units::quantity<boost::units::si::time>(
                              1.0 * units::boost_units::us) /
                          units::boost_units::us;
constexpr auto J_to_meV =
    units::boost_units::meV / boost::units::quantity<boost::units::si::energy>(
                                  1.0 * units::boost_units::meV);
constexpr auto m_to_angstrom = units::boost_units::angstrom /
                               boost::units::quantity<boost::units::si::length>(
                                   1.0 * units::boost_units::angstrom);

// In tof-to-energy conversions we *divide* by time-of-flight (squared), so the
// tof_to_s factor is in the denominator.
constexpr auto tof_to_energy_physical_constants =
    0.5 * boost::units::si::constants::codata::m_n * J_to_meV /
    (tof_to_s * tof_to_s);

constexpr auto tof_to_dspacing_physical_constants =
    2.0 * boost::units::si::constants::codata::m_n /
    boost::units::si::constants::codata::h / (m_to_angstrom * tof_to_s);

constexpr auto tof_to_wavelength_physical_constants =
    tof_to_s * m_to_angstrom * boost::units::si::constants::codata::h /
    boost::units::si::constants::codata::m_n;

template <class T> auto tof_to_dspacing(const T &d) {
  const auto &sourcePos = source_position(d);
  const auto &samplePos = sample_position(d);

  auto beam = samplePos - sourcePos;
  const auto l1 = norm(beam);
  beam /= l1;
  auto scattered = position(d) - samplePos;
  const auto l2 = norm(scattered);
  scattered /= l2;

  // l_total = l1 + l2
  auto conversionFactor(l1 + l2);

  conversionFactor *= Variable(tof_to_dspacing_physical_constants * sqrt(0.5));
  conversionFactor *= sqrt(1.0 * units::one - dot(beam, scattered));

  reciprocal(conversionFactor, conversionFactor);
  return conversionFactor;
}

template <class T> static auto tof_to_wavelength(const T &d) {
  return Variable(tof_to_wavelength_physical_constants) / flight_path_length(d);
}

template <class T> auto tof_to_energy(const T &d) {
  if (incident_energy(d) || final_energy(d))
    throw std::runtime_error(
        "Data contains coords for incident or final energy. Conversion to "
        "energy for inelastic data not implemented yet.");
  // l_total = l1 + l2
  auto conversionFactor = flight_path_length(d);
  // l_total^2
  conversionFactor *= conversionFactor;
  conversionFactor *= Variable(tof_to_energy_physical_constants);
  return conversionFactor;
}

template <class T> auto tof_to_energy_transfer(const T &d) {
  const auto Ei = incident_energy(d);
  const auto Ef = final_energy(d);
  if (Ei && Ef)
    throw std::runtime_error(
        "Data contains coords for incident *and* final energy, cannot have "
        "both for inelastic scattering.");
  if (!Ei && !Ef)
    throw std::runtime_error(
        "Data contains neither coords for incident nor for final energy, this "
        "does not appear to be inelastic-scattering data, cannot convert to "
        "energy transfer.");
  auto l1_square = l1(d);
  l1_square *= l1_square;
  l1_square *= Variable(tof_to_energy_physical_constants);
  auto l2_square = l2(d);
  l2_square *= l2_square;
  l2_square *= Variable(tof_to_energy_physical_constants);
  if (Ei) { // Direct-inelastic.
    return std::tuple{-l2_square, sqrt(l1_square / Ei), -Ei};
  } else { // Indirect-inelastic.
    return std::tuple{std::move(l1_square), sqrt(l2_square / Ef), Variable(Ef)};
  }
}

template <class T> auto wavelength_to_q(const T &d) {
  return sin(scattering_angle(d)) * (4.0 * scipp::pi<double> * units::one);
}

} // namespace scipp::neutron::constants
