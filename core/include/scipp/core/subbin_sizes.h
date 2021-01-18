// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once

#include <string>
#include <vector>

#include "scipp-core_export.h"
#include "scipp/common/index.h"

namespace scipp::core {

// instead of this helper class, we could also just use two variables for
// offset and sizes (the latter one a bin-variable), and implement custom sum
// and cumsum ops.
class SCIPP_CORE_EXPORT SubbinSizes {
public:
  using container_type = std::vector<scipp::index>;
  SubbinSizes() = default;
  SubbinSizes(const scipp::index value);
  SubbinSizes(const scipp::index offset, container_type &&sizes);
  const auto &offset() const noexcept { return m_offset; }
  const auto &sizes() const noexcept { return m_sizes; }
  void operator=(const scipp::index value);
  SubbinSizes &operator+=(const SubbinSizes &other);
  SubbinSizes &operator-=(const SubbinSizes &other);

  SubbinSizes cumsum() const;
  scipp::index sum() const;
  void trim_to(const SubbinSizes &other);
  SubbinSizes &add_intersection(const SubbinSizes &other);

private:
  scipp::index m_offset{0};
  container_type m_sizes;
};

[[nodiscard]] SCIPP_CORE_EXPORT bool operator==(const SubbinSizes &a,
                                                const SubbinSizes &b);

[[nodiscard]] SCIPP_CORE_EXPORT SubbinSizes operator+(const SubbinSizes &a,
                                                      const SubbinSizes &b);

[[nodiscard]] SCIPP_CORE_EXPORT SubbinSizes operator-(const SubbinSizes &a,
                                                      const SubbinSizes &b);

[[nodiscard]] SCIPP_CORE_EXPORT std::string to_string(const SubbinSizes &s);

} // namespace scipp::core