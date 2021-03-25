// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
/// @file
/// @author Simon Heybrock
#pragma once
#include <utility>

#include "scipp-core_export.h"
#include "scipp/common/index.h"

namespace scipp {
using index_pair = std::pair<scipp::index, scipp::index>;
}

namespace scipp::core {

struct bucket_base {
  using range_type = index_pair;
};
template <class T> struct bucket : bucket_base {
  using buffer_type = T;
  using element_type = T;
  using const_element_type = T;
};

template <class T> using bin = bucket<T>;

} // namespace scipp::core

namespace scipp {
using core::bucket;
} // namespace scipp
